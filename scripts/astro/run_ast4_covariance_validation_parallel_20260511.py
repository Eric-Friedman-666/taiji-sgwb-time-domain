#!/usr/bin/env python3
"""AST4 covariance-first validation for unequal-arm second-generation XYZ TDI.

This runner simulates three isolated ingredients through the Triangle
time-domain pipeline and compares empirical XYZ covariance matrices against
the deterministic response/noise-transfer tables used by the full-covariance
likelihood code.

No Bilby PE is run here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import h5py
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import windows

REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents
    if (p / "py_code").exists() and (p / "data").exists()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TRIANGLE_ROOT = Path(os.getenv("TRIANGLE_ROOT", str(REPO_ROOT / "vendor/Triangle-Simulator")))
if TRIANGLE_ROOT.exists() and str(TRIANGLE_ROOT) not in sys.path:
    sys.path.append(str(TRIANGLE_ROOT))
os.chdir(REPO_ROOT)

from py_code.parameters import L, c, unit_factor  # noqa: E402
from py_code.triangle_code import (  # noqa: E402
    C,
    DAY,
    F_LASER,
    MOSA_labels,
    SC_labels,
    TDI,
    TDIStringManipulation,
    TDISensitivity,
    TWOPI,
)

import py_code.lisa_generate as lisa_generate  # noqa: E402

CAMPAIGN = "ast4_cov_validation_20260510"
DATA_ROOT = REPO_ROOT / "data" / CAMPAIGN
LOG_ROOT = REPO_ROOT / "logs" / CAMPAIGN
FIG_ROOT = REPO_ROOT / "figure" / CAMPAIGN

CPU_PYTHON = sys.executable
CHANNELS = ("X", "Y", "Z")
ENTRIES = ("XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ")
INDEPENDENT_ENTRIES = ("XX", "XY", "XZ", "YY", "YZ", "ZZ")
EXPERIMENTS = ("acc", "oms", "response")
DAYS = {
    "day01": {
        "segment_index": 1,
        "day_start": 0.0,
        "day_end": 1.0,
        "day_midpoint": 0.5,
        "orbit_dir": "data/orbit_unequal_dayseg_20260506/day01",
    },
    "day10": {
        "segment_index": 10,
        "day_start": 9.0,
        "day_end": 10.0,
        "day_midpoint": 9.5,
        "orbit_dir": "data/orbit_unequal_dayseg_20260506/day10",
    },
}
TRUTH_N_ACC = 3.0e-15
TRUTH_DELTA_X = 8.0e-12
DEFAULT_FS = 0.4
DEFAULT_SEGMENT_DAYS = 1.0
DEFAULT_F_MIN = 3.0e-5
DEFAULT_F_MAX = 0.15
DEFAULT_NOTCH_HALF_WIDTH = 0.001
DEFAULT_N_SOURCE = 1024
DEFAULT_N_REAL = 10
DEFAULT_BASE_SEED = 20260510
DEFAULT_NSIDE = 5


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def stage_start(message: str) -> float:
    log(message)
    return time.perf_counter()


def stage_end(start: float, message: str) -> None:
    log(f"{message} in {format_seconds(time.perf_counter() - start)}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fresh_tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp_{os.getpid()}_{int(time.time())}")


def repo_rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_csv_choices(raw: str, allowed: Iterable[str], label: str) -> tuple[str, ...]:
    allowed_set = set(allowed)
    values = tuple(x.strip() for x in raw.split(",") if x.strip())
    bad = [x for x in values if x not in allowed_set]
    if bad:
        raise ValueError(f"Unsupported {label}: {bad}; allowed={sorted(allowed_set)}")
    if not values:
        raise ValueError(f"No {label} selected.")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AST4 covariance-first validation without PE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "ast4_covariance_validation_20260510.h5")
    parser.add_argument("--summary-json", type=Path, default=DATA_ROOT / "ast4_covariance_validation_summary_20260510.json")
    parser.add_argument("--summary-md", type=Path, default=LOG_ROOT / "ast4_covariance_validation_summary_20260510.md")
    parser.add_argument("--n-real", type=int, default=DEFAULT_N_REAL)
    parser.add_argument("--n-workers", type=int, default=1, help="Number of CPU worker processes for independent realizations.")
    parser.add_argument("--n-source", type=int, default=DEFAULT_N_SOURCE)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--segment-days", type=float, default=DEFAULT_SEGMENT_DAYS)
    parser.add_argument("--f-min", type=float, default=DEFAULT_F_MIN)
    parser.add_argument("--f-max", type=float, default=DEFAULT_F_MAX)
    parser.add_argument("--notch-half-width", type=float, default=DEFAULT_NOTCH_HALF_WIDTH)
    parser.add_argument("--nside", type=int, default=DEFAULT_NSIDE)
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--days", default="day01,day10")
    parser.add_argument(
        "--sim-days",
        default=None,
        help=(
            "Comma-separated days to simulate through Triangle. "
            "Defaults to --days. This can be narrower than --days when "
            "extra deterministic calculated curves are needed for plotting."
        ),
    )
    parser.add_argument("--window", default="hann", choices=("hann", "boxcar"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-simulation", action="store_true", help="Only recompute summaries from an existing HDF5 file.")
    return parser


def build_orbit_sensitivity(day_midpoint: float) -> TDISensitivity:
    orbit_time = day_midpoint * DAY
    orbit = lisa_generate.Orbit(OrbitDir="./data/OrbitData/MicroSateOrbitEclipticTCB/", pn_order=2)
    ri = {key: orbit.Positionfunctions()[key](orbit_time) for key in SC_labels}
    nij = {key: orbit.ArmVectorfunctions()[key](orbit_time) for key in MOSA_labels}
    dij = {key: orbit.LTTfunctions()[key](orbit_time) for key in MOSA_labels}
    return TDISensitivity(Ri=ri, nij=nij, dij=dij)


def xyz_strings() -> tuple[dict, dict, dict]:
    x_string = TDI.X2_string
    y_string = TDIStringManipulation.TDIStringCyc(x_string)
    z_string = TDIStringManipulation.TDIStringCyc(y_string)
    return x_string, y_string, z_string


def tdi_response_amplitudes(
    sens: TDISensitivity,
    lam: float,
    beta: float,
    frequency: np.ndarray,
    p_ij,
) -> tuple[np.ndarray, np.ndarray]:
    k = sens.wave_vector(lam=lam, beta=beta)
    ep, ec = sens.polar_basis(lam=lam, beta=beta)
    prefactor = sens.Prefactor_ij(f=frequency)
    exp_factor = sens.Exp_ij(k=k, f=frequency)
    sinc_factor = sens.Sinc_ij(k=k, f=frequency)
    fp, fc = sens.PatternFunction_ij(ep=ep, ec=ec)
    hp = np.zeros_like(frequency, dtype=np.complex128)
    hc = np.zeros_like(frequency, dtype=np.complex128)
    for key in MOSA_labels:
        tmp = p_ij[key] * prefactor[key] * exp_factor[key] * sinc_factor[key]
        hp += tmp * fp[key]
        hc += tmp * fc[key]
    return hp, hc


def calculate_response_matrix(
    frequency: np.ndarray,
    day_midpoint: float,
    *,
    n_source: int,
    seed: int,
) -> np.ndarray:
    sens = build_orbit_sensitivity(day_midpoint)
    operators = [sens.TDI_P_ij(P_ij_strings=s, f=frequency) for s in xyz_strings()]
    rng = np.random.default_rng(seed)
    response_sum = np.zeros((len(frequency), 3, 3), dtype=np.complex128)
    for _ in range(n_source):
        longitude = rng.uniform(0.0, TWOPI)
        latitude = np.arcsin(rng.uniform(-1.0, 1.0))
        hp = []
        hc = []
        for op in operators:
            hp_i, hc_i = tdi_response_amplitudes(sens, longitude, latitude, frequency, op)
            hp.append(hp_i)
            hc.append(hc_i)
        hp_arr = np.asarray(hp)
        hc_arr = np.asarray(hc)
        for i in range(3):
            for j in range(3):
                response_sum[:, i, j] += 0.5 * (
                    np.conjugate(hp_arr[i]) * hp_arr[j]
                    + np.conjugate(hc_arr[i]) * hc_arr[j]
                )
    return response_sum / float(n_source) / unit_factor(frequency)[:, None, None] * 2.0


def calculate_noise_transfer_matrices(
    frequency: np.ndarray,
    day_midpoint: float,
) -> tuple[np.ndarray, np.ndarray]:
    sens = build_orbit_sensitivity(day_midpoint)
    operators = [sens.TDI_P_ij(P_ij_strings=s, f=frequency) for s in xyz_strings()]
    sens_acc = TDISensitivity(Ri=sens.Ri, nij=sens.nij, dij=sens.dij, S_OMS=0.0, S_ACC=1.0)
    sens_oms = TDISensitivity(Ri=sens.Ri, nij=sens.nij, dij=sens.dij, S_OMS=1.0, S_ACC=0.0)
    acc = np.zeros((len(frequency), 3, 3), dtype=np.complex128)
    oms = np.zeros((len(frequency), 3, 3), dtype=np.complex128)
    for i, op_i in enumerate(operators):
        for j, op_j in enumerate(operators):
            acc[:, i, j] = sens_acc.TDI_noise_CSD(f=frequency, P_ij=op_i, Q_ij=op_j) / unit_factor(frequency)
            oms[:, i, j] = sens_oms.TDI_noise_CSD(f=frequency, P_ij=op_i, Q_ij=op_j) / unit_factor(frequency)
    return acc, oms


def build_notch_metadata(
    frequency: np.ndarray,
    day_midpoints: dict[str, float],
    *,
    f_min: float,
    f_max: float,
    half_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    in_range = (frequency >= f_min) & (frequency <= f_max)
    mask = in_range.copy()
    rows = []
    dtype = np.dtype(
        [
            ("day", "S8"),
            ("day_midpoint", np.float64),
            ("link", "S3"),
            ("n", np.int16),
            ("tau_s", np.float64),
            ("center_hz", np.float64),
            ("half_width_hz", np.float64),
        ]
    )
    for day_label, day_midpoint in day_midpoints.items():
        sens = build_orbit_sensitivity(day_midpoint)
        for link in MOSA_labels:
            tau = float(sens.dij[link])
            max_n = int(math.ceil((f_max + half_width) * 4.0 * tau)) + 1
            for n in range(1, max_n + 1):
                center = n / (4.0 * tau)
                if not (f_min - half_width <= center <= f_max + half_width):
                    continue
                mask &= ~((frequency >= center - half_width) & (frequency <= center + half_width))
                rows.append((day_label.encode("ascii"), day_midpoint, link.encode("ascii"), n, tau, center, half_width))
    centers = np.asarray(rows, dtype=dtype)
    notch_mask = in_range & ~mask
    return mask, notch_mask, centers


def generate_tdi_data_isolated(
    *,
    include_acc: bool,
    include_oms: bool,
    signal_type: str | None,
    segment_days: float,
    fsample: float,
    orbit_dir: str,
    nside: int,
) -> lisa_generate.TDI:
    """Generate one isolated TDI realization through the same Triangle path.

    This is intentionally a narrow fork of py_code.lisa_generate.generate_TDI_data
    so ACC and OMS can be switched independently for this validation.
    """
    orbit = lisa_generate.Orbit(OrbitDir=orbit_dir, pn_order=2)
    data_time = segment_days * DAY
    size = int(data_time * fsample)
    interp_order = 31
    offset = lisa_generate.OffsetFunctions(laser_flag=False, clock_flag=False)
    instrumental_noise = lisa_generate.InstrumentalPSDs(L=lisa_generate.L_nominal)

    def s_sgwb(f):
        if signal_type == "response":
            return f * 0.0 + 1.0
        raise ValueError(f"Unsupported signal_type for isolated validation: {signal_type!r}")

    tdata = np.arange(size) / fsample
    if signal_type is None:
        gw = None
    else:
        npix = lisa_generate.hlp.nside2npix(nside)
        colatitude, longitude = lisa_generate.hlp.pix2ang(nside=nside, ipix=np.arange(npix))
        latitude = np.pi / 2.0 - colatitude
        sgwb_generator = lisa_generate.GeneralNoise(s_sgwb)
        gw = []
        for i in range(npix):
            hp_k = sgwb_generator(fsample=fsample, size=size) / np.sqrt(npix)
            hc_k = sgwb_generator(fsample=fsample, size=size) / np.sqrt(npix)
            waveform = lisa_generate.GeneralWaveform(tdata=tdata, hpdata=hp_k, hcdata=hc_k)
            gw.append(
                lisa_generate.GW(
                    orbit=orbit,
                    ext_params=[longitude[i], latitude[i], 0.0],
                    GWwaveform=waveform,
                )
            )

    ifo = lisa_generate.Interferometers(
        size=size,
        fsample=fsample,
        time_frame="ProperTime",
        acc_noise=include_acc,
        ro_noise=include_oms,
        laser_noise=False,
        clock_noise=False,
        ob_noise=False,
        bl_noise=False,
        op_noise=False,
        ranging_noise=False,
        modulation_noise=False,
        orbit_class=orbit,
        offset_class=offset,
        noise_class=instrumental_noise,
        gw_class=gw,
        fplan=None,
        order=interp_order,
        pool=None,
        clean_memory=True,
    )
    ifo.SimulateInterferometers()
    measurements = ifo.OutputMeasurements(mode="decomposed")
    measurements_tcb = lisa_generate.TPStoTCB(measurements, orbit_class=orbit, order=interp_order, pool=None)
    ifo.clean()
    tdi = lisa_generate.TDI(
        measurements=measurements_tcb,
        delays=measurements_tcb["ltt"],
        fsample=measurements_tcb["fsample"],
        order=31,
    )
    tdi.CalculateXi(doppler=True, pool=None)
    tdi.CalculateEta(doppler=True)
    tdi.FastMichelson(doppler=True, channel="AET")
    return tdi


def scaled_frequency_data_from_tdi(
    tdi,
    *,
    fsample: float,
    segment_days: float,
    window: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency and scaled complex XYZ vectors.

    The returned z has shape (Nf, 3) and is scaled so that
    mean(conj(z_i) z_j) is identical to
    (T_seg fs^2 / 2) * scipy.signal.csd(x_i, x_j) / unit_factor(f).
    This is the left-conjugated CSD orientation used by the PE tables.
    """
    trim = 100
    channels = [
        np.asarray(tdi.measurements["X2"][trim:-trim], dtype=np.float64) / F_LASER,
        np.asarray(tdi.measurements["Y2"][trim:-trim], dtype=np.float64) / F_LASER,
        np.asarray(tdi.measurements["Z2"][trim:-trim], dtype=np.float64) / F_LASER,
    ]
    n_data = len(channels[0])
    if any(len(ch) != n_data for ch in channels):
        raise RuntimeError("X/Y/Z channel lengths do not match.")
    expected_n = int(round(segment_days * DAY * fsample))
    expected_fft_n = expected_n - 2 * trim
    if n_data != expected_fft_n:
        raise RuntimeError(
            f"Expected one simulated segment of {expected_n} samples with "
            f"[{trim}:-{trim}] FFT trim, i.e. {expected_fft_n} samples, got {n_data}."
        )
    if window == "hann":
        window_array = windows.hann(n_data)
    elif window == "boxcar":
        window_array = np.ones(n_data)
    else:
        raise ValueError(f"Unsupported window: {window}")
    sum_w2 = float(np.sum(window_array**2))
    frequency_full = rfftfreq(n_data, 1.0 / fsample)
    positive = np.arange(1, len(frequency_full))
    frequency = frequency_full[positive]
    one_sided = np.full(len(frequency), 2.0, dtype=np.float64)
    if n_data % 2 == 0:
        one_sided[-1] = 1.0
    scale = np.sqrt((segment_days * DAY) * fsample * one_sided / (2.0 * sum_w2))
    uf = np.sqrt(unit_factor(frequency))
    z = np.zeros((len(frequency), 3), dtype=np.complex128)
    for i, channel in enumerate(channels):
        detrended = channel - np.mean(channel)
        fft_values = rfft(detrended * window_array)[positive]
        z[:, i] = fft_values * scale / uf
    return frequency, z


def covariances_from_dtilde(dtilde: np.ndarray) -> dict[str, np.ndarray]:
    mean_z = np.mean(dtilde, axis=0)
    centered = dtilde - mean_z[None, :, :]
    cov_standard = np.einsum("rfi,rfj->fij", dtilde, np.conjugate(dtilde), optimize=True) / dtilde.shape[0]
    cov_left = np.einsum("rfi,rfj->fij", np.conjugate(dtilde), dtilde, optimize=True) / dtilde.shape[0]
    cov_standard_demeaned = (
        np.einsum("rfi,rfj->fij", centered, np.conjugate(centered), optimize=True) / dtilde.shape[0]
    )
    cov_left_demeaned = (
        np.einsum("rfi,rfj->fij", np.conjugate(centered), centered, optimize=True) / dtilde.shape[0]
    )
    return {
        "mean_d_tilde": mean_z,
        "cov_standard": cov_standard,
        "cov_left": cov_left,
        "cov_standard_demeaned": cov_standard_demeaned,
        "cov_left_demeaned": cov_left_demeaned,
    }


def experiment_config(experiment: str) -> dict[str, object]:
    if experiment == "acc":
        return {
            "include_acc": True,
            "include_oms": False,
            "signal_type": None,
            "amplitude": TRUTH_N_ACC,
            "calc_name": "acc_transfer",
        }
    if experiment == "oms":
        return {
            "include_acc": False,
            "include_oms": True,
            "signal_type": None,
            "amplitude": TRUTH_DELTA_X,
            "calc_name": "oms_transfer",
        }
    if experiment == "response":
        return {
            "include_acc": False,
            "include_oms": False,
            "signal_type": "response",
            "amplitude": 1.0,
            "calc_name": "response",
        }
    raise ValueError(f"Unsupported experiment: {experiment}")


def normalize_transfer(cov: np.ndarray, experiment: str, *, segment_days: float, fsample: float) -> np.ndarray:
    prefactor = 2.0 / (segment_days * DAY * fsample**2)
    if experiment == "acc":
        return prefactor * cov / (TRUTH_N_ACC**2)
    if experiment == "oms":
        return prefactor * cov / (TRUTH_DELTA_X**2)
    if experiment == "response":
        return prefactor * cov
    raise ValueError(experiment)


def relative_frobenius(sim: np.ndarray, calc: np.ndarray) -> np.ndarray:
    diff_norm = np.linalg.norm(sim - calc, axis=(-2, -1))
    calc_norm = np.linalg.norm(calc, axis=(-2, -1))
    return diff_norm / np.maximum(calc_norm, np.finfo(np.float64).tiny)


def hermitian_residual(matrix: np.ndarray) -> np.ndarray:
    diff = matrix - np.conjugate(np.swapaxes(matrix, -1, -2))
    return np.linalg.norm(diff, axis=(-2, -1)) / np.maximum(
        np.linalg.norm(matrix, axis=(-2, -1)),
        np.finfo(np.float64).tiny,
    )


def phase_residual(sim: np.ndarray, calc: np.ndarray, i: int, j: int) -> np.ndarray:
    return np.angle(sim[:, i, j] / np.where(np.abs(calc[:, i, j]) > 0.0, calc[:, i, j], np.nan))


def scalar_fit_residual(sim: np.ndarray, calc: np.ndarray, mask: np.ndarray) -> tuple[complex, float]:
    s = sim[mask].reshape(-1)
    c = calc[mask].reshape(-1)
    denom = np.vdot(c, c)
    if denom == 0.0:
        return np.nan + 0.0j, np.nan
    alpha = np.vdot(c, s) / denom
    scaled = sim / alpha if alpha != 0.0 else sim
    med = float(np.median(relative_frobenius(scaled, calc)[mask]))
    return complex(alpha), med


def best_permutation_residual(sim: np.ndarray, calc: np.ndarray, mask: np.ndarray) -> tuple[tuple[int, int, int], float]:
    best_perm = (0, 1, 2)
    best_med = np.inf
    for perm in itertools.permutations(range(3)):
        permuted = sim[:, perm, :][:, :, perm]
        med = float(np.median(relative_frobenius(permuted, calc)[mask]))
        if med < best_med:
            best_med = med
            best_perm = tuple(int(x) for x in perm)
    return best_perm, best_med


def entry_worst(sim: np.ndarray, calc: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    out: dict[str, object] = {}
    diag_scale = np.sqrt(
        np.maximum(np.abs(np.diagonal(calc, axis1=-2, axis2=-1)), np.finfo(np.float64).tiny)
    )
    for i, a in enumerate(CHANNELS):
        for j, b in enumerate(CHANNELS):
            denom = np.maximum(
                np.abs(calc[:, i, j]),
                1.0e-6 * diag_scale[:, i] * diag_scale[:, j],
            )
            rel = np.abs(sim[:, i, j] - calc[:, i, j]) / denom
            rel_masked = np.where(mask, rel, -np.inf)
            idx = int(np.argmax(rel_masked))
            out[a + b] = {
                "max_relative": float(rel[idx]),
                "median_relative": float(np.median(rel[mask])),
                "frequency_hz_at_max": float(np.asarray(np.nonzero(mask)[0]).size and np.arange(len(rel))[idx]),
            }
            out[a + b]["frequency_hz_at_max"] = None
    return out


def build_summary_for_group(
    frequency: np.ndarray,
    primary_mask: np.ndarray,
    sim: np.ndarray,
    sim_standard: np.ndarray,
    calc: np.ndarray,
) -> dict[str, object]:
    frob = relative_frobenius(sim, calc)
    frob_standard = relative_frobenius(sim_standard, calc)
    herm_sim = hermitian_residual(sim)
    herm_calc = hermitian_residual(calc)
    outside = primary_mask & np.isfinite(frob)
    if not np.any(outside):
        raise RuntimeError("Primary mask removed all bins.")

    entry_stats: dict[str, object] = {}
    diag_scale = np.sqrt(
        np.maximum(np.abs(np.diagonal(calc, axis1=-2, axis2=-1)), np.finfo(np.float64).tiny)
    )
    worst_entry = None
    worst_value = -np.inf
    worst_frequency = None
    for i, a in enumerate(CHANNELS):
        for j, b in enumerate(CHANNELS):
            denom = np.maximum(
                np.abs(calc[:, i, j]),
                1.0e-6 * diag_scale[:, i] * diag_scale[:, j],
            )
            rel = np.abs(sim[:, i, j] - calc[:, i, j]) / denom
            rel_masked = np.where(outside, rel, -np.inf)
            idx = int(np.argmax(rel_masked))
            value = float(rel[idx])
            entry = a + b
            entry_stats[entry] = {
                "max_relative": value,
                "median_relative": float(np.median(rel[outside])),
                "frequency_hz_at_max": float(frequency[idx]),
                "sim_abs_at_max": float(abs(sim[idx, i, j])),
                "calc_abs_at_max": float(abs(calc[idx, i, j])),
            }
            if value > worst_value:
                worst_value = value
                worst_entry = entry
                worst_frequency = float(frequency[idx])

    alpha, scaled_median = scalar_fit_residual(sim, calc, outside)
    best_perm, best_perm_median = best_permutation_residual(sim, calc, outside)
    median_frob = float(np.median(frob[outside]))
    max_frob = float(np.max(frob[outside]))
    median_standard = float(np.median(frob_standard[outside]))

    diagnosis = "stochastic_precision_or_model_detail"
    if median_standard < 0.7 * median_frob:
        diagnosis = "conjugation_or_csd_orientation"
    elif np.isfinite(scaled_median) and scaled_median < 0.5 * median_frob and abs(abs(alpha) - 1.0) > 0.2:
        diagnosis = "normalization"
    elif best_perm != (0, 1, 2) and best_perm_median < 0.7 * median_frob:
        diagnosis = "channel_ordering"
    elif median_frob < 0.75:
        diagnosis = "consistent_with_finite_realization_precision"

    passes = bool(median_frob < 0.75 and np.nanmedian(herm_sim[outside]) < 1.0e-10)
    return {
        "median_frobenius_residual": median_frob,
        "max_frobenius_residual": max_frob,
        "median_frobenius_residual_standard_ddagger_vs_calc": median_standard,
        "max_hermitian_residual_sim": float(np.max(herm_sim[outside])),
        "median_hermitian_residual_sim": float(np.median(herm_sim[outside])),
        "max_hermitian_residual_calc": float(np.max(herm_calc[outside])),
        "worst_entry": worst_entry,
        "worst_entry_frequency_hz": worst_frequency,
        "entry_stats": entry_stats,
        "best_complex_scale_sim_over_calc": {"real": float(np.real(alpha)), "imag": float(np.imag(alpha)), "abs": float(abs(alpha))},
        "median_residual_after_best_complex_scale": float(scaled_median),
        "best_channel_permutation": best_perm,
        "median_residual_best_channel_permutation": float(best_perm_median),
        "diagnosis": diagnosis,
        "passes_enough_to_trust_tables": passes,
    }


def write_summary_files(summary: dict[str, object], summary_json: Path, summary_md: Path) -> None:
    ensure_parent(summary_json)
    ensure_parent(summary_md)
    with summary_json.open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    lines = [
        "# AST4 covariance validation summary",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"HDF5: `{summary['hdf5_path']}`",
        "",
        "Primary residuals compare the PE-table left-conjugated CSD orientation,",
        "`matrix[i,j] = <d_i^* d_j>`, against the deterministic tables.",
        "The literal `d d^dagger` covariance is also stored and summarized as a",
        "conjugation/CSD convention check.",
        "",
        "| experiment | day | median Frobenius | max Frobenius | max Hermitian | worst entry | diagnosis | pass |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for exp in EXPERIMENTS:
        if exp not in summary["results"]:
            continue
        for day in DAYS:
            if day not in summary["results"][exp]:
                continue
            item = summary["results"][exp][day]
            lines.append(
                "| {exp} | {day} | {med:.6g} | {mx:.6g} | {herm:.3e} | {worst} | {diag} | {passed} |".format(
                    exp=exp,
                    day=day,
                    med=item["median_frobenius_residual"],
                    mx=item["max_frobenius_residual"],
                    herm=item["max_hermitian_residual_sim"],
                    worst=item["worst_entry"],
                    diag=item["diagnosis"],
                    passed="yes" if item["passes_enough_to_trust_tables"] else "no",
                )
            )
    lines.extend(
        [
            "",
            "Notch convention: remove frequencies within +/-0.001 Hz of each",
            "second-generation TDI zero-point center `f_n = n/(4 tau_ij)` for",
            "all six unequal arms at the day-1 and day-10 midpoint orbit times.",
            "The total intended removed width per center is 0.002 Hz.",
            "",
        ]
    )
    with summary_md.open("w") as fp:
        fp.write("\n".join(lines))


def simulate_one_realization_worker(payload: tuple) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Worker process for one independent simulator realization."""
    (
        r,
        seed,
        experiment,
        day_info,
        segment_days,
        fsample,
        nside,
        window,
    ) = payload
    config = experiment_config(experiment)
    np.random.seed(int(seed))
    tdi = generate_tdi_data_isolated(
        include_acc=bool(config["include_acc"]),
        include_oms=bool(config["include_oms"]),
        signal_type=config["signal_type"],
        segment_days=segment_days,
        fsample=fsample,
        orbit_dir=str(REPO_ROOT / str(day_info["orbit_dir"])),
        nside=nside,
    )
    try:
        frequency, dtilde = scaled_frequency_data_from_tdi(
            tdi,
            fsample=fsample,
            segment_days=segment_days,
            window=window,
        )
    finally:
        del tdi
    return int(r), int(seed), frequency, dtilde


def simulate_experiment_day(
    h5: h5py.File,
    *,
    experiment: str,
    day_label: str,
    day_info: dict[str, object],
    args: argparse.Namespace,
    frequency_ref: np.ndarray | None,
) -> np.ndarray:
    config = experiment_config(experiment)
    day_group = h5.require_group(experiment).require_group(day_label)
    seed0 = int(args.base_seed + 100000 * EXPERIMENTS.index(experiment) + 1000 * int(day_info["segment_index"]))
    seeds = np.asarray([seed0 + r for r in range(args.n_real)], dtype=np.int64)
    dtilde_list = []
    start = stage_start(
        f"Simulating {experiment}/{day_label}: n_real={args.n_real}, n_workers={args.n_workers}, orbit={day_info['orbit_dir']}"
    )
    worker_count = max(1, min(int(args.n_workers), int(args.n_real)))
    if worker_count == 1:
        for r, seed in enumerate(seeds):
            seg_start = stage_start(f"{experiment}/{day_label} realization {r + 1:02d}/{args.n_real}, seed={int(seed)}")
            _, _, frequency, dtilde = simulate_one_realization_worker(
                (r, int(seed), experiment, day_info, args.segment_days, args.fs, args.nside, args.window)
            )
            if frequency_ref is None:
                frequency_ref = frequency
            elif not np.array_equal(frequency_ref, frequency):
                raise RuntimeError(f"Frequency grid mismatch in {experiment}/{day_label}.")
            dtilde_list.append(dtilde)
            stage_end(seg_start, f"Finished {experiment}/{day_label} realization {r + 1:02d}")
    else:
        dtilde_list = [None] * int(args.n_real)
        payloads = [
            (r, int(seed), experiment, day_info, args.segment_days, args.fs, args.nside, args.window)
            for r, seed in enumerate(seeds)
        ]
        completed = 0
        log(f"Launching {worker_count} CPU workers for {experiment}/{day_label}.")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(simulate_one_realization_worker, payload): payload[0] for payload in payloads}
            for future in as_completed(future_map):
                r, seed, frequency, dtilde = future.result()
                if frequency_ref is None:
                    frequency_ref = frequency
                elif not np.array_equal(frequency_ref, frequency):
                    raise RuntimeError(f"Frequency grid mismatch in {experiment}/{day_label}.")
                dtilde_list[r] = dtilde
                completed += 1
                log(f"Finished {experiment}/{day_label} realization {r + 1:02d}/{args.n_real}, seed={seed}; completed {completed}/{args.n_real}")
        if any(item is None for item in dtilde_list):
            raise RuntimeError(f"Missing completed realizations for {experiment}/{day_label}.")
    dtilde_arr = np.asarray(dtilde_list, dtype=np.complex128)
    covs = covariances_from_dtilde(dtilde_arr)
    cov_left = covs["cov_left"]
    cov_left_demeaned = covs["cov_left_demeaned"]
    cov_standard = covs["cov_standard"]
    cov_standard_demeaned = covs["cov_standard_demeaned"]
    transfer_left = normalize_transfer(cov_left, experiment, segment_days=args.segment_days, fsample=args.fs)
    transfer_left_demeaned = normalize_transfer(cov_left_demeaned, experiment, segment_days=args.segment_days, fsample=args.fs)
    transfer_standard = normalize_transfer(cov_standard, experiment, segment_days=args.segment_days, fsample=args.fs)

    for name in list(day_group.keys()):
        del day_group[name]
    day_group.attrs["day_label"] = day_label
    day_group.attrs["data_interval_start_days"] = float(day_info["day_start"])
    day_group.attrs["data_interval_end_days"] = float(day_info["day_end"])
    day_group.attrs["data_interval_start_seconds"] = float(day_info["day_start"]) * DAY
    day_group.attrs["data_interval_end_seconds"] = float(day_info["day_end"]) * DAY
    day_group.attrs["data_duration_seconds"] = float(args.segment_days * DAY)
    day_group.attrs["day_midpoint_days"] = float(day_info["day_midpoint"])
    day_group.attrs["orbit_timestamp_seconds"] = float(day_info["day_midpoint"]) * DAY
    day_group.attrs["orbit_dir"] = str(day_info["orbit_dir"])
    day_group.attrs["orientation"] = "primary cov_sim is left-conjugated CSD: matrix[i,j] = <d_i^* d_j>"
    day_group.attrs["standard_covariance_dataset"] = "cov_sim_standard stores literal d d^dagger"
    day_group.attrs["normalization"] = "transfer_sim = 2*cov_sim/(T_seg*fs^2*amplitude^2) except response amplitude=1"
    day_group.create_dataset("realization_seeds", data=seeds)
    day_group.create_dataset("d_tilde", data=dtilde_arr, compression="gzip", compression_opts=2)
    day_group.create_dataset("mean_d_tilde", data=covs["mean_d_tilde"], compression="gzip", compression_opts=2)
    day_group.create_dataset("cov_sim", data=cov_left, compression="gzip", compression_opts=2)
    day_group.create_dataset("cov_sim_demeaned", data=cov_left_demeaned, compression="gzip", compression_opts=2)
    day_group.create_dataset("cov_sim_standard", data=cov_standard, compression="gzip", compression_opts=2)
    day_group.create_dataset("cov_sim_standard_demeaned", data=cov_standard_demeaned, compression="gzip", compression_opts=2)
    day_group.create_dataset("transfer_sim", data=transfer_left, compression="gzip", compression_opts=2)
    day_group.create_dataset("transfer_sim_demeaned", data=transfer_left_demeaned, compression="gzip", compression_opts=2)
    day_group.create_dataset("transfer_sim_standard", data=transfer_standard, compression="gzip", compression_opts=2)
    stage_end(start, f"Stored {experiment}/{day_label}")
    return frequency_ref


def write_root_metadata(
    h5: h5py.File,
    *,
    args: argparse.Namespace,
    frequency: np.ndarray,
    primary_mask: np.ndarray,
    notch_mask: np.ndarray,
    notch_centers: np.ndarray,
    selected_experiments: tuple[str, ...],
    selected_days: tuple[str, ...],
) -> None:
    for name in ("frequency", "mask", "notch_mask", "notch_centers", "day_label", "day_midpoint_days"):
        if name in h5:
            del h5[name]
    h5.attrs["description"] = "AST4 covariance-first simulator-vs-table validation"
    h5.attrs["campaign"] = CAMPAIGN
    h5.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    h5.attrs["script_path"] = repo_rel(Path(__file__))
    h5.attrs["script_sha256"] = sha256_file(Path(__file__).resolve())
    h5.attrs["cpu_python"] = CPU_PYTHON
    h5.attrs["tdi_generation"] = 2
    h5.attrs["channels"] = ",".join(CHANNELS)
    h5.attrs["fs"] = float(args.fs)
    h5.attrs["T_seg"] = float(args.segment_days * DAY)
    h5.attrs["data_duration_seconds_per_segment"] = float(args.segment_days * DAY)
    h5.attrs["day01_data_interval"] = "0-1 day = 0-86400 s"
    h5.attrs["day10_data_interval"] = "9-10 day = 777600-864000 s"
    h5.attrs["fft_input_trim"] = "[100:-100] after simulating the full one-day X2/Y2/Z2 arrays; kept to reduce edge/spectral leakage"
    h5.attrs["segment_days"] = float(args.segment_days)
    h5.attrs["N_real"] = int(args.n_real)
    h5.attrs["N_workers"] = int(args.n_workers)
    h5.attrs["N_source_calc_response"] = int(args.n_source)
    h5.attrs["nside_response_simulator"] = int(args.nside)
    h5.attrs["N_acc"] = TRUTH_N_ACC
    h5.attrs["delta_x"] = TRUTH_DELTA_X
    h5.attrs["f_min"] = float(args.f_min)
    h5.attrs["f_max"] = float(args.f_max)
    h5.attrs["notch_center_convention"] = "unequal-arm per-link f_n = n/(4*tau_ij) = n*c/(4L_ij)"
    h5.attrs["notch_half_width_hz"] = float(args.notch_half_width)
    h5.attrs["notch_total_removed_width_hz"] = float(2.0 * args.notch_half_width)
    h5.attrs["window"] = str(args.window)
    h5.attrs["selected_experiments"] = ",".join(selected_experiments)
    h5.attrs["selected_days"] = ",".join(selected_days)
    h5.attrs["orientation"] = (
        "Calculated tables and primary cov_sim/transfer_sim use left-conjugated CSD "
        "matrix[i,j]=<d_i^* d_j>; cov_sim_standard stores literal d d^dagger."
    )
    h5.create_dataset("frequency", data=frequency)
    h5.create_dataset("mask", data=primary_mask)
    h5.create_dataset("notch_mask", data=notch_mask)
    h5.create_dataset("notch_centers", data=notch_centers)
    h5.create_dataset("day_label", data=np.asarray(list(DAYS.keys()), dtype="S8"))
    h5.create_dataset("day_start_days", data=np.asarray([DAYS[d]["day_start"] for d in DAYS], dtype=np.float64))
    h5.create_dataset("day_end_days", data=np.asarray([DAYS[d]["day_end"] for d in DAYS], dtype=np.float64))
    h5.create_dataset("day_midpoint_days", data=np.asarray([DAYS[d]["day_midpoint"] for d in DAYS], dtype=np.float64))


def compute_and_store_calc_tables(
    h5: h5py.File,
    *,
    frequency: np.ndarray,
    selected_days: tuple[str, ...],
    args: argparse.Namespace,
) -> None:
    if "calc" in h5:
        del h5["calc"]
    calc = h5.create_group("calc")
    calc.attrs["orientation"] = "left-conjugated CSD/model covariance: matrix[i,j]=<d_i^* d_j>"
    calc.attrs["tdi_generation"] = 2
    calc.attrs["response_n_source"] = int(args.n_source)
    calc.create_dataset("day_label", data=np.asarray(selected_days, dtype="S8"))
    calc.create_dataset("day_midpoint_days", data=np.asarray([DAYS[d]["day_midpoint"] for d in selected_days], dtype=np.float64))
    acc_table = []
    oms_table = []
    response_table = []
    start = stage_start(f"Computing deterministic calc tables: days={selected_days}, bins={len(frequency)}")
    for day_label in selected_days:
        day_midpoint = float(DAYS[day_label]["day_midpoint"])
        seg_start = stage_start(f"Calc tables {day_label} at t0+{day_midpoint} days")
        acc, oms = calculate_noise_transfer_matrices(frequency, day_midpoint)
        response = calculate_response_matrix(
            frequency,
            day_midpoint,
            n_source=args.n_source,
            seed=args.base_seed + 900000 + int(DAYS[day_label]["segment_index"]),
        )
        acc_table.append(acc)
        oms_table.append(oms)
        response_table.append(response)
        stage_end(seg_start, f"Finished calc tables {day_label}")
    calc.create_dataset("acc_transfer", data=np.asarray(acc_table), compression="gzip", compression_opts=2)
    calc.create_dataset("oms_transfer", data=np.asarray(oms_table), compression="gzip", compression_opts=2)
    calc.create_dataset("response", data=np.asarray(response_table), compression="gzip", compression_opts=2)
    stage_end(start, "Stored deterministic calc tables")


def build_summary(h5_path: Path, summary_json: Path, summary_md: Path) -> dict[str, object]:
    with h5py.File(h5_path, "r+") as h5:
        frequency = np.asarray(h5["frequency"][:], dtype=np.float64)
        mask = np.asarray(h5["mask"][:], dtype=bool)
        calc_group = h5["calc"]
        calc_days = [x.decode("ascii") for x in calc_group["day_label"][:]]
        summary: dict[str, object] = {
            "hdf5_path": repo_rel(h5_path),
            "frequency_bins_total": int(frequency.size),
            "frequency_bins_primary": int(np.sum(mask)),
            "f_min": float(h5.attrs["f_min"]),
            "f_max": float(h5.attrs["f_max"]),
            "notch_half_width_hz": float(h5.attrs["notch_half_width_hz"]),
            "notch_total_removed_width_hz": float(h5.attrs["notch_total_removed_width_hz"]),
            "orientation": str(h5.attrs["orientation"]),
            "results": {},
        }
        if "summary" in h5:
            del h5["summary"]
        h5_summary = h5.create_group("summary")
        h5_summary.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        for exp in EXPERIMENTS:
            if exp not in h5:
                continue
            summary["results"][exp] = {}
            exp_summary = h5_summary.create_group(exp)
            calc_name = str(experiment_config(exp)["calc_name"])
            for day_label in DAYS:
                if day_label not in h5[exp]:
                    continue
                day_idx = calc_days.index(day_label)
                sim = np.asarray(h5[f"{exp}/{day_label}/transfer_sim"][:], dtype=np.complex128)
                sim_standard = np.asarray(h5[f"{exp}/{day_label}/transfer_sim_standard"][:], dtype=np.complex128)
                calc = np.asarray(calc_group[calc_name][day_idx], dtype=np.complex128)
                item = build_summary_for_group(frequency, mask, sim, sim_standard, calc)
                summary["results"][exp][day_label] = item
                day_sum = exp_summary.create_group(day_label)
                day_sum.attrs["median_frobenius_residual"] = item["median_frobenius_residual"]
                day_sum.attrs["max_frobenius_residual"] = item["max_frobenius_residual"]
                day_sum.attrs["max_hermitian_residual_sim"] = item["max_hermitian_residual_sim"]
                day_sum.attrs["worst_entry"] = item["worst_entry"]
                day_sum.attrs["diagnosis"] = item["diagnosis"]
                day_sum.attrs["passes_enough_to_trust_tables"] = item["passes_enough_to_trust_tables"]
                day_sum.create_dataset("frobenius_residual", data=relative_frobenius(sim, calc))
                day_sum.create_dataset("hermitian_residual_sim", data=hermitian_residual(sim))
                for pair, (i, j) in {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}.items():
                    day_sum.create_dataset(f"phase_residual_{pair}", data=phase_residual(sim, calc, i, j))
        h5.flush()
    write_summary_files(summary, summary_json, summary_md)
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    selected_experiments = parse_csv_choices(args.experiments, EXPERIMENTS, "experiments")
    selected_days = parse_csv_choices(args.days, DAYS.keys(), "days")
    selected_sim_days = (
        selected_days
        if args.sim_days is None
        else parse_csv_choices(args.sim_days, DAYS.keys(), "sim-days")
    )
    for day in selected_sim_days:
        if day not in selected_days:
            raise ValueError(f"sim day {day!r} must also be listed in --days so its calc table is available")
    ensure_parent(args.output)
    ensure_parent(args.summary_json)
    ensure_parent(args.summary_md)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    if args.skip_simulation:
        if not args.output.exists():
            raise FileNotFoundError(args.output)
        return build_summary(args.output, args.summary_json, args.summary_md)

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output without --overwrite: {args.output}")
    tmp = fresh_tmp_path(args.output)
    if tmp.exists():
        tmp.unlink()

    frequency_ref = None
    with h5py.File(tmp, "w") as h5:
        h5.attrs["status"] = "simulation_incomplete"
        for exp in selected_experiments:
            for day_label in selected_sim_days:
                frequency_ref = simulate_experiment_day(
                    h5,
                    experiment=exp,
                    day_label=day_label,
                    day_info=DAYS[day_label],
                    args=args,
                    frequency_ref=frequency_ref,
                )
                h5.flush()
        assert frequency_ref is not None
        day_midpoints = {day: float(DAYS[day]["day_midpoint"]) for day in selected_days}
        primary_mask, notch_mask, notch_centers = build_notch_metadata(
            frequency_ref,
            day_midpoints,
            f_min=args.f_min,
            f_max=args.f_max,
            half_width=args.notch_half_width,
        )
        write_root_metadata(
            h5,
            args=args,
            frequency=frequency_ref,
            primary_mask=primary_mask,
            notch_mask=notch_mask,
            notch_centers=notch_centers,
            selected_experiments=selected_experiments,
            selected_days=selected_days,
        )
        h5.attrs["selected_sim_days"] = ",".join(selected_sim_days)
        compute_and_store_calc_tables(h5, frequency=frequency_ref, selected_days=selected_days, args=args)
        h5.attrs["status"] = "simulation_complete"
    tmp.replace(args.output)
    log(f"Saved validation HDF5: {args.output}")
    return build_summary(args.output, args.summary_json, args.summary_md)


def main() -> None:
    args = build_arg_parser().parse_args()
    start = stage_start("Starting AST4 covariance validation")
    log(f"Repository: {REPO_ROOT}")
    log(f"Output: {args.output}")
    summary = run(args)
    stage_end(start, "AST4 covariance validation finished")
    compact = {
        exp: {
            day: {
                "median": item["median_frobenius_residual"],
                "max": item["max_frobenius_residual"],
                "pass": item["passes_enough_to_trust_tables"],
                "diagnosis": item["diagnosis"],
            }
            for day, item in days.items()
        }
        for exp, days in summary["results"].items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
