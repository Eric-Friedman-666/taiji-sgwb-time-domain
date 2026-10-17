#!/usr/bin/env python3
"""Four-parameter unequal-arm XYZ full-covariance PE prototype.

This script is intentionally separate from the formal AET production chain.
It tests the same one-year, 36 x 10-day, second-generation TDI setup used by
the current paper, but stores and fits the full non-diagonal XYZ covariance.

Typical use:
  MODE=generate python scripts/astro/xyz4_unequal_fullcov_pe.py
  CUDA_VISIBLE_DEVICES=0 MODE=pe python scripts/astro/xyz4_unequal_fullcov_pe.py
  MODE=check CHECK_NF=64 python scripts/astro/xyz4_direct_fullcov_pe_debug.py
"""

from __future__ import annotations

from pathlib import Path
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import h5py
import numpy as np
from tqdm import tqdm


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

from py_code.parameters import (  # noqa: E402
    H0,
    L,
    c,
    input_parameters,
    para_range,
    unit_factor,
)
from py_code.sample import Likelihood, get_priors  # noqa: E402
from py_code.triangle_code import (  # noqa: E402
    DAY,
    F_LASER,
    MOSA_labels,
    Orbit,
    SC_labels,
    TDI,
    TDIStringManipulation,
    TDISensitivity,
    TWOPI,
)

def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


MODE = os.getenv("MODE", "check").strip().lower()
DATA_PATH = Path(os.getenv("DATA_PATH", "./data/ast_xyz_fullcov_1yr.h5"))
TABLE_PATH = Path(os.getenv("TABLE_PATH", "./data/xyz4_fullcov_tables_unequal_1yr.h5"))
RESULT_PATH = Path(os.getenv("RESULT_PATH", "")) if os.getenv("RESULT_PATH") else None
RESULT_GROUP = os.getenv("RESULT_GROUP", "fullcov_xyz_pe")
MODEL = "noise+ast"
SEGMENTS = int(os.getenv("SEGMENTS", "36"))
SEGMENT_DAYS = float(os.getenv("SEGMENT_DAYS", "10"))
FSAMPLE = float(os.getenv("FS", "0.4"))
T_SEG = SEGMENT_DAYS * DAY
OBSERVATION_DAYS = SEGMENTS * SEGMENT_DAYS
F_RANGE = (float(os.getenv("F_MIN", "3e-5")), float(os.getenv("F_MAX", "0.15")))
NOTCH_WIDTH = float(os.getenv("NOTCH_WIDTH", "0.001"))
GENERATE_ALL_FREQUENCIES = env_bool("GENERATE_ALL_FREQUENCIES", False)
PE_F_MIN = os.getenv("PE_F_MIN")
PE_F_MAX = os.getenv("PE_F_MAX")
PE_NOTCH_WIDTH = os.getenv("PE_NOTCH_WIDTH")
PE_F_RANGE = (
    float(PE_F_MIN) if PE_F_MIN is not None else None,
    float(PE_F_MAX) if PE_F_MAX is not None else None,
)
PE_NOTCH_WIDTH_VALUE = float(PE_NOTCH_WIDTH) if PE_NOTCH_WIDTH is not None else None
N_SOURCE = int(os.getenv("N_SOURCE", str(2**10)))
BASE_SEED = int(os.getenv("SEED", "1234"))
SAMPLE_NUMBER = int(float(os.getenv("SAMPLE_NUMBER", "1000")))
DLOGZ = float(os.getenv("DLOGZ", "0.1"))
WALKS = int(os.getenv("WALKS", "64"))
LABEL = os.getenv("LABEL", "xyz4_direct_fullcov_raw_response")
RESPONSE_CONVENTION = os.getenv("RESPONSE_CONVENTION", "as_is").strip().lower()
ORBIT_DIR_TEMPLATE = os.getenv("ORBIT_DIR_TEMPLATE", "")
DIAGNOSTIC_THETA = np.asarray(
    [
        float(x)
        for x in os.getenv("DIAGNOSTIC_THETA", "2.3366,6.4969,-10.4690,3.0").split(",")
    ],
    dtype=np.float64,
)
REGENERATE = env_bool("REGENERATE", False)
RECALCULATE_TABLES = env_bool("RECALCULATE_TABLES", False)
STRICT_TABLE_CACHE = env_bool("STRICT_TABLE_CACHE", False)
CHECK_NF = int(os.getenv("CHECK_NF", "64"))
NOISE_AVERAGE_DAY_OFFSETS = tuple(
    float(x)
    for x in os.getenv("NOISE_AVERAGE_DAY_OFFSETS", "").replace(" ", "").split(",")
    if x
)
RESPONSE_AVERAGE_DAY_OFFSETS = tuple(
    float(x)
    for x in os.getenv("RESPONSE_AVERAGE_DAY_OFFSETS", "").replace(" ", "").split(",")
    if x
)
if MODE == "check" and "TABLE_PATH" not in os.environ:
    TABLE_PATH = Path(f"./data/xyz4_fullcov_tables_unequal_1yr_check_nf{CHECK_NF}.h5")

CHANNELS = ("X", "Y", "Z")


def orbit_dir_for_segment(segment: int) -> str:
    if ORBIT_DIR_TEMPLATE:
        return ORBIT_DIR_TEMPLATE.format(segment=segment, segment1=segment + 1)
    return str(REPO_ROOT / f"data/orbit_unequal_360day/seg{segment + 1}")


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def log_step(message: str) -> float:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)
    return time.perf_counter()


def finish_step(start: float, message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message} in {format_seconds(time.perf_counter() - start)}", flush=True)


def build_frequency_mask(frequency: np.ndarray) -> np.ndarray:
    mask = (frequency >= F_RANGE[0]) & (frequency <= F_RANGE[1])
    for idx in range(1, 100):
        center = 0.025 * idx
        mask &= ~((frequency >= center - NOTCH_WIDTH) & (frequency <= center + NOTCH_WIDTH))
    return mask


def build_generation_mask(frequency: np.ndarray) -> np.ndarray:
    if GENERATE_ALL_FREQUENCIES:
        return frequency > 0.0
    return build_frequency_mask(frequency)


def build_pe_frequency_mask(frequency: np.ndarray) -> np.ndarray:
    mask = np.ones_like(frequency, dtype=bool)
    if PE_F_RANGE[0] is not None:
        mask &= frequency >= PE_F_RANGE[0]
    if PE_F_RANGE[1] is not None:
        mask &= frequency <= PE_F_RANGE[1]
    if PE_NOTCH_WIDTH_VALUE is not None:
        for idx in range(1, 100):
            center = 0.025 * idx
            mask &= ~(
                (frequency >= center - PE_NOTCH_WIDTH_VALUE)
                & (frequency <= center + PE_NOTCH_WIDTH_VALUE)
            )
    return mask


def pl_spectrum(frequency, log_omega, alpha):
    return 10.0**log_omega * (frequency / 1.0e-3) ** alpha


def s_h_ast_np(frequency: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    omega = pl_spectrum(frequency, parameters[2], parameters[3])
    return 3.0 * H0**2 / (4.0 * np.pi**2) * omega / frequency**3


def calculate_xyz_csd_single_segment(
    x_time: np.ndarray,
    y_time: np.ndarray,
    z_time: np.ndarray,
    fsample: float,
    *,
    window_type: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return frequency and left-conjugated CSD matrix Q_star[i,j].

    The orientation is Q_star[i,j] = factor * <d_i^* d_j>/unit_factor.
    This is the transpose of the usual d d^\dagger outer-product covariance.
    """
    from Triangle.FFTTools import CSD_window

    channels = [
        np.asarray(x_time[100:-100], dtype=np.float64) / F_LASER,
        np.asarray(y_time[100:-100], dtype=np.float64) / F_LASER,
        np.asarray(z_time[100:-100], dtype=np.float64) / F_LASER,
    ]
    factor = T_SEG * fsample**2 / 2.0
    frequency_ref = None
    matrix = None
    for i in range(3):
        for j in range(3):
            frequency, csd = CSD_window(
                channels[i],
                channels[j],
                fsample=fsample,
                nbin=1,
                window_type=window_type,
            )
            if frequency_ref is None:
                frequency_ref = frequency
                matrix = np.zeros((len(frequency), 3, 3), dtype=np.complex128)
            elif not np.allclose(frequency, frequency_ref):
                raise RuntimeError("Frequency grid mismatch while building XYZ CSD matrix.")
            matrix[:, i, j] = factor * csd / unit_factor(frequency)
    return frequency_ref, matrix


def generate_xyz_fullcov_data() -> None:
    from py_code.lisa_generate import generate_TDI_data

    if DATA_PATH.exists() and not REGENERATE:
        print(f"Data already exists: {DATA_PATH}. Set REGENERATE=1 to overwrite.", flush=True)
        return

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    parameters_true = np.asarray(input_parameters(model=MODEL), dtype=np.float64)
    tmp_path = DATA_PATH.with_suffix(DATA_PATH.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    start = log_step(
        f"Generating one-year unequal-arm XYZ data: segments={SEGMENTS}, segment_days={SEGMENT_DAYS}, "
        f"fs={FSAMPLE}, all_frequencies={GENERATE_ALL_FREQUENCIES}"
    )
    obs_segments = []
    frequency_filtered = None
    mask = None

    for seg in tqdm(range(SEGMENTS), desc="Generating Triangle segments"):
        orbit_dir = orbit_dir_for_segment(seg)
        tdi = generate_TDI_data(
            noise=True,
            signal_type=MODEL,
            day_number=SEGMENT_DAYS,
            fsample=FSAMPLE,
            orbit_type="unequal",
            orbit_dir=orbit_dir,
            parameters=parameters_true,
        )
        frequency, matrix = calculate_xyz_csd_single_segment(
            tdi.measurements["X2"],
            tdi.measurements["Y2"],
            tdi.measurements["Z2"],
            FSAMPLE,
            window_type="hann",
        )
        if mask is None:
            mask = build_generation_mask(frequency)
            frequency_filtered = np.asarray(frequency[mask], dtype=np.float64)
            print(f"Filtered frequency bins: {len(frequency_filtered)} / {len(frequency)}", flush=True)
        obs_segments.append(matrix[mask])
        del tdi

    obs_csd_star = np.asarray(obs_segments, dtype=np.complex128)
    with h5py.File(tmp_path, "w") as h5:
        h5.attrs["description"] = "Unequal-arm 2nd TDI XYZ full covariance data for 4-parameter noise+ast PE"
        h5.attrs["orientation"] = (
            "left-conjugated CSD: matrix[i,j] = <d_i^* d_j>; "
            "trace(inv(P_star) @ obs_csd_star) equals d^dagger P^{-1} d"
        )
        h5.attrs["channels"] = ",".join(CHANNELS)
        h5.create_dataset("frequency", data=frequency_filtered)
        h5.create_dataset("obs_csd_star", data=obs_csd_star, compression="gzip", compression_opts=2)
        params = h5.create_group("parameters")
        params.create_dataset("parameters_true", data=parameters_true)
        params.create_dataset("segments", data=SEGMENTS)
        params.create_dataset("segments_duration_days", data=SEGMENT_DAYS)
        params.create_dataset("fsample", data=FSAMPLE)
        params.create_dataset("observation_days", data=OBSERVATION_DAYS)
        params.attrs["model"] = MODEL
    tmp_path.replace(DATA_PATH)
    finish_step(start, f"Saved XYZ full-covariance data to {DATA_PATH}")


def build_orbit_sensitivity(seg_index: int, day_offset: float = 5.0) -> TDISensitivity:
    orbit_time = (SEGMENT_DAYS * (seg_index - 1) + day_offset) * DAY
    orbit = Orbit(OrbitDir="./data/OrbitData/MicroSateOrbitEclipticTCB/", pn_order=2)
    ri = {key: orbit.Positionfunctions()[key](orbit_time) for key in SC_labels}
    nij = {key: orbit.ArmVectorfunctions()[key](orbit_time) for key in MOSA_labels}
    dij = {key: orbit.LTTfunctions()[key](orbit_time) for key in MOSA_labels}
    return TDISensitivity(Ri=ri, nij=nij, dij=dij)


def xyz_strings():
    x_string = TDI.X2_string
    y_string = TDIStringManipulation.TDIStringCyc(x_string)
    z_string = TDIStringManipulation.TDIStringCyc(y_string)
    return x_string, y_string, z_string


def calculate_xyz_noise_transfer_for_segment(
    frequency: np.ndarray,
    seg_index: int,
    *,
    day_offsets: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    strings = xyz_strings()
    acc = np.zeros((len(frequency), 3, 3), dtype=np.complex128)
    oms = np.zeros((len(frequency), 3, 3), dtype=np.complex128)
    for day_offset in day_offsets:
        sens = build_orbit_sensitivity(seg_index, day_offset=day_offset)
        operators = [sens.TDI_P_ij(P_ij_strings=s, f=frequency) for s in strings]
        sens_acc = TDISensitivity(Ri=sens.Ri, nij=sens.nij, dij=sens.dij, S_OMS=0.0, S_ACC=1.0)
        sens_oms = TDISensitivity(Ri=sens.Ri, nij=sens.nij, dij=sens.dij, S_OMS=1.0, S_ACC=0.0)
        for i, op_i in enumerate(operators):
            for j, op_j in enumerate(operators):
                acc[:, i, j] += sens_acc.TDI_noise_CSD(f=frequency, P_ij=op_i, Q_ij=op_j) / unit_factor(frequency)
                oms[:, i, j] += sens_oms.TDI_noise_CSD(f=frequency, P_ij=op_i, Q_ij=op_j) / unit_factor(frequency)
    acc /= float(len(day_offsets))
    oms /= float(len(day_offsets))
    return acc, oms


def tdi_response_amplitudes(sens: TDISensitivity, lam: float, beta: float, frequency: np.ndarray, p_ij):
    """Return complex plus/cross response amplitudes for one TDI operator."""
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


def calculate_xyz_response_for_segment(
    frequency: np.ndarray,
    seg_index: int,
    *,
    nsource: int,
    seed: int,
    day_offset: float = 5.0,
) -> np.ndarray:
    sens = build_orbit_sensitivity(seg_index, day_offset=day_offset)
    strings = xyz_strings()
    operators = [sens.TDI_P_ij(P_ij_strings=s, f=frequency) for s in strings]

    rng = np.random.default_rng(seed + seg_index)
    response_sum = np.zeros((len(frequency), 3, 3), dtype=np.complex128)
    for _ in tqdm(range(nsource), desc=f"seg {seg_index} response", leave=False):
        longitude = rng.uniform(0.0, TWOPI)
        latitude = np.arcsin(rng.uniform(-1.0, 1.0))
        hp = []
        hc = []
        for op in operators:
            hp_i, hc_i = tdi_response_amplitudes(sens, longitude, latitude, frequency, op)
            hp.append(hp_i)
            hc.append(hc_i)
        hp = np.asarray(hp)
        hc = np.asarray(hc)
        for i in range(3):
            for j in range(3):
                response_sum[:, i, j] += 0.5 * (np.conjugate(hp[i]) * hp[j] + np.conjugate(hc[i]) * hc[j])
    return response_sum / float(nsource) / unit_factor(frequency)[:, None, None] * 2.0


def calculate_xyz_response_averaged_for_segment(
    frequency: np.ndarray,
    seg_index: int,
    *,
    nsource: int,
    seed: int,
    day_offsets: tuple[float, ...],
) -> np.ndarray:
    response = np.zeros((len(frequency), 3, 3), dtype=np.complex128)
    for day_offset in day_offsets:
        response += calculate_xyz_response_for_segment(
            frequency,
            seg_index,
            nsource=nsource,
            seed=seed,
            day_offset=day_offset,
        )
    return response / float(len(day_offsets))


def calculate_xyz_response_and_transfer_for_segment(
    frequency: np.ndarray,
    seg_index: int,
    *,
    nsource: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sens = build_orbit_sensitivity(seg_index)
    strings = xyz_strings()
    operators = [sens.TDI_P_ij(P_ij_strings=s, f=frequency) for s in strings]
    response = calculate_xyz_response_for_segment(
        frequency,
        seg_index,
        nsource=nsource,
        seed=seed,
        day_offset=5.0,
    )

    sens_acc = TDISensitivity(Ri=sens.Ri, nij=sens.nij, dij=sens.dij, S_OMS=0.0, S_ACC=1.0)
    sens_oms = TDISensitivity(Ri=sens.Ri, nij=sens.nij, dij=sens.dij, S_OMS=1.0, S_ACC=0.0)
    acc = np.zeros((len(frequency), 3, 3), dtype=np.complex128)
    oms = np.zeros((len(frequency), 3, 3), dtype=np.complex128)
    for i, op_i in enumerate(operators):
        for j, op_j in enumerate(operators):
            acc[:, i, j] = sens_acc.TDI_noise_CSD(f=frequency, P_ij=op_i, Q_ij=op_j) / unit_factor(frequency)
            oms[:, i, j] = sens_oms.TDI_noise_CSD(f=frequency, P_ij=op_i, Q_ij=op_j) / unit_factor(frequency)
    return response, acc, oms


def frequency_subset_indices(cached_frequency: np.ndarray, frequency: np.ndarray) -> np.ndarray | None:
    if len(frequency) > len(cached_frequency):
        return None
    indices = np.searchsorted(cached_frequency, frequency)
    if np.any(indices >= len(cached_frequency)):
        return None
    if not np.array_equal(cached_frequency[indices], frequency):
        return None
    return indices


def prepare_model_tables(frequency: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if TABLE_PATH.exists() and not RECALCULATE_TABLES:
        with h5py.File(TABLE_PATH, "r") as h5:
            cached_frequency = h5["frequency"][:]
            if np.array_equal(cached_frequency, frequency):
                print(f"Loading cached XYZ response/noise tables: {TABLE_PATH}", flush=True)
                response = h5["response"][:]
                if RESPONSE_CONVENTION == "transpose":
                    response = np.swapaxes(response, -1, -2).copy()
                    print("Using direct XYZ response with transposed off-diagonal convention.", flush=True)
                elif RESPONSE_CONVENTION == "as_is":
                    print("Using direct XYZ response as stored.", flush=True)
                else:
                    raise ValueError("RESPONSE_CONVENTION must be 'as_is' or 'transpose'.")
                if NOISE_AVERAGE_DAY_OFFSETS:
                    start = log_step(
                        "Computing direct XYZ noise transfer averaged inside each segment "
                        f"at day offsets {NOISE_AVERAGE_DAY_OFFSETS}"
                    )
                    acc_table = []
                    oms_table = []
                    for seg_index in range(1, SEGMENTS + 1):
                        acc, oms = calculate_xyz_noise_transfer_for_segment(
                            frequency,
                            seg_index,
                            day_offsets=NOISE_AVERAGE_DAY_OFFSETS,
                        )
                        acc_table.append(acc)
                        oms_table.append(oms)
                    finish_step(start, "Computed averaged direct XYZ noise transfer")
                    return response, np.asarray(acc_table), np.asarray(oms_table)
                return response, h5["acc_transfer"][:], h5["oms_transfer"][:]
            subset_indices = frequency_subset_indices(cached_frequency, frequency)
            if subset_indices is not None:
                print(f"Loading cached XYZ response table subset: {TABLE_PATH}", flush=True)
                response = h5["response"][:, subset_indices]
                if RESPONSE_CONVENTION == "transpose":
                    response = np.swapaxes(response, -1, -2).copy()
                    print("Using direct XYZ response with transposed off-diagonal convention.", flush=True)
                elif RESPONSE_CONVENTION == "as_is":
                    print("Using direct XYZ response as stored.", flush=True)
                else:
                    raise ValueError("RESPONSE_CONVENTION must be 'as_is' or 'transpose'.")
                if NOISE_AVERAGE_DAY_OFFSETS:
                    start = log_step(
                        "Computing direct XYZ noise transfer averaged inside each segment "
                        f"at day offsets {NOISE_AVERAGE_DAY_OFFSETS}"
                    )
                    acc_table = []
                    oms_table = []
                    for seg_index in range(1, SEGMENTS + 1):
                        acc, oms = calculate_xyz_noise_transfer_for_segment(
                            frequency,
                            seg_index,
                            day_offsets=NOISE_AVERAGE_DAY_OFFSETS,
                        )
                        acc_table.append(acc)
                        oms_table.append(oms)
                    finish_step(start, "Computed averaged direct XYZ noise transfer")
                    return response, np.asarray(acc_table), np.asarray(oms_table)
                return response, h5["acc_transfer"][:, subset_indices], h5["oms_transfer"][:, subset_indices]
            if STRICT_TABLE_CACHE:
                raise RuntimeError(
                    "Cached table frequency grid is neither an exact match nor a superset of the PE grid. "
                    "Refusing to recalculate because STRICT_TABLE_CACHE=1."
                )
            print("Cached table frequency grid mismatch; recalculating.", flush=True)

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    start = log_step(f"Computing XYZ response/noise tables for {SEGMENTS} segments and {len(frequency)} frequency bins")
    response_table = []
    acc_table = []
    oms_table = []
    for seg_index in range(1, SEGMENTS + 1):
        if RESPONSE_AVERAGE_DAY_OFFSETS:
            response = calculate_xyz_response_averaged_for_segment(
                frequency,
                seg_index,
                nsource=N_SOURCE,
                seed=BASE_SEED,
                day_offsets=RESPONSE_AVERAGE_DAY_OFFSETS,
            )
            if NOISE_AVERAGE_DAY_OFFSETS:
                acc, oms = calculate_xyz_noise_transfer_for_segment(
                    frequency,
                    seg_index,
                    day_offsets=NOISE_AVERAGE_DAY_OFFSETS,
                )
            else:
                _, acc, oms = calculate_xyz_response_and_transfer_for_segment(
                    frequency,
                    seg_index,
                    nsource=N_SOURCE,
                    seed=BASE_SEED,
                )
        else:
            response, acc, oms = calculate_xyz_response_and_transfer_for_segment(
                frequency,
                seg_index,
                nsource=N_SOURCE,
                seed=BASE_SEED,
            )
        response_table.append(response)
        acc_table.append(acc)
        oms_table.append(oms)
    response_table = np.asarray(response_table, dtype=np.complex128)
    acc_table = np.asarray(acc_table, dtype=np.complex128)
    oms_table = np.asarray(oms_table, dtype=np.complex128)
    tmp_path = TABLE_PATH.with_suffix(TABLE_PATH.suffix + ".tmp")
    with h5py.File(tmp_path, "w") as h5:
        h5.attrs["orientation"] = (
            "left-conjugated CSD/model covariance: matrix[i,j] = <d_i^* d_j>; "
            "same orientation as obs_csd_star"
        )
        h5.attrs["channels"] = ",".join(CHANNELS)
        h5.attrs["response_average_day_offsets"] = ",".join(str(x) for x in RESPONSE_AVERAGE_DAY_OFFSETS)
        h5.attrs["noise_average_day_offsets"] = ",".join(str(x) for x in NOISE_AVERAGE_DAY_OFFSETS)
        h5.attrs["n_source"] = N_SOURCE
        h5.attrs["seed"] = BASE_SEED
        h5.create_dataset("frequency", data=frequency)
        h5.create_dataset("response", data=response_table, compression="gzip", compression_opts=2)
        h5.create_dataset("acc_transfer", data=acc_table, compression="gzip", compression_opts=2)
        h5.create_dataset("oms_transfer", data=oms_table, compression="gzip", compression_opts=2)
    tmp_path.replace(TABLE_PATH)
    finish_step(start, f"Saved XYZ response/noise tables to {TABLE_PATH}")
    if RESPONSE_CONVENTION == "transpose":
        response_table = np.swapaxes(response_table, -1, -2).copy()
        print("Using direct XYZ response with transposed off-diagonal convention.", flush=True)
    elif RESPONSE_CONVENTION != "as_is":
        raise ValueError("RESPONSE_CONVENTION must be 'as_is' or 'transpose'.")
    return response_table, acc_table, oms_table


def load_data(check_nf: int | None = None):
    with h5py.File(DATA_PATH, "r") as h5:
        frequency = np.asarray(h5["frequency"][:], dtype=np.float64)
        obs = np.asarray(h5["obs_csd_star"][:], dtype=np.complex128)
        parameters_true = np.asarray(h5["parameters/parameters_true"][:], dtype=np.float64)
    pe_mask = build_pe_frequency_mask(frequency)
    if not np.all(pe_mask):
        frequency = frequency[pe_mask]
        obs = obs[:, pe_mask]
    print(
        "PE frequency bins: "
        f"{len(frequency)}; range=[{frequency[0]:.6e}, {frequency[-1]:.6e}], "
        f"PE_F_RANGE={PE_F_RANGE}, PE_NOTCH_WIDTH={PE_NOTCH_WIDTH_VALUE}",
        flush=True,
    )
    if check_nf is not None and check_nf > 0:
        frequency = frequency[:check_nf]
        obs = obs[:, :check_nf]
    return frequency, obs, parameters_true


def run_pe(check_only: bool = False) -> None:
    import bilby
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    print("JAX devices:", jax.devices(), flush=True)

    frequency, obs, parameters_true = load_data(CHECK_NF if check_only else None)
    response, acc_transfer, oms_transfer = prepare_model_tables(frequency)

    frequency_j = jnp.asarray(frequency, dtype=jnp.float64)
    obs_j = jnp.asarray(obs, dtype=jnp.complex128)
    response_j = jnp.asarray(response, dtype=jnp.complex128)
    acc_j = jnp.asarray(acc_transfer, dtype=jnp.complex128)
    oms_j = jnp.asarray(oms_transfer, dtype=jnp.complex128)

    log_const = np.log((np.pi**3) * (T_SEG**3 * FSAMPLE**6) / 8.0)
    prefactor = 2.0 / (T_SEG * FSAMPLE**2)

    @jax.jit
    def sh_ast(theta):
        omega = 10.0 ** theta[2] * (frequency_j / 1.0e-3) ** theta[3]
        return 3.0 * H0**2 / (4.0 * jnp.pi**2) * omega / frequency_j**3

    @jax.jit
    def covariance_matrices(theta):
        n_acc = theta[0] * 1.0e-15
        delta_x = theta[1] * 1.0e-12
        signal = sh_ast(theta)
        return signal[None, :, None, None] * response_j + (n_acc**2) * acc_j + (delta_x**2) * oms_j

    @jax.jit
    def loglike(theta):
        cov = covariance_matrices(theta)
        sign, logabsdet = jnp.linalg.slogdet(cov)
        # obs_csd_star and cov both use Q_star[i,j]=d_i^* d_j.
        # Therefore trace(P_star^-1 Q_star) is identical to the standard
        # quadratic form d^\dagger P^{-1} d without transposing only one term.
        solved = jnp.linalg.solve(cov, obs_j)
        quad = jnp.real(jnp.trace(solved, axis1=-2, axis2=-1))
        return -jnp.sum(log_const + jnp.real(logabsdet) + prefactor * quad)

    theta_true_10 = jnp.asarray(
        [parameters_true[0], parameters_true[1], parameters_true[2], parameters_true[3], -100.0, 0.0, -100.0, 0.0, -100.0, 0.0],
        dtype=jnp.float64,
    )
    start = log_step("Compiling and evaluating full-covariance XYZ log-likelihood at injected parameters")
    injected_loglike = float(loglike(theta_true_10))
    finish_step(start, f"Injected log-likelihood = {injected_loglike:.6e}")

    diagnostic_theta_10 = jnp.asarray(
        [DIAGNOSTIC_THETA[0], DIAGNOSTIC_THETA[1], DIAGNOSTIC_THETA[2], DIAGNOSTIC_THETA[3], -100.0, 0.0, -100.0, 0.0, -100.0, 0.0],
        dtype=jnp.float64,
    )
    start = log_step("Evaluating a previously wrong-convergence diagnostic point")
    diagnostic_loglike = float(loglike(diagnostic_theta_10))
    finish_step(start, f"Diagnostic log-likelihood = {diagnostic_loglike:.6e}")
    print(
        f"Injected minus diagnostic log-likelihood = {injected_loglike - diagnostic_loglike:.6e}",
        flush=True,
    )

    if check_only:
        print("Check mode finished; sampler was not run.", flush=True)
        return

    priors = get_priors(model=MODEL)
    likelihood = Likelihood(params=para_range, model=MODEL, loglike=loglike)
    start = log_step(f"Running Bilby.dynesty PE: nlive={SAMPLE_NUMBER}, dlogz={DLOGZ}, walks={WALKS}")
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=SAMPLE_NUMBER,
        walks=WALKS,
        npool=1,
        outdir="./data/bilby",
        label=LABEL,
        clean=True,
        resume=False,
        dlogz=DLOGZ,
        update_interval=100,
    )
    finish_step(start, "Bilby.dynesty PE finished")

    trace = np.asarray(result.posterior)[:, : len(parameters_true)]
    output_path = RESULT_PATH or DATA_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "a") as h5:
        if RESULT_GROUP in h5:
            del h5[RESULT_GROUP]
        group = h5.create_group(RESULT_GROUP)
        group.create_dataset("trace", data=trace)
        group.create_dataset("evidence", data=result.log_evidence)
        group.create_dataset("frequency", data=frequency)
        group.create_dataset("parameters_true", data=parameters_true)
        group.attrs["sampler"] = "Bilby.dynesty"
        group.attrs["nlive"] = SAMPLE_NUMBER
        group.attrs["dlogz"] = DLOGZ
        group.attrs["walks"] = WALKS
        group.attrs["label"] = LABEL
        group.attrs["data_path"] = str(DATA_PATH)
        group.attrs["table_path"] = str(TABLE_PATH)
        group.attrs["pe_f_min"] = "None" if PE_F_RANGE[0] is None else PE_F_RANGE[0]
        group.attrs["pe_f_max"] = "None" if PE_F_RANGE[1] is None else PE_F_RANGE[1]
        group.attrs["pe_notch_width"] = "None" if PE_NOTCH_WIDTH_VALUE is None else PE_NOTCH_WIDTH_VALUE
        group.attrs["noise_average_day_offsets"] = ",".join(str(x) for x in NOISE_AVERAGE_DAY_OFFSETS)
        group.attrs["response_average_day_offsets"] = ",".join(str(x) for x in RESPONSE_AVERAGE_DAY_OFFSETS)
    print(f"Stored PE result in {output_path}: {RESULT_GROUP}", flush=True)


def main() -> None:
    print(f"Repository: {REPO_ROOT}", flush=True)
    print(f"Mode: {MODE}", flush=True)
    print(f"Data path: {DATA_PATH}", flush=True)
    print(f"Table path: {TABLE_PATH}", flush=True)
    if RESULT_PATH is not None:
        print(f"Result path: {RESULT_PATH}", flush=True)
    if MODE == "generate":
        generate_xyz_fullcov_data()
    elif MODE == "pe":
        run_pe(check_only=False)
    elif MODE == "both":
        generate_xyz_fullcov_data()
        run_pe(check_only=False)
    elif MODE == "check":
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Missing data file {DATA_PATH}; run MODE=generate first.")
        run_pe(check_only=True)
    else:
        raise ValueError(f"Unsupported MODE={MODE!r}. Use generate, pe, both, or check.")


if __name__ == "__main__":
    main()
