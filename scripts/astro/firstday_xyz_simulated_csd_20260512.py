#!/usr/bin/env python3
"""First-day XYZ simulated CSD comparison for three controlled cases.

Cases:
  1. Frequency-domain equal-arm, second-generation TDI.
  2. Time-domain Triangle simulation, equal-arm, second-generation TDI.
  3. Time-domain Triangle simulation, unequal-arm, second-generation TDI.

The plotted matrix is the PSD-normalized left-conjugated covariance

    P_IJ(f) = 2 < d_I(f)^* d_J(f) > / (T_seg fs^2),

for I,J in X,Y,Z.  We show the six independent entries
XX, YY, ZZ, XY, XZ, YZ, with amplitude and phase in [0, 2 pi).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import windows


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "py_code").exists() and (parent / "data").exists():
            return parent
    fallback = Path(__file__).resolve().parents[2]
    if (fallback / "py_code").exists():
        return fallback
    raise RuntimeError("Could not locate lisa_td_paper_codex repo root.")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TRIANGLE_ROOT = Path(os.getenv("TRIANGLE_ROOT", str(REPO_ROOT / "vendor/Triangle-Simulator")))
if TRIANGLE_ROOT.exists() and str(TRIANGLE_ROOT) not in sys.path:
    sys.path.append(str(TRIANGLE_ROOT))
os.chdir(REPO_ROOT)

from py_code.parameters import unit_factor  # noqa: E402
from py_code.triangle_code import (  # noqa: E402
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


CHANNELS = ("X", "Y", "Z")
ENTRIES = (("XX", 0, 0), ("YY", 1, 1), ("ZZ", 2, 2), ("XY", 0, 1), ("XZ", 0, 2), ("YZ", 1, 2))
FID_N_ACC = 3.0e-15
FID_DELTA_X = 8.0e-12
FID_LOG10_OMEGA = -11.5
FID_EPSILON = 2.0 / 3.0
DEFAULT_FS = 0.4
DEFAULT_SEGMENT_DAYS = 1.0
DEFAULT_F_MIN = 3.0e-5
DEFAULT_F_MAX = 0.15
DEFAULT_NSIDE = 3
DEFAULT_N_SOURCE = 256
DEFAULT_N_REAL = 24
DEFAULT_SEED = 20260512


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def xyz_strings() -> tuple[dict, dict, dict]:
    x_string = TDI.X2_string
    y_string = TDIStringManipulation.TDIStringCyc(x_string)
    z_string = TDIStringManipulation.TDIStringCyc(y_string)
    return x_string, y_string, z_string


def orbit_config(geometry: str) -> tuple[str, int]:
    if geometry == "equal":
        return "./data/OrbitData/TaijiEqualArmOrbit/", 0
    if geometry == "unequal":
        return "./data/OrbitData/MicroSateOrbitEclipticTCB/", 2
    raise ValueError(f"unknown geometry {geometry!r}")


def build_orbit_sensitivity(geometry: str, day_midpoint: float = 0.5) -> TDISensitivity:
    orbit_dir, pn_order = orbit_config(geometry)
    orbit_time = day_midpoint * DAY
    orbit = lisa_generate.Orbit(OrbitDir=orbit_dir, pn_order=pn_order)
    ri = {key: orbit.Positionfunctions()[key](orbit_time) for key in SC_labels}
    nij = {key: orbit.ArmVectorfunctions()[key](orbit_time) for key in MOSA_labels}
    dij = {key: orbit.LTTfunctions()[key](orbit_time) for key in MOSA_labels}
    return TDISensitivity(Ri=ri, nij=nij, dij=dij)


def tdi_response_amplitudes(
    sens: TDISensitivity,
    lam: float,
    beta: float,
    frequency: np.ndarray,
    p_ij: dict,
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


def response_matrix(
    frequency: np.ndarray,
    geometry: str,
    *,
    n_source: int,
    seed: int,
) -> np.ndarray:
    sens = build_orbit_sensitivity(geometry, 0.5)
    operators = [sens.TDI_P_ij(P_ij_strings=s, f=frequency) for s in xyz_strings()]
    rng = np.random.default_rng(seed)
    acc = np.zeros((len(frequency), 3, 3), dtype=np.complex128)
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
                acc[:, i, j] += 0.5 * (
                    np.conjugate(hp_arr[i]) * hp_arr[j]
                    + np.conjugate(hc_arr[i]) * hc_arr[j]
                )
    return acc / float(n_source) / unit_factor(frequency)[:, None, None] * 2.0


def noise_transfer_matrices(frequency: np.ndarray, geometry: str) -> tuple[np.ndarray, np.ndarray]:
    sens = build_orbit_sensitivity(geometry, 0.5)
    operators = [sens.TDI_P_ij(P_ij_strings=s, f=frequency) for s in xyz_strings()]
    sens_acc = TDISensitivity(Ri=sens.Ri, nij=sens.nij, dij=sens.dij, S_OMS=0.0, S_ACC=1.0)
    sens_oms = TDISensitivity(Ri=sens.Ri, nij=sens.nij, dij=sens.dij, S_OMS=1.0, S_ACC=0.0)
    acc = np.zeros((len(frequency), 3, 3), dtype=np.complex128)
    oms = np.zeros_like(acc)
    uf = unit_factor(frequency)
    for i, op_i in enumerate(operators):
        for j, op_j in enumerate(operators):
            acc[:, i, j] = sens_acc.TDI_noise_CSD(f=frequency, P_ij=op_i, Q_ij=op_j) / uf
            oms[:, i, j] = sens_oms.TDI_noise_CSD(f=frequency, P_ij=op_i, Q_ij=op_j) / uf
    return acc, oms


def omega_ast(frequency: np.ndarray) -> np.ndarray:
    f = np.maximum(np.asarray(frequency, dtype=float), 1.0e-30)
    return 10.0**FID_LOG10_OMEGA * (f / 1.0e-3) ** FID_EPSILON


def model_psd_matrix(frequency: np.ndarray, geometry: str, *, n_source: int, seed: int) -> np.ndarray:
    acc, oms = noise_transfer_matrices(frequency, geometry)
    resp = response_matrix(frequency, geometry, n_source=n_source, seed=seed)
    return (
        FID_N_ACC**2 * acc
        + FID_DELTA_X**2 * oms
        + omega_ast(frequency)[:, None, None] * resp
    )


def frequency_grid(*, fs: float, segment_days: float) -> np.ndarray:
    n_time = int(round(segment_days * DAY * fs))
    n_fft = n_time - 200
    return rfftfreq(n_fft, 1.0 / fs)[1:]


def draw_fd_realization(seed: int, frequency: np.ndarray, p_left: np.ndarray, *, fs: float, segment_days: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prefactor = segment_days * DAY * fs**2 / 2.0
    z = np.zeros((len(frequency), 3), dtype=np.complex128)
    eye = np.eye(3)
    for k in range(len(frequency)):
        # p_left is <d^* d>.  Cholesky wants <d d^H>.
        c_standard = np.conjugate(prefactor * p_left[k])
        c_standard = 0.5 * (c_standard + c_standard.conj().T)
        jitter = max(float(np.max(np.abs(np.diag(c_standard)))) * 1.0e-12, 1.0e-300)
        chol = np.linalg.cholesky(c_standard + jitter * eye)
        u = (rng.normal(size=3) + 1j * rng.normal(size=3)) / np.sqrt(2.0)
        z[k] = chol @ u
    return z


def generate_triangle_tdi(
    seed: int,
    *,
    geometry: str,
    fs: float,
    segment_days: float,
    nside: int,
):
    np.random.seed(seed)
    orbit_dir, pn_order = orbit_config(geometry)
    orbit = lisa_generate.Orbit(OrbitDir=orbit_dir, pn_order=pn_order)
    size = int(round(segment_days * DAY * fs))
    tdata = np.arange(size) / fs
    interp_order = 31
    offset = lisa_generate.OffsetFunctions(laser_flag=False, clock_flag=False)
    instrumental_noise = lisa_generate.InstrumentalPSDs(L=lisa_generate.L_nominal)

    def s_sgwb(f):
        return omega_ast(np.asarray(f))

    npix = lisa_generate.hlp.nside2npix(nside)
    colatitude, longitude = lisa_generate.hlp.pix2ang(nside=nside, ipix=np.arange(npix))
    latitude = np.pi / 2.0 - colatitude
    sgwb_generator = lisa_generate.GeneralNoise(s_sgwb)
    gw = []
    for i in range(npix):
        hp_k = sgwb_generator(fsample=fs, size=size) / np.sqrt(npix)
        hc_k = sgwb_generator(fsample=fs, size=size) / np.sqrt(npix)
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
        fsample=fs,
        time_frame="ProperTime",
        acc_noise=True,
        ro_noise=True,
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


def scaled_frequency_data_from_tdi(tdi, *, fs: float, segment_days: float, window: str) -> tuple[np.ndarray, np.ndarray]:
    trim = 100
    channels = [
        np.asarray(tdi.measurements["X2"][trim:-trim], dtype=np.float64) / F_LASER,
        np.asarray(tdi.measurements["Y2"][trim:-trim], dtype=np.float64) / F_LASER,
        np.asarray(tdi.measurements["Z2"][trim:-trim], dtype=np.float64) / F_LASER,
    ]
    n_data = len(channels[0])
    if window == "hann":
        win = windows.hann(n_data)
    elif window == "boxcar":
        win = np.ones(n_data)
    else:
        raise ValueError(window)
    frequency_full = rfftfreq(n_data, 1.0 / fs)
    positive = np.arange(1, len(frequency_full))
    frequency = frequency_full[positive]
    one_sided = np.full(len(frequency), 2.0)
    if n_data % 2 == 0:
        one_sided[-1] = 1.0
    scale = np.sqrt((segment_days * DAY) * fs * one_sided / (2.0 * float(np.sum(win**2))))
    z = np.zeros((len(frequency), 3), dtype=np.complex128)
    uf = np.sqrt(unit_factor(frequency))
    for i, channel in enumerate(channels):
        detrended = channel - np.mean(channel)
        z[:, i] = rfft(detrended * win)[positive] * scale / uf
    return frequency, z


def triangle_realization(seed: int, *, geometry: str, fs: float, segment_days: float, nside: int, window: str) -> np.ndarray:
    tdi = generate_triangle_tdi(seed, geometry=geometry, fs=fs, segment_days=segment_days, nside=nside)
    _, z = scaled_frequency_data_from_tdi(tdi, fs=fs, segment_days=segment_days, window=window)
    return z


def covariance_psd(dtilde: np.ndarray, *, fs: float, segment_days: float) -> np.ndarray:
    cov_left = np.einsum("rfi,rfj->fij", np.conjugate(dtilde), dtilde, optimize=True) / dtilde.shape[0]
    return 2.0 * cov_left / (segment_days * DAY * fs**2)


def fd_worker(payload: tuple[int, np.ndarray, np.ndarray, float, float]) -> np.ndarray:
    seed, frequency, p_left, fs, segment_days = payload
    return draw_fd_realization(seed, frequency, p_left, fs=fs, segment_days=segment_days)


def td_worker(payload: tuple[int, str, float, float, int, str]) -> np.ndarray:
    seed, geometry, fs, segment_days, nside, window = payload
    return triangle_realization(
        seed,
        geometry=geometry,
        fs=fs,
        segment_days=segment_days,
        nside=nside,
        window=window,
    )


def run_ensemble(case: str, args: argparse.Namespace, frequency: np.ndarray) -> np.ndarray:
    seed_offsets = {
        "fd_equal_2ndtdi": 101,
        "td_equal_triangle_2ndtdi": 202,
        "td_unequal_triangle_2ndtdi": 303,
    }
    if case == "fd_equal_2ndtdi":
        p_left = model_psd_matrix(
            frequency,
            "equal",
            n_source=args.n_source,
            seed=args.base_seed + 17,
        )
        worker = fd_worker
        payloads = [
            (args.base_seed + seed_offsets[case] + 1000 * (i + 1), frequency, p_left, args.fs, args.segment_days)
            for i in range(args.n_real)
        ]
    elif case == "td_equal_triangle_2ndtdi":
        worker = td_worker
        payloads = [
            (args.base_seed + seed_offsets[case] + 1000 * (i + 1), "equal", args.fs, args.segment_days, args.nside, args.window)
            for i in range(args.n_real)
        ]
    elif case == "td_unequal_triangle_2ndtdi":
        worker = td_worker
        payloads = [
            (args.base_seed + seed_offsets[case] + 1000 * (i + 1), "unequal", args.fs, args.segment_days, args.nside, args.window)
            for i in range(args.n_real)
        ]
    else:
        raise ValueError(case)

    results: list[np.ndarray] = []
    if args.n_workers <= 1:
        for idx, payload in enumerate(payloads, 1):
            log(f"{case}: realization {idx}/{len(payloads)}")
            results.append(worker(payload))
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futs = {pool.submit(worker, payload): i for i, payload in enumerate(payloads, 1)}
            for fut in as_completed(futs):
                idx = futs[fut]
                results.append(fut.result())
                log(f"{case}: completed realization {idx}/{len(payloads)} ({len(results)}/{len(payloads)})")
    dtilde = np.stack(results, axis=0)
    return covariance_psd(dtilde, fs=args.fs, segment_days=args.segment_days)


def phase_0_2pi(values: np.ndarray) -> np.ndarray:
    return np.mod(np.angle(values), 2.0 * np.pi)


def plot_case(case: str, frequency: np.ndarray, p_sim: np.ndarray, mask: np.ndarray, output_dir: Path) -> Path:
    title = {
        "fd_equal_2ndtdi": "FD equal-arm, second-generation TDI",
        "td_equal_triangle_2ndtdi": "Triangle TD equal-arm, second-generation TDI",
        "td_unequal_triangle_2ndtdi": "Triangle TD unequal-arm, second-generation TDI",
    }[case]
    fig, axes = plt.subplots(4, 3, figsize=(15.0, 11.5), constrained_layout=True)
    fig.suptitle(f"First one-day segment simulated XYZ CSD: {title}", fontsize=15)
    axes_flat = axes.ravel()
    for ax, (name, i, j) in zip(axes_flat[:6], ENTRIES):
        y = np.full(len(frequency), np.nan)
        y[mask] = np.abs(p_sim[:, i, j])[mask]
        ax.loglog(frequency, y, ".", ms=2.0, color="#0072B2", alpha=0.75)
        ax.set_title(f"|{name}|")
        ax.set_ylabel(r"$|P_{IJ}(f)|$")
        ax.grid(True, which="both", alpha=0.22)
    for ax, (name, i, j) in zip(axes_flat[6:], ENTRIES):
        y = np.full(len(frequency), np.nan)
        y[mask] = phase_0_2pi(p_sim[:, i, j])[mask]
        ax.semilogx(frequency, y, ".", ms=2.0, color="#D55E00", alpha=0.75)
        ax.set_title(f"arg {name}")
        ax.set_ylabel(r"phase $[0,2\pi)$")
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylim(0.0, 2.0 * np.pi)
        ax.set_yticks([0.0, np.pi, 2.0 * np.pi])
        ax.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
        ax.grid(True, which="both", alpha=0.22)
    for ax in axes_flat:
        ax.set_xlim(float(frequency[mask].min()), float(frequency[mask].max()))
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{case}_firstday_xyz_csd_abs_arg_20260512.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/firstday_xyz_csd_20260512/firstday_xyz_csd_20260512.h5")
    parser.add_argument("--figure-dir", type=Path, default=REPO_ROOT / "figure/firstday_xyz_csd_20260512")
    parser.add_argument("--n-real", type=int, default=DEFAULT_N_REAL)
    parser.add_argument("--n-workers", type=int, default=6)
    parser.add_argument("--n-source", type=int, default=DEFAULT_N_SOURCE)
    parser.add_argument("--nside", type=int, default=DEFAULT_NSIDE)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--segment-days", type=float, default=DEFAULT_SEGMENT_DAYS)
    parser.add_argument("--f-min", type=float, default=DEFAULT_F_MIN)
    parser.add_argument("--f-max", type=float, default=DEFAULT_F_MAX)
    parser.add_argument("--window", choices=("hann", "boxcar"), default="hann")
    parser.add_argument("--cases", default="fd_equal_2ndtdi,td_equal_triangle_2ndtdi,td_unequal_triangle_2ndtdi")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    frequency = frequency_grid(fs=args.fs, segment_days=args.segment_days)
    mask = (frequency >= args.f_min) & (frequency <= args.f_max)
    if not args.overwrite and args.output.exists():
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.output, "w") as h5:
        h5.attrs["description"] = "First-day FD/TD equal/unequal second-generation XYZ simulated CSD diagnostic."
        h5.attrs["fs"] = args.fs
        h5.attrs["segment_days"] = args.segment_days
        h5.attrs["day_interval_days"] = "[0, 1]"
        h5.attrs["day_midpoint_days"] = 0.5
        h5.attrs["frequency_mask"] = f"[{args.f_min}, {args.f_max}] Hz, no notch applied for display"
        h5.attrs["n_real"] = args.n_real
        h5.attrs["nside"] = args.nside
        h5.attrs["n_source"] = args.n_source
        h5.attrs["fiducial"] = "(N_acc, delta_x, log10_Omega_ast, epsilon) = (3.0, 8.0, -11.5, 2/3)"
        h5.attrs["covariance_convention"] = "P_IJ(f)=2<d_I(f)^* d_J(f)>/(T_seg fs^2), left-conjugated XYZ orientation."
        h5.create_dataset("frequency", data=frequency)
        h5.create_dataset("mask", data=mask)
        for case in cases:
            t0 = time.perf_counter()
            log(f"Starting {case}")
            p_sim = run_ensemble(case, args, frequency)
            grp = h5.create_group(case)
            grp.create_dataset("P_sim", data=p_sim, compression="gzip", compression_opts=4)
            grp.attrs["elapsed_seconds"] = time.perf_counter() - t0
            if "fd_" in case:
                grp.attrs["simulation_domain"] = "frequency-domain complex Gaussian draw"
                grp.attrs["orbit"] = "equal-arm"
            elif "td_equal" in case:
                grp.attrs["simulation_domain"] = "Triangle time-domain"
                grp.attrs["orbit"] = "equal-arm, ./data/OrbitData/TaijiEqualArmOrbit/, pn_order=0"
            else:
                grp.attrs["simulation_domain"] = "Triangle time-domain"
                grp.attrs["orbit"] = "unequal-arm, ./data/OrbitData/MicroSateOrbitEclipticTCB/, pn_order=2"
            out = plot_case(case, frequency, p_sim, mask, args.figure_dir)
            grp.attrs["figure"] = str(out)
            log(f"Finished {case}; figure={out}; elapsed={grp.attrs['elapsed_seconds']:.1f}s")
    log(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
