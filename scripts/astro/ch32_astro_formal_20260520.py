#!/usr/bin/env python3
"""Chapter 3.2 formal astro campaign utilities.

This is a compact, additive wrapper for the 2026-05-20 Ch.3.2 campaign.  It
starts with the mandatory FD equal-arm benchmark using second-generation TDI,
full XYZ covariance, and the required eight-parameter foreground/background
fit vector.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import h5py
import numpy as np


REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "py_code").exists() and (p / "data").exists()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from py_code.model import calculate_response_function, calculate_transfer_function  # noqa: E402
from py_code.parameters import H0, L, c, para_range, unit_factor  # noqa: E402
from py_code.triangle_code import DAY, F_LASER  # noqa: E402


CAMPAIGN = "ch32_astro_formal_20260520"
RUN_TAG = "ch32_fd_equal_fullxyz_20260520"
DATA_ROOT = REPO_ROOT / "data" / CAMPAIGN
LOG_ROOT = REPO_ROOT / "logs" / CAMPAIGN
FIG_ROOT = REPO_ROOT / "figure" / CAMPAIGN
FD_ROOT = DATA_ROOT / "fd_equal"
FD_TABLE_DIR = FD_ROOT / "tables"
FD_DATA_DIR = FD_ROOT / "data"
FD_BUNDLE_DIR = FD_ROOT / "bundles"
FD_RESULT_DIR = FD_ROOT / "results"
FD_BILBY_DIR = FD_ROOT / "bilby"
TD_EQUAL_RUN_TAG = "ch32_td_equal_fullxyz_20260520"
TD_EQUAL_ROOT = DATA_ROOT / "td_equal"
TD_EQUAL_TABLE_DIR = TD_EQUAL_ROOT / "tables"
TD_EQUAL_DATA_DIR = TD_EQUAL_ROOT / "data"
TD_EQUAL_BUNDLE_DIR = TD_EQUAL_ROOT / "bundles"
TD_EQUAL_RESULT_DIR = TD_EQUAL_ROOT / "results"
TD_EQUAL_BILBY_DIR = TD_EQUAL_ROOT / "bilby"
TD_UNEQUAL_RUN_TAG = "ch32_td_unequal_fullxyz_20260520"
TD_UNEQUAL_ROOT = DATA_ROOT / "td_unequal"
TD_UNEQUAL_TABLE_DIR = TD_UNEQUAL_ROOT / "tables"
TD_UNEQUAL_DATA_DIR = TD_UNEQUAL_ROOT / "data"
TD_UNEQUAL_BUNDLE_DIR = TD_UNEQUAL_ROOT / "bundles"
TD_UNEQUAL_RESULT_DIR = TD_UNEQUAL_ROOT / "results"
TD_UNEQUAL_BILBY_DIR = TD_UNEQUAL_ROOT / "bilby"
TD_FULLFREQ_PRECOMPUTE = (
    DATA_ROOT
    / "precomputed_tables"
    / "ch32_response_noise_fullfreq_20260521"
    / "ch32_response_noise_fullfreq_20260521_equal_unequal_fullfreq_seg360.h5"
)

FREQUENCY_SOURCE = REPO_ROOT / "data/frequency_grid.h5"

MODEL_FULL = "noise+dwd+ast"
MODEL_BF = "noise+dwd"
FIT_NAMES_FULL = (
    "N_acc_amp",
    "delta_x_amp",
    "log_A1",
    "alpha1",
    "log_A2",
    "alpha2",
    "log_Omega_ast",
    "alpha_ast",
)
FIT_DISPLAY_NAMES_FULL = (
    "N_acc",
    "delta_x",
    "log10_A1",
    "alpha1",
    "log10_A2",
    "alpha2",
    "log10_Omega_ast",
    "epsilon",
)
FIT_NAMES_BF = FIT_NAMES_FULL[:6]
TRUTH_FULL = np.asarray([3.0, 8.0, -15.40, -5.7, -6.32, -6.2, -11.5, 2.0 / 3.0], dtype=np.float64)
DEFAULT_LOG_OMEGA_AST = float(TRUTH_FULL[6])
PRIOR_BOUNDS_FULL = np.asarray([[para_range[name][0], para_range[name][1]] for name in FIT_NAMES_FULL], dtype=np.float64)

CHANNELS = ("X", "Y", "Z")
U_AET_FROM_XYZ = np.asarray(
    [
        [-1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
        [1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)],
        [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
    ],
    dtype=np.float64,
)

DEFAULT_SEGMENTS = 360
DEFAULT_SEGMENT_DAYS = 1.0
DEFAULT_FSAMPLE = 0.4
DEFAULT_F_MIN = 1.0e-4
DEFAULT_F_MAX = 0.15
DEFAULT_NOTCH_WIDTH = 0.010
DEFAULT_RESPONSE_SEED = 20260520
DEFAULT_DATA_SEED = 2026052000
DEFAULT_N_SOURCE = 2**10
DEFAULT_ORBIT_CHUNKS = 360
DEFAULT_DAYS_PER_ORBIT_CHUNK = 1
DEFAULT_EQUAL_ORBIT_DIR = str(REPO_ROOT / "data/OrbitData/TaijiEqualArmOrbit")
DEFAULT_EQUAL_ORBIT_DIR_TEMPLATE = str(REPO_ROOT / "data/orbit_equal_360day/seg{chunk1}")
DEFAULT_UNEQUAL_ORBIT_DIR_TEMPLATE = str(REPO_ROOT / "data/orbit_unequal_360day/seg{chunk1}")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S %z')}] {message}", flush=True)


def stage_start(message: str) -> float:
    log(message)
    return time.perf_counter()


def stage_end(start: float, message: str) -> None:
    elapsed = time.perf_counter() - start
    log(f"{message} in {elapsed:.2f} s")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fresh_tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp_{os.getpid()}_{int(time.time())}")


def repo_path(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def fmt_float_for_tag(value: float) -> str:
    return f"{value:.0e}".replace("+", "").replace("-", "m").replace(".", "p")


def fmt_decimal_for_tag(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace("+", "p").replace("-", "m").replace(".", "p")


def fd_injection_suffix(log_omega_ast: float) -> str:
    if math.isclose(float(log_omega_ast), DEFAULT_LOG_OMEGA_AST, rel_tol=0.0, abs_tol=1.0e-12):
        return ""
    return f"_omast{fmt_decimal_for_tag(float(log_omega_ast))}"


def injection_folder(log_omega_ast: float) -> str:
    return f"omast{fmt_decimal_for_tag(float(log_omega_ast))}"


def truth_full_from_args(args: argparse.Namespace) -> np.ndarray:
    truth = TRUTH_FULL.copy()
    truth[6] = float(getattr(args, "log_omega_ast", DEFAULT_LOG_OMEGA_AST))
    return truth


def mask_tag(f_min: float, f_max: float, notch_width: float) -> str:
    return f"fmin{fmt_float_for_tag(f_min)}_fmax{f_max:g}_w{notch_width:g}".replace(".", "p")


def fd_table_path(f_min: float, f_max: float, notch_width: float) -> Path:
    return FD_TABLE_DIR / f"fd_equal_fullxyz_table_{mask_tag(f_min, f_max, notch_width)}_{RUN_TAG}.h5"


def fd_data_path(
    realization: int,
    seed: int,
    segments: int,
    f_min: float,
    f_max: float,
    notch_width: float,
    log_omega_ast: float = DEFAULT_LOG_OMEGA_AST,
) -> Path:
    suffix = fd_injection_suffix(log_omega_ast)
    return (
        FD_DATA_DIR
        / f"fd_equal_fullxyz_r{realization:02d}_seed{seed}_seg{segments}_{mask_tag(f_min, f_max, notch_width)}_{RUN_TAG}{suffix}.h5"
    )


def fd_bundle_path(
    realization: int,
    seed: int,
    segments: int,
    f_min: float,
    f_max: float,
    notch_width: float,
    log_omega_ast: float = DEFAULT_LOG_OMEGA_AST,
) -> Path:
    suffix = fd_injection_suffix(log_omega_ast)
    return (
        FD_BUNDLE_DIR
        / f"fd_equal_fullxyz_r{realization:02d}_seed{seed}_seg{segments}_{mask_tag(f_min, f_max, notch_width)}_{RUN_TAG}{suffix}_bundle.h5"
    )


def fd_result_path(bundle: Path, stage: str, nlive: int, fit_model: str) -> Path:
    return FD_RESULT_DIR / f"{bundle.stem}_{stage}_nlive{nlive}_{fit_model}.h5"


def normalize_orbit_type(orbit_type: Optional[str]) -> str:
    orbit_type = orbit_type or "equal"
    if orbit_type not in {"equal", "unequal"}:
        raise ValueError(f"Unsupported TD orbit_type={orbit_type!r}")
    return orbit_type


def td_orbit_type(args: argparse.Namespace) -> str:
    return normalize_orbit_type(getattr(args, "orbit_type", "equal"))


def td_run_tag(orbit_type: str) -> str:
    orbit_type = normalize_orbit_type(orbit_type)
    return TD_EQUAL_RUN_TAG if orbit_type == "equal" else TD_UNEQUAL_RUN_TAG


def td_table_dir(orbit_type: str) -> Path:
    orbit_type = normalize_orbit_type(orbit_type)
    return TD_EQUAL_TABLE_DIR if orbit_type == "equal" else TD_UNEQUAL_TABLE_DIR


def td_data_dir(orbit_type: str) -> Path:
    orbit_type = normalize_orbit_type(orbit_type)
    return TD_EQUAL_DATA_DIR if orbit_type == "equal" else TD_UNEQUAL_DATA_DIR


def td_bundle_dir(orbit_type: str) -> Path:
    orbit_type = normalize_orbit_type(orbit_type)
    return TD_EQUAL_BUNDLE_DIR if orbit_type == "equal" else TD_UNEQUAL_BUNDLE_DIR


def td_phase(orbit_type: str) -> str:
    return f"td_{normalize_orbit_type(orbit_type)}"


def td_method(orbit_type: str) -> str:
    return f"td_{normalize_orbit_type(orbit_type)}_full_xyz"


def td_orbit_dir_for_chunk(args: argparse.Namespace, chunk: int) -> str:
    orbit_type = td_orbit_type(args)
    if orbit_type == "equal":
        template = str(getattr(args, "equal_orbit_dir_template", "") or "")
        if template:
            orbit_dir = template.format(
                chunk=chunk,
                chunk1=chunk + 1,
                orbit_chunk=chunk,
                orbit_chunk1=chunk + 1,
            )
        else:
            orbit_dir = str(getattr(args, "equal_orbit_dir", DEFAULT_EQUAL_ORBIT_DIR))
    else:
        template = str(getattr(args, "unequal_orbit_dir_template", DEFAULT_UNEQUAL_ORBIT_DIR_TEMPLATE))
        orbit_dir = template.format(
            chunk=chunk,
            chunk1=chunk + 1,
            orbit_chunk=chunk,
            orbit_chunk1=chunk + 1,
        )
    if not Path(orbit_dir).exists():
        raise FileNotFoundError(f"TD {orbit_type}-arm orbit_dir for chunk {chunk + 1} does not exist: {orbit_dir}")
    return orbit_dir


def td_segment_mid_time_days(args: argparse.Namespace, segment0: np.ndarray | int) -> np.ndarray:
    return (np.asarray(segment0, dtype=np.float64) + 0.5) * float(args.segment_days)


def validate_td_segment_config(args: argparse.Namespace) -> int:
    if int(args.orbit_chunks) != 360 or int(args.days_per_orbit_chunk) != 1:
        raise ValueError(
            "TD production orbit products must be 360 one-day chunks; "
            f"got orbit_chunks={args.orbit_chunks}, "
            f"days_per_orbit_chunk={args.days_per_orbit_chunk}."
        )
    segments = int(args.orbit_chunks) * int(args.days_per_orbit_chunk)
    if segments != 360:
        raise ValueError(
            "TD production data must contain exactly 360 one-day segments; "
            f"got orbit_chunks={args.orbit_chunks}, "
            f"days_per_orbit_chunk={args.days_per_orbit_chunk}, segments={segments}."
        )
    if not np.isclose(float(args.segment_days), 1.0):
        raise ValueError(f"TD production segment_days must be 1.0, got {args.segment_days}.")
    if int(args.days_per_orbit_chunk) <= 0 or int(args.orbit_chunks) <= 0:
        raise ValueError("TD production orbit_chunks and days_per_orbit_chunk must be positive.")
    if int(getattr(args, "max_parallel", 1)) <= 0:
        raise ValueError(f"TD production max_parallel must be positive, got {args.max_parallel}.")
    return segments


def td_equal_table_path(f_min: float, f_max: float, notch_width: float, orbit_type: str = "equal") -> Path:
    orbit_type = normalize_orbit_type(orbit_type)
    return (
        td_table_dir(orbit_type)
        / f"td_{orbit_type}_fullxyz_table_{mask_tag(f_min, f_max, notch_width)}_{td_run_tag(orbit_type)}.h5"
    )


def td_equal_data_path(
    realization: int,
    seed: int,
    segments: int,
    f_min: float,
    f_max: float,
    notch_width: float,
    log_omega_ast: float = DEFAULT_LOG_OMEGA_AST,
    orbit_type: str = "equal",
) -> Path:
    orbit_type = normalize_orbit_type(orbit_type)
    folder = injection_folder(log_omega_ast)
    return (
        td_data_dir(orbit_type)
        / folder
        / f"td_{orbit_type}_fullxyz_r{realization:02d}_seed{seed}_seg{segments}_{mask_tag(f_min, f_max, notch_width)}_{td_run_tag(orbit_type)}_{folder}.h5"
    )


def td_equal_bundle_path(
    realization: int,
    seed: int,
    segments: int,
    f_min: float,
    f_max: float,
    notch_width: float,
    log_omega_ast: float = DEFAULT_LOG_OMEGA_AST,
    orbit_type: str = "equal",
) -> Path:
    orbit_type = normalize_orbit_type(orbit_type)
    folder = injection_folder(log_omega_ast)
    return (
        td_bundle_dir(orbit_type)
        / folder
        / f"td_{orbit_type}_fullxyz_r{realization:02d}_seed{seed}_seg{segments}_{mask_tag(f_min, f_max, notch_width)}_{td_run_tag(orbit_type)}_{folder}_bundle.h5"
    )


def result_path_for_bundle(bundle: Path, stage: str, nlive: int, fit_model: str) -> Path:
    parts = set(Path(bundle).parts)
    if "td_equal" in parts:
        return TD_EQUAL_RESULT_DIR / f"{bundle.stem}_{stage}_nlive{nlive}_{fit_model}.h5"
    if "td_unequal" in parts:
        return TD_UNEQUAL_RESULT_DIR / f"{bundle.stem}_{stage}_nlive{nlive}_{fit_model}.h5"
    return fd_result_path(bundle, stage, nlive, fit_model)


def bilby_dir_for_bundle(bundle: Path) -> Path:
    parts = set(Path(bundle).parts)
    if "td_equal" in parts:
        return TD_EQUAL_BILBY_DIR
    if "td_unequal" in parts:
        return TD_UNEQUAL_BILBY_DIR
    return FD_BILBY_DIR


def read_frequency_source(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return np.asarray(h5["frequency"][:], dtype=np.float64)


def build_equal_arm_mask(
    frequency: np.ndarray,
    *,
    f_min: float,
    f_max: float,
    notch_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (frequency >= f_min) & (frequency <= f_max)
    rows = []
    max_n = int(math.ceil((f_max + notch_width) * 4.0 * L / c)) + 1
    for n in range(1, max_n + 1):
        center = n * c / (4.0 * L)
        if center > f_max + notch_width:
            continue
        mask &= ~((frequency >= center - notch_width) & (frequency <= center + notch_width))
        rows.append((n, center))
    dtype = np.dtype([("n", np.int16), ("center_hz", np.float64)])
    return mask, np.asarray(rows, dtype=dtype)


def calculate_xyz_csd_one_day(
    x_time: np.ndarray,
    y_time: np.ndarray,
    z_time: np.ndarray,
    *,
    fsample: float,
    segment_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    from Triangle.FFTTools import CSD_window

    channels = [
        np.asarray(x_time[100:-100], dtype=np.float64) / F_LASER,
        np.asarray(y_time[100:-100], dtype=np.float64) / F_LASER,
        np.asarray(z_time[100:-100], dtype=np.float64) / F_LASER,
    ]
    factor = segment_days * DAY * fsample**2 / 2.0
    frequency_ref = None
    matrix = None
    for i in range(3):
        for j in range(3):
            frequency, csd = CSD_window(channels[i], channels[j], fsample=fsample, nbin=1, window_type="hann")
            if frequency_ref is None:
                frequency_ref = frequency
                matrix = np.zeros((len(frequency), 3, 3), dtype=np.complex128)
            elif not np.array_equal(frequency, frequency_ref):
                raise RuntimeError("Frequency grid mismatch while building daily XYZ CSD.")
            matrix[:, i, j] = factor * csd / unit_factor(frequency)
    return np.asarray(frequency_ref, dtype=np.float64), matrix


def split_equal_tdi_days(
    tdi,
    *,
    fsample: float,
    segment_days: float,
    days_per_chunk: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    samples_per_day = int(round(DAY * fsample))
    x = tdi.measurements["X2"]
    y = tdi.measurements["Y2"]
    z = tdi.measurements["Z2"]
    expected = days_per_chunk * samples_per_day
    if len(x) < expected or len(y) < expected or len(z) < expected:
        raise RuntimeError(f"TDI stream too short for {days_per_chunk} days: {len(x)}, {len(y)}, {len(z)}")
    out = []
    for local_day in range(days_per_chunk):
        start = local_day * samples_per_day
        end = (local_day + 1) * samples_per_day
        out.append(calculate_xyz_csd_one_day(x[start:end], y[start:end], z[start:end], fsample=fsample, segment_days=segment_days))
    return out


def calculate_equal_arm_xyz_tables(frequency: np.ndarray, *, response_seed: int, nsource: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(response_seed)
    ra, re, rt = calculate_response_function(
        frequency,
        1,
        orbit_type="equal",
        generation_index=2,
        show_progress=True,
    )
    acc_a, oms_a, acc_e, oms_e, acc_t, oms_t = calculate_transfer_function(
        frequency,
        1,
        orbit_type="equal",
    )
    response_aet = np.stack([ra, re, rt], axis=-1).astype(np.float64)
    acc_aet = np.stack([acc_a, acc_e, acc_t], axis=-1).astype(np.float64)
    oms_aet = np.stack([oms_a, oms_e, oms_t], axis=-1).astype(np.float64)
    response_xyz = aet_diag_to_xyz(response_aet).astype(np.complex128)
    acc_xyz = aet_diag_to_xyz(acc_aet).astype(np.complex128)
    oms_xyz = aet_diag_to_xyz(oms_aet).astype(np.complex128)
    if nsource <= 0:
        raise ValueError("nsource must be positive")
    return response_xyz, acc_xyz, oms_xyz, response_aet, acc_aet, oms_aet


def aet_diag_to_xyz(diag: np.ndarray) -> np.ndarray:
    return np.einsum("ai,...a,aj->...ij", U_AET_FROM_XYZ, diag, U_AET_FROM_XYZ, optimize=True)


def omega_dwd_ast_np(frequency: np.ndarray, theta: np.ndarray, *, include_ast: bool = True) -> np.ndarray:
    dwd = (
        10.0 ** theta[2]
        * (frequency / 0.016) ** theta[3]
        / (1.0 + 10.0 ** theta[4] * (frequency / 0.016) ** theta[5])
    )
    if not include_ast:
        return dwd
    ast = 10.0 ** theta[6] * (frequency / 1.0e-3) ** theta[7]
    return dwd + ast


def sh_dwd_ast_np(frequency: np.ndarray, theta: np.ndarray, *, include_ast: bool = True) -> np.ndarray:
    omega = omega_dwd_ast_np(frequency, theta, include_ast=include_ast)
    return 3.0 * H0**2 / (4.0 * np.pi**2) * omega / frequency**3


def model_covariance_xyz_np(
    frequency: np.ndarray,
    response_xyz: np.ndarray,
    acc_xyz: np.ndarray,
    oms_xyz: np.ndarray,
    theta: np.ndarray,
    *,
    include_ast: bool = True,
) -> np.ndarray:
    signal = sh_dwd_ast_np(frequency, theta, include_ast=include_ast)
    n_acc = theta[0] * 1.0e-15
    delta_x = theta[1] * 1.0e-12
    cov = signal[:, None, None] * response_xyz + (n_acc**2) * acc_xyz + (delta_x**2) * oms_xyz
    return 0.5 * (cov + np.conjugate(np.swapaxes(cov, -1, -2)))


def build_fd_equal_table(args: argparse.Namespace) -> Path:
    table = fd_table_path(args.f_min, args.f_max, args.notch_width)
    if table.exists() and not args.overwrite:
        log(f"FD equal-arm table already exists: {table}")
        return table

    start = stage_start(
        "PHASE=FD_EQUAL_TABLE "
        f"frequency_source={args.frequency_source} f_min={args.f_min} f_max={args.f_max} "
        f"notch_half_width={args.notch_width} nsource={args.nsource}"
    )
    frequency_full = read_frequency_source(args.frequency_source)
    pe_mask, centers = build_equal_arm_mask(
        frequency_full,
        f_min=args.f_min,
        f_max=args.f_max,
        notch_width=args.notch_width,
    )
    frequency = np.asarray(frequency_full[pe_mask], dtype=np.float64)
    if frequency.size == 0:
        raise RuntimeError("PE mask removed all FD equal-arm frequency bins.")

    np.random.seed(args.response_seed)
    ra, re, rt = calculate_response_function(
        frequency,
        1,
        orbit_type="equal",
        generation_index=2,
        show_progress=True,
    )
    acc_a, oms_a, acc_e, oms_e, acc_t, oms_t = calculate_transfer_function(
        frequency,
        1,
        orbit_type="equal",
    )
    response_aet = np.stack([ra, re, rt], axis=-1).astype(np.float64)
    acc_aet = np.stack([acc_a, acc_e, acc_t], axis=-1).astype(np.float64)
    oms_aet = np.stack([oms_a, oms_e, oms_t], axis=-1).astype(np.float64)
    response_xyz = aet_diag_to_xyz(response_aet).astype(np.complex128)
    acc_xyz = aet_diag_to_xyz(acc_aet).astype(np.complex128)
    oms_xyz = aet_diag_to_xyz(oms_aet).astype(np.complex128)

    ensure_parent(table)
    tmp = fresh_tmp_path(table)
    with h5py.File(tmp, "w") as h5:
        h5.attrs["description"] = "Ch.3.2 FD equal-arm second-generation TDI full-XYZ response/noise table"
        h5.attrs["campaign"] = CAMPAIGN
        h5.attrs["run_tag"] = RUN_TAG
        h5.attrs["phase"] = "fd_equal"
        h5.attrs["method"] = "fd_equal_full_xyz"
        h5.attrs["orbit_type"] = "equal"
        h5.attrs["tdi_generation"] = 2
        h5.attrs["channels"] = ",".join(CHANNELS)
        h5.attrs["basis_source"] = "analytic AET diagonal table transformed to XYZ full covariance"
        h5.attrs["orientation"] = "left-conjugated covariance: matrix[i,j] = <d_i^* d_j>"
        h5.attrs["frequency_source"] = repo_path(args.frequency_source)
        h5.attrs["f_min"] = float(args.f_min)
        h5.attrs["f_max"] = float(args.f_max)
        h5.attrs["notch_half_width_hz"] = float(args.notch_width)
        h5.attrs["notch_center_mode"] = "fixed_n_c_over_4L_equal_arm"
        h5.attrs["response_seed"] = int(args.response_seed)
        h5.attrs["n_source"] = int(args.nsource)
        h5.attrs["frequency_bins_full"] = int(frequency_full.size)
        h5.attrs["frequency_bins_kept"] = int(frequency.size)
        h5.attrs["parameter_order"] = ",".join(FIT_NAMES_FULL)
        h5.attrs["parameter_display_order"] = ",".join(FIT_DISPLAY_NAMES_FULL)
        h5.attrs["priors"] = json.dumps({name: [float(lo), float(hi)] for name, (lo, hi) in zip(FIT_NAMES_FULL, PRIOR_BOUNDS_FULL)})
        h5.create_dataset("frequency_full", data=frequency_full)
        h5.create_dataset("frequency", data=frequency)
        h5.create_dataset("pe_frequency_mask", data=pe_mask)
        h5.create_dataset("notch_centers", data=centers)
        h5.create_dataset("response_aet_diag", data=response_aet, compression="gzip", compression_opts=2)
        h5.create_dataset("acc_transfer_aet_diag", data=acc_aet, compression="gzip", compression_opts=2)
        h5.create_dataset("oms_transfer_aet_diag", data=oms_aet, compression="gzip", compression_opts=2)
        h5.create_dataset("response", data=response_xyz, compression="gzip", compression_opts=2)
        h5.create_dataset("acc_transfer", data=acc_xyz, compression="gzip", compression_opts=2)
        h5.create_dataset("oms_transfer", data=oms_xyz, compression="gzip", compression_opts=2)
    tmp.replace(table)
    stage_end(start, f"FILES_CREATED table={table} kept_bins={frequency.size}/{frequency_full.size}")
    return table


def build_fd_equal_data_and_bundle(args: argparse.Namespace) -> Path:
    table = build_fd_equal_table(args)
    truth = truth_full_from_args(args)
    data = fd_data_path(args.realization, args.seed, args.segments, args.f_min, args.f_max, args.notch_width, truth[6])
    bundle = fd_bundle_path(args.realization, args.seed, args.segments, args.f_min, args.f_max, args.notch_width, truth[6])
    if bundle.exists() and not args.overwrite:
        log(f"FD equal-arm bundle already exists: {bundle}")
        return bundle

    start = stage_start(
        "PHASE=FD_EQUAL_DATA "
        f"realization={args.realization} seed={args.seed} segments={args.segments} segment_days={args.segment_days} "
        f"log_Omega_ast={truth[6]}"
    )
    with h5py.File(table, "r") as h5:
        frequency_full = np.asarray(h5["frequency_full"][:], dtype=np.float64)
        frequency = np.asarray(h5["frequency"][:], dtype=np.float64)
        pe_mask = np.asarray(h5["pe_frequency_mask"][:], dtype=bool)
        centers = np.asarray(h5["notch_centers"][:])
        response = np.asarray(h5["response"][:], dtype=np.complex128)
        acc = np.asarray(h5["acc_transfer"][:], dtype=np.complex128)
        oms = np.asarray(h5["oms_transfer"][:], dtype=np.complex128)
        response_aet = np.asarray(h5["response_aet_diag"][:], dtype=np.float64)
        acc_aet = np.asarray(h5["acc_transfer_aet_diag"][:], dtype=np.float64)
        oms_aet = np.asarray(h5["oms_transfer_aet_diag"][:], dtype=np.float64)

    signal = sh_dwd_ast_np(frequency, truth, include_ast=True)[:, None]
    psd_aet = signal * response_aet + (truth[0] * 1.0e-15) ** 2 * acc_aet + (truth[1] * 1.0e-12) ** 2 * oms_aet
    if np.any(psd_aet <= 0.0) or not np.all(np.isfinite(psd_aet)):
        raise RuntimeError("Invalid FD equal-arm AET PSD at injected parameters.")

    factor = args.segment_days * DAY * args.fsample**2 / 2.0
    sigma = np.sqrt(factor * psd_aet / 2.0)
    rng = np.random.default_rng(args.seed)
    real = rng.normal(0.0, sigma[None, :, :], size=(args.segments, frequency.size, 3))
    imag = rng.normal(0.0, sigma[None, :, :], size=(args.segments, frequency.size, 3))
    z_aet = real + 1j * imag
    z_xyz = np.einsum("...a,ai->...i", z_aet, U_AET_FROM_XYZ, optimize=True)
    obs = np.einsum("...i,...j->...ij", np.conjugate(z_xyz), z_xyz, optimize=True).astype(np.complex128)
    segment_seed = np.asarray([args.seed + i for i in range(args.segments)], dtype=np.int64)

    ensure_parent(data)
    tmp_data = fresh_tmp_path(data)
    with h5py.File(tmp_data, "w") as h5:
        h5.attrs["description"] = "Ch.3.2 FD equal-arm second-generation TDI full-XYZ synthetic data"
        h5.attrs["campaign"] = CAMPAIGN
        h5.attrs["run_tag"] = RUN_TAG
        h5.attrs["phase"] = "fd_equal"
        h5.attrs["method"] = "fd_equal_full_xyz"
        h5.attrs["realization_index"] = int(args.realization)
        h5.attrs["seed"] = int(args.seed)
        h5.attrs["segments"] = int(args.segments)
        h5.attrs["segment_days"] = float(args.segment_days)
        h5.attrs["observation_days"] = float(args.segments * args.segment_days)
        h5.attrs["fsample"] = float(args.fsample)
        h5.attrs["likelihood_factor"] = float(factor)
        h5.attrs["orbit_type"] = "equal"
        h5.attrs["tdi_generation"] = 2
        h5.attrs["channels"] = ",".join(CHANNELS)
        h5.attrs["orientation"] = "left-conjugated covariance: matrix[i,j] = <d_i^* d_j>"
        h5.attrs["model"] = MODEL_FULL
        h5.attrs["injected_log_Omega_ast"] = float(truth[6])
        h5.attrs["f_min"] = float(args.f_min)
        h5.attrs["f_max"] = float(args.f_max)
        h5.attrs["notch_half_width_hz"] = float(args.notch_width)
        h5.attrs["notch_center_mode"] = "fixed_n_c_over_4L_equal_arm"
        h5.attrs["parameter_order"] = ",".join(FIT_NAMES_FULL)
        h5.attrs["parameter_display_order"] = ",".join(FIT_DISPLAY_NAMES_FULL)
        h5.create_dataset("frequency_full", data=frequency_full)
        h5.create_dataset("frequency", data=frequency)
        h5.create_dataset("pe_frequency_mask", data=pe_mask)
        h5.create_dataset("notch_centers", data=centers)
        h5.create_dataset("obs_csd_star", data=obs, compression="gzip", compression_opts=2)
        h5.create_dataset("segment_index", data=np.arange(1, args.segments + 1, dtype=np.int64))
        h5.create_dataset("segment_seed", data=segment_seed)
        h5.create_dataset("parameters_true", data=truth)
        h5.create_dataset("parameter_names", data=np.asarray(FIT_NAMES_FULL, dtype="S32"))
        h5.create_dataset("parameter_display_names", data=np.asarray(FIT_DISPLAY_NAMES_FULL, dtype="S32"))
        h5.create_dataset("prior_bounds", data=PRIOR_BOUNDS_FULL)
    tmp_data.replace(data)

    ensure_parent(bundle)
    tmp_bundle = fresh_tmp_path(bundle)
    with h5py.File(tmp_bundle, "w") as h5:
        h5.attrs["description"] = "Self-contained Ch.3.2 FD equal-arm full-XYZ PE bundle"
        h5.attrs["campaign"] = CAMPAIGN
        h5.attrs["run_tag"] = RUN_TAG
        h5.attrs["phase"] = "fd_equal"
        h5.attrs["method"] = "fd_equal_full_xyz"
        h5.attrs["source_table_path"] = repo_path(table)
        h5.attrs["source_data_path"] = repo_path(data)
        h5.attrs["realization_index"] = int(args.realization)
        h5.attrs["seed"] = int(args.seed)
        h5.attrs["segments"] = int(args.segments)
        h5.attrs["segment_days"] = float(args.segment_days)
        h5.attrs["observation_days"] = float(args.segments * args.segment_days)
        h5.attrs["fsample"] = float(args.fsample)
        h5.attrs["likelihood_factor"] = float(factor)
        h5.attrs["orbit_type"] = "equal"
        h5.attrs["tdi_generation"] = 2
        h5.attrs["channels"] = ",".join(CHANNELS)
        h5.attrs["orientation"] = "left-conjugated covariance: matrix[i,j] = <d_i^* d_j>"
        h5.attrs["model"] = MODEL_FULL
        h5.attrs["injected_log_Omega_ast"] = float(truth[6])
        h5.attrs["fit_model_default"] = "full8"
        h5.attrs["table_segment_independent"] = True
        h5.attrs["f_min"] = float(args.f_min)
        h5.attrs["f_max"] = float(args.f_max)
        h5.attrs["notch_half_width_hz"] = float(args.notch_width)
        h5.attrs["notch_center_mode"] = "fixed_n_c_over_4L_equal_arm"
        h5.attrs["parameter_order"] = ",".join(FIT_NAMES_FULL)
        h5.attrs["parameter_display_order"] = ",".join(FIT_DISPLAY_NAMES_FULL)
        h5.attrs["priors"] = json.dumps({name: [float(lo), float(hi)] for name, (lo, hi) in zip(FIT_NAMES_FULL, PRIOR_BOUNDS_FULL)})
        h5.create_dataset("frequency_full", data=frequency_full)
        h5.create_dataset("frequency", data=frequency)
        h5.create_dataset("pe_frequency_mask", data=pe_mask)
        h5.create_dataset("notch_centers", data=centers)
        h5.create_dataset("parameters_true", data=truth)
        h5.create_dataset("parameter_names", data=np.asarray(FIT_NAMES_FULL, dtype="S32"))
        h5.create_dataset("parameter_display_names", data=np.asarray(FIT_DISPLAY_NAMES_FULL, dtype="S32"))
        h5.create_dataset("prior_bounds", data=PRIOR_BOUNDS_FULL)
        h5.create_dataset("segment_index", data=np.arange(1, args.segments + 1, dtype=np.int64))
        h5.create_dataset("segment_seed", data=segment_seed)
        data_group = h5.create_group("data")
        tables_group = h5.create_group("tables")
        data_group.create_dataset("obs_csd_star", data=obs, compression="gzip", compression_opts=2)
        tables_group.create_dataset("response", data=response, compression="gzip", compression_opts=2)
        tables_group.create_dataset("acc_transfer", data=acc, compression="gzip", compression_opts=2)
        tables_group.create_dataset("oms_transfer", data=oms, compression="gzip", compression_opts=2)
    tmp_bundle.replace(bundle)
    stage_end(start, f"FILES_CREATED data={data} bundle={bundle} obs_shape={obs.shape}")
    return bundle


def initialize_td_equal_data_h5(
    path: Path,
    *,
    args: argparse.Namespace,
    frequency_full: np.ndarray,
    frequency: np.ndarray,
    pe_mask: np.ndarray,
    centers: np.ndarray,
) -> None:
    ensure_parent(path)
    tmp = fresh_tmp_path(path)
    segments = validate_td_segment_config(args)
    factor = args.segment_days * DAY * args.fsample**2 / 2.0
    truth = truth_full_from_args(args)
    orbit_type = td_orbit_type(args)
    with h5py.File(tmp, "w") as h5:
        h5.attrs["description"] = f"Ch.3.2 TD {orbit_type}-arm second-generation TDI full-XYZ data"
        h5.attrs["campaign"] = CAMPAIGN
        h5.attrs["run_tag"] = td_run_tag(orbit_type)
        h5.attrs["phase"] = td_phase(orbit_type)
        h5.attrs["method"] = td_method(orbit_type)
        h5.attrs["realization_index"] = int(args.realization)
        h5.attrs["seed"] = int(args.seed)
        h5.attrs["seed_mode"] = "chunk_seed = seed + orbit_chunk; one orbit chunk is a single one-day segment"
        h5.attrs["complete"] = False
        h5.attrs["orbit_chunks"] = int(args.orbit_chunks)
        h5.attrs["days_per_orbit_chunk"] = int(args.days_per_orbit_chunk)
        h5.attrs["td_segments"] = int(segments)
        h5.attrs["segment_count_mode"] = "360 explicit one-day segments"
        h5.attrs["orbit_parameter_mode"] = "explicit_orbit_dir_per_one_day_orbit_chunk"
        h5.attrs["orbit_file_grouping"] = "one orbit directory is loaded per one-day orbit/data segment"
        h5.attrs["equal_orbit_dir"] = str(getattr(args, "equal_orbit_dir", DEFAULT_EQUAL_ORBIT_DIR))
        h5.attrs["unequal_orbit_dir_template"] = str(
            getattr(args, "unequal_orbit_dir_template", DEFAULT_UNEQUAL_ORBIT_DIR_TEMPLATE)
        )
        h5.attrs["segments"] = int(segments)
        h5.attrs["segment_days"] = float(args.segment_days)
        h5.attrs["observation_days"] = float(segments * args.segment_days)
        h5.attrs["fsample"] = float(args.fsample)
        h5.attrs["likelihood_factor"] = float(factor)
        h5.attrs["orbit_type"] = orbit_type
        h5.attrs["tdi_generation"] = 2
        h5.attrs["channels"] = ",".join(CHANNELS)
        h5.attrs["orientation"] = "left-conjugated covariance: matrix[i,j] = <d_i^* d_j>"
        h5.attrs["model"] = MODEL_FULL
        h5.attrs["injected_log_Omega_ast"] = float(truth[6])
        h5.attrs["f_min"] = float(args.f_min)
        h5.attrs["f_max"] = float(args.f_max)
        h5.attrs["notch_half_width_hz"] = float(args.notch_width)
        h5.attrs["notch_center_mode"] = "fixed_n_c_over_4L_equal_arm"
        h5.attrs["frequency_bins_full"] = int(frequency_full.size)
        h5.attrs["frequency_bins_kept"] = int(frequency.size)
        h5.attrs["parameter_order"] = ",".join(FIT_NAMES_FULL)
        h5.attrs["parameter_display_order"] = ",".join(FIT_DISPLAY_NAMES_FULL)
        h5.attrs["priors"] = json.dumps({name: [float(lo), float(hi)] for name, (lo, hi) in zip(FIT_NAMES_FULL, PRIOR_BOUNDS_FULL)})
        h5.create_dataset("frequency_full", data=frequency_full)
        h5.create_dataset("frequency", data=frequency)
        h5.create_dataset("pe_frequency_mask", data=pe_mask)
        h5.create_dataset("notch_centers", data=centers)
        h5.create_dataset(
            "obs_csd_star",
            shape=(segments, len(frequency), 3, 3),
            dtype=np.complex128,
            chunks=(1, min(len(frequency), 4096), 3, 3),
            compression="gzip",
            compression_opts=2,
        )
        h5.create_dataset("segment_index", data=np.arange(1, segments + 1, dtype=np.int64))
        h5.create_dataset("segment_complete", data=np.zeros(segments, dtype=bool))
        h5.create_dataset("orbit_chunk_complete", data=np.zeros(args.orbit_chunks, dtype=bool))
        h5.create_dataset("segment_seed", data=np.zeros(segments, dtype=np.int64))
        h5.create_dataset("orbit_chunk_seed", data=np.zeros(args.orbit_chunks, dtype=np.int64))
        h5.create_dataset("segment_mid_time_days", data=td_segment_mid_time_days(args, np.arange(segments)))
        h5.create_dataset(
            "orbit_chunk_index",
            data=np.repeat(np.arange(1, args.orbit_chunks + 1, dtype=np.int64), args.days_per_orbit_chunk),
        )
        str_dtype = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset("segment_orbit_dir", shape=(segments,), dtype=str_dtype)
        h5.create_dataset("orbit_chunk_dir", shape=(args.orbit_chunks,), dtype=str_dtype)
        h5.create_dataset("parameters_true", data=truth)
        h5.create_dataset("parameter_names", data=np.asarray(FIT_NAMES_FULL, dtype="S32"))
        h5.create_dataset("parameter_display_names", data=np.asarray(FIT_DISPLAY_NAMES_FULL, dtype="S32"))
        h5.create_dataset("prior_bounds", data=PRIOR_BOUNDS_FULL)
    tmp.replace(path)


def generate_td_equal_chunk(args: argparse.Namespace, chunk: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from py_code.lisa_generate import generate_TDI_data

    orbit_type = td_orbit_type(args)
    validate_td_segment_config(args)
    phase_prefix = f"TD_{orbit_type.upper()}"
    chunk_seed = args.seed + chunk
    orbit_dir = td_orbit_dir_for_chunk(args, chunk)
    start_seg = chunk * args.days_per_orbit_chunk
    end_seg = start_seg + args.days_per_orbit_chunk
    np.random.seed(chunk_seed)
    start = stage_start(
        f"PHASE={phase_prefix}_DATA_CHUNK "
        f"realization={args.realization} chunk={chunk + 1:02d}/{args.orbit_chunks} "
        f"segments={start_seg + 1:03d}-{end_seg:03d}/360 segment_days={args.segment_days} "
        f"seed={chunk_seed} orbit_dir={orbit_dir}"
    )
    tdi = generate_TDI_data(
        noise=True,
        signal_type=MODEL_FULL,
        day_number=args.days_per_orbit_chunk,
        fsample=args.fsample,
        orbit_type=orbit_type,
        orbit_dir=orbit_dir,
        parameters=truth_full_from_args(args),
    )
    try:
        measured = split_equal_tdi_days(
            tdi,
            fsample=args.fsample,
            segment_days=args.segment_days,
            days_per_chunk=args.days_per_orbit_chunk,
        )
    finally:
        del tdi
    stage_end(start, f"TD {orbit_type}-arm chunk complete chunk={chunk + 1:02d}")
    return measured


def write_td_equal_chunk(
    args: argparse.Namespace,
    data: Path,
    chunk: int,
    measured: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    orbit_type = td_orbit_type(args)
    if not data.exists():
        frequency_full = measured[0][0]
        pe_mask, centers = build_equal_arm_mask(
            frequency_full,
            f_min=args.f_min,
            f_max=args.f_max,
            notch_width=args.notch_width,
        )
        frequency = np.asarray(frequency_full[pe_mask], dtype=np.float64)
        if frequency.size == 0:
            raise RuntimeError(f"PE mask removed all TD {orbit_type}-arm frequency bins.")
        initialize_td_equal_data_h5(
            data,
            args=args,
            frequency_full=frequency_full,
            frequency=frequency,
            pe_mask=pe_mask,
            centers=centers,
        )
    with h5py.File(data, "r+") as h5:
        if bool(h5["orbit_chunk_complete"][chunk]):
            return
        frequency_full = np.asarray(h5["frequency_full"][:], dtype=np.float64)
        frequency = np.asarray(h5["frequency"][:], dtype=np.float64)
        pe_mask = np.asarray(h5["pe_frequency_mask"][:], dtype=bool)
        start_seg = chunk * args.days_per_orbit_chunk
        chunk_seed = args.seed + chunk
        orbit_dir = td_orbit_dir_for_chunk(args, chunk)
        for local_day, (seg_frequency, matrix) in enumerate(measured):
            if not np.array_equal(seg_frequency, frequency_full):
                raise RuntimeError(f"TD {orbit_type}-arm frequency grid changed at chunk {chunk}, local_day {local_day}")
            if not np.array_equal(seg_frequency[pe_mask], frequency):
                raise RuntimeError(f"TD {orbit_type}-arm masked frequency grid changed at chunk {chunk}, local_day {local_day}")
            global_seg = start_seg + local_day
            h5["obs_csd_star"][global_seg] = matrix[pe_mask]
            h5["segment_complete"][global_seg] = True
            h5["segment_seed"][global_seg] = chunk_seed
            h5["segment_orbit_dir"][global_seg] = orbit_dir
        h5["orbit_chunk_complete"][chunk] = True
        h5["orbit_chunk_seed"][chunk] = chunk_seed
        h5["orbit_chunk_dir"][chunk] = orbit_dir
        h5.attrs["complete"] = bool(np.all(h5["segment_complete"][:]))
        h5.flush()


def build_td_equal_data(args: argparse.Namespace) -> Path:
    orbit_type = td_orbit_type(args)
    segments = validate_td_segment_config(args)
    data = td_equal_data_path(
        args.realization,
        args.seed,
        segments,
        args.f_min,
        args.f_max,
        args.notch_width,
        args.log_omega_ast,
        orbit_type,
    )
    if data.exists():
        with h5py.File(data, "r") as h5:
            if bool(h5.attrs.get("complete", False)):
                log(f"TD {orbit_type}-arm data already complete: {data}")
                return data

    pending_chunks = []
    for chunk in range(args.orbit_chunks):
        if data.exists():
            with h5py.File(data, "r") as h5:
                if bool(h5["orbit_chunk_complete"][chunk]):
                    continue
        pending_chunks.append(chunk)

    max_parallel = max(1, int(getattr(args, "max_parallel", 1)))
    max_workers = min(max_parallel, len(pending_chunks)) if pending_chunks else 0
    log(
        f"TD_{orbit_type.upper()}_DATA_PENDING file={data} "
        f"pending_chunks={len(pending_chunks)}/{args.orbit_chunks} max_parallel={max_workers}"
    )
    if max_workers <= 1:
        for chunk in pending_chunks:
            measured = generate_td_equal_chunk(args, chunk)
            write_td_equal_chunk(args, data, chunk, measured)
            log(f"TD_{orbit_type.upper()}_DATA_PROGRESS file={data} chunks={chunk + 1}/{args.orbit_chunks}")
        return data

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(generate_td_equal_chunk, args, chunk): chunk for chunk in pending_chunks}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            measured = future.result()
            write_td_equal_chunk(args, data, chunk, measured)
            completed += 1
            log(
                f"TD_{orbit_type.upper()}_DATA_PROGRESS file={data} "
                f"parallel_completed={completed}/{len(pending_chunks)} "
                f"chunk={chunk + 1}/{args.orbit_chunks}"
            )
    return data


def build_td_equal_table(args: argparse.Namespace, data: Path) -> Path:
    orbit_type = td_orbit_type(args)
    table = td_equal_table_path(args.f_min, args.f_max, args.notch_width, orbit_type)
    ensure_parent(table)
    lock_path = table.with_suffix(table.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build_td_equal_table_locked(args, data, orbit_type, table)


def _build_td_equal_table_locked(
    args: argparse.Namespace,
    data: Path,
    orbit_type: str,
    table: Path,
) -> Path:
    if table.exists():
        log(f"TD {orbit_type}-arm table already exists: {table}")
        return table

    start = stage_start(
        f"PHASE=TD_{orbit_type.upper()}_TABLE data={data} response_seed={args.response_seed} nsource={args.nsource}"
    )
    with h5py.File(data, "r") as h5:
        frequency = np.asarray(h5["frequency"][:], dtype=np.float64)
        frequency_full = np.asarray(h5["frequency_full"][:], dtype=np.float64)
        pe_mask = np.asarray(h5["pe_frequency_mask"][:], dtype=bool)
        centers = np.asarray(h5["notch_centers"][:])

    copied_from_fd = False
    copied_from_precomputed = False
    fd_table = fd_table_path(args.f_min, args.f_max, args.notch_width)
    precomputed_source = ""
    response = acc = oms = response_aet = acc_aet = oms_aet = None

    if TD_FULLFREQ_PRECOMPUTE.exists():
        with h5py.File(TD_FULLFREQ_PRECOMPUTE, "r") as h5:
            if orbit_type in h5:
                group = h5[orbit_type]
                table_frequency_full = np.asarray(group["frequency"][:], dtype=np.float64)
                if np.array_equal(table_frequency_full, frequency_full):
                    response = np.asarray(group["response"][:, pe_mask], dtype=np.complex128)
                    acc = np.asarray(group["acc_transfer"][:, pe_mask], dtype=np.complex128)
                    oms = np.asarray(group["oms_transfer"][:, pe_mask], dtype=np.complex128)
                    copied_from_precomputed = True
                    precomputed_source = repo_path(TD_FULLFREQ_PRECOMPUTE)
                else:
                    log(
                        f"TD {orbit_type}-arm precomputed frequency grid does not match data: "
                        f"{TD_FULLFREQ_PRECOMPUTE}"
                    )

    if response is None and orbit_type == "equal":
        if fd_table.exists():
            with h5py.File(fd_table, "r") as h5:
                if np.array_equal(np.asarray(h5["frequency"][:], dtype=np.float64), frequency):
                    response = np.asarray(h5["response"][:], dtype=np.complex128)
                    acc = np.asarray(h5["acc_transfer"][:], dtype=np.complex128)
                    oms = np.asarray(h5["oms_transfer"][:], dtype=np.complex128)
                    response_aet = np.asarray(h5["response_aet_diag"][:], dtype=np.float64)
                    acc_aet = np.asarray(h5["acc_transfer_aet_diag"][:], dtype=np.float64)
                    oms_aet = np.asarray(h5["oms_transfer_aet_diag"][:], dtype=np.float64)
                    copied_from_fd = True
                else:
                    response = acc = oms = None

        if response is None:
            response, acc, oms, response_aet, acc_aet, oms_aet = calculate_equal_arm_xyz_tables(
                frequency,
                response_seed=args.response_seed,
                nsource=args.nsource,
            )

    if response is None:
        raise RuntimeError(f"Could not build TD {orbit_type}-arm response/noise table.")

    tmp = fresh_tmp_path(table)
    with h5py.File(tmp, "w") as h5:
        h5.attrs["description"] = f"Ch.3.2 TD {orbit_type}-arm second-generation TDI full-XYZ response/noise table"
        h5.attrs["campaign"] = CAMPAIGN
        h5.attrs["run_tag"] = td_run_tag(orbit_type)
        h5.attrs["phase"] = td_phase(orbit_type)
        h5.attrs["method"] = td_method(orbit_type)
        h5.attrs["orbit_type"] = orbit_type
        h5.attrs["tdi_generation"] = 2
        h5.attrs["channels"] = ",".join(CHANNELS)
        h5.attrs["basis_source"] = (
            "precomputed 360-segment full XYZ 3x3 covariance tables"
            if copied_from_precomputed
            else "analytic AET diagonal table transformed to XYZ full covariance"
        )
        h5.attrs["copied_from_fd_equal_table"] = bool(copied_from_fd)
        h5.attrs["source_fd_table_path"] = repo_path(fd_table) if copied_from_fd else ""
        h5.attrs["copied_from_fullfreq_precompute"] = bool(copied_from_precomputed)
        h5.attrs["source_fullfreq_precompute_path"] = precomputed_source
        h5.attrs["table_segment_independent"] = bool(response.ndim == 3)
        h5.attrs["orientation"] = "left-conjugated covariance: matrix[i,j] = <d_i^* d_j>"
        h5.attrs["f_min"] = float(args.f_min)
        h5.attrs["f_max"] = float(args.f_max)
        h5.attrs["notch_half_width_hz"] = float(args.notch_width)
        h5.attrs["notch_center_mode"] = "fixed_n_c_over_4L_equal_arm"
        h5.attrs["response_seed"] = int(args.response_seed)
        h5.attrs["n_source"] = int(args.nsource)
        h5.attrs["frequency_bins_full"] = int(frequency_full.size)
        h5.attrs["frequency_bins_kept"] = int(frequency.size)
        h5.attrs["parameter_order"] = ",".join(FIT_NAMES_FULL)
        h5.attrs["parameter_display_order"] = ",".join(FIT_DISPLAY_NAMES_FULL)
        h5.attrs["priors"] = json.dumps({name: [float(lo), float(hi)] for name, (lo, hi) in zip(FIT_NAMES_FULL, PRIOR_BOUNDS_FULL)})
        h5.create_dataset("frequency_full", data=frequency_full)
        h5.create_dataset("frequency", data=frequency)
        h5.create_dataset("pe_frequency_mask", data=pe_mask)
        h5.create_dataset("notch_centers", data=centers)
        if orbit_type == "equal" and response_aet is not None:
            h5.create_dataset("response_aet_diag", data=response_aet, compression="gzip", compression_opts=2)
            h5.create_dataset("acc_transfer_aet_diag", data=acc_aet, compression="gzip", compression_opts=2)
            h5.create_dataset("oms_transfer_aet_diag", data=oms_aet, compression="gzip", compression_opts=2)
        h5.create_dataset("response", data=response, compression="gzip", compression_opts=2)
        h5.create_dataset("acc_transfer", data=acc, compression="gzip", compression_opts=2)
        h5.create_dataset("oms_transfer", data=oms, compression="gzip", compression_opts=2)
    tmp.replace(table)
    stage_end(
        start,
        f"FILES_CREATED table={table} copied_from_fd={copied_from_fd} "
        f"copied_from_precomputed={copied_from_precomputed}",
    )
    return table


def build_td_equal_bundle(args: argparse.Namespace) -> Path:
    orbit_type = td_orbit_type(args)
    segments = validate_td_segment_config(args)
    data = build_td_equal_data(args)
    table = build_td_equal_table(args, data)
    bundle = td_equal_bundle_path(
        args.realization,
        args.seed,
        segments,
        args.f_min,
        args.f_max,
        args.notch_width,
        args.log_omega_ast,
        orbit_type,
    )
    if bundle.exists():
        log(f"TD {orbit_type}-arm bundle already exists: {bundle}")
        return bundle

    start = stage_start(f"PHASE=TD_{orbit_type.upper()}_BUNDLE data={data} table={table}")
    with h5py.File(data, "r") as dh5, h5py.File(table, "r") as th5:
        frequency = np.asarray(dh5["frequency"][:], dtype=np.float64)
        if not np.array_equal(frequency, np.asarray(th5["frequency"][:], dtype=np.float64)):
            raise RuntimeError(f"TD {orbit_type}-arm table frequency grid does not match data.")
        obs = np.asarray(dh5["obs_csd_star"][:], dtype=np.complex128)
        response = np.asarray(th5["response"][:], dtype=np.complex128)
        acc = np.asarray(th5["acc_transfer"][:], dtype=np.complex128)
        oms = np.asarray(th5["oms_transfer"][:], dtype=np.complex128)
        ensure_parent(bundle)
        tmp = fresh_tmp_path(bundle)
        with h5py.File(tmp, "w") as h5:
            for key, value in dh5.attrs.items():
                h5.attrs[key] = value
            h5.attrs["description"] = f"Self-contained Ch.3.2 TD {orbit_type}-arm full-XYZ PE bundle"
            h5.attrs["source_table_path"] = repo_path(table)
            h5.attrs["source_data_path"] = repo_path(data)
            h5.attrs["fit_model_default"] = "full8"
            h5.attrs["table_segment_independent"] = orbit_type == "equal"
            h5.attrs["priors"] = json.dumps({name: [float(lo), float(hi)] for name, (lo, hi) in zip(FIT_NAMES_FULL, PRIOR_BOUNDS_FULL)})
            for name in (
                "frequency_full",
                "frequency",
                "pe_frequency_mask",
                "notch_centers",
                "parameters_true",
                "parameter_names",
                "parameter_display_names",
                "prior_bounds",
                "segment_index",
                "segment_seed",
                "orbit_chunk_seed",
                "segment_mid_time_days",
                "orbit_chunk_index",
                "segment_orbit_dir",
                "orbit_chunk_dir",
            ):
                if name in dh5:
                    dh5.copy(name, h5, name=name)
            data_group = h5.create_group("data")
            tables_group = h5.create_group("tables")
            data_group.create_dataset("obs_csd_star", data=obs, compression="gzip", compression_opts=2)
            tables_group.create_dataset("response", data=response, compression="gzip", compression_opts=2)
            tables_group.create_dataset("acc_transfer", data=acc, compression="gzip", compression_opts=2)
            tables_group.create_dataset("oms_transfer", data=oms, compression="gzip", compression_opts=2)
        tmp.replace(bundle)
    stage_end(start, f"FILES_CREATED bundle={bundle} obs_shape={obs.shape}")
    return bundle


def validate_bundle(path: Path, overwrite: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r+") as h5:
        if "validation" in h5 and not overwrite:
            log(f"Validation already exists in {path}")
            return
        frequency = np.asarray(h5["frequency"][:], dtype=np.float64)
        obs = np.asarray(h5["data/obs_csd_star"][:], dtype=np.complex128)
        response = np.asarray(h5["tables/response"][:], dtype=np.complex128)
        acc = np.asarray(h5["tables/acc_transfer"][:], dtype=np.complex128)
        oms = np.asarray(h5["tables/oms_transfer"][:], dtype=np.complex128)
        truth = np.asarray(h5["parameters_true"][:], dtype=np.float64)
        factor = float(h5.attrs["likelihood_factor"])
        cov_one = model_covariance_xyz_np(frequency, response, acc, oms, truth, include_ast=True)
        eig = np.linalg.eigvalsh(cov_one)
        min_eig = float(np.min(np.real(eig)))
        hermitian_cov = float(np.max(np.abs(cov_one - np.conjugate(np.swapaxes(cov_one, -1, -2)))))
        hermitian_obs = float(np.max(np.abs(obs[: min(4, obs.shape[0])] - np.conjugate(np.swapaxes(obs[: min(4, obs.shape[0])], -1, -2)))))
        sign, logabsdet = np.linalg.slogdet(cov_one)
        if cov_one.ndim == 3:
            cov_for_solve = cov_one[None, :, :, :]
            logabsdet_for_sum = np.real(logabsdet)[None, :]
        else:
            cov_for_solve = cov_one
            logabsdet_for_sum = np.real(logabsdet)
        solved = np.linalg.solve(cov_for_solve, obs)
        quad = np.real(np.trace(solved, axis1=-2, axis2=-1))
        log_const = np.log(np.pi**3) + 3.0 * np.log(factor)
        truth_loglike = -float(np.sum(log_const + logabsdet_for_sum + quad / factor))
        ok = bool(np.all(np.real(sign) > 0.0) and np.isfinite(truth_loglike) and min_eig > 0.0)
        if "validation" in h5:
            del h5["validation"]
        group = h5.create_group("validation")
        group.create_dataset("theta_truth", data=truth)
        group.create_dataset("truth_loglike", data=truth_loglike)
        group.create_dataset("min_model_eigenvalue", data=min_eig)
        group.create_dataset("max_model_hermitian_residual", data=hermitian_cov)
        group.create_dataset("max_obs_hermitian_residual_sample", data=hermitian_obs)
        group.attrs["ok"] = ok
        group.attrs["validated_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        group.attrs["frequency_bins_kept"] = int(frequency.size)
        group.attrs["segments"] = int(h5.attrs["segments"])
    with h5py.File(path, "r") as h5:
        phase = str(h5.attrs.get("phase", "unknown"))
    log(f"PHASE=BUNDLE_VALIDATE phase={phase} bundle={path} ok={ok} truth_loglike={truth_loglike:.6e} min_eig={min_eig:.6e}")


def result_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as h5:
            return "pe" in h5 and "trace" in h5["pe"] and h5["pe/trace"].shape[0] > 0
    except OSError:
        return False


def fit_names_for_model(fit_model: str) -> tuple[str, ...]:
    if fit_model == "full8":
        return FIT_NAMES_FULL
    if fit_model == "dwd_only":
        return FIT_NAMES_BF
    raise ValueError(f"Unsupported fit_model={fit_model!r}")


def theta_from_fit_parameters(fit_model: str, values: dict[str, float]) -> np.ndarray:
    theta = TRUTH_FULL.copy()
    for name in fit_names_for_model(fit_model):
        theta[FIT_NAMES_FULL.index(name)] = float(values[name])
    if fit_model == "dwd_only":
        theta[6] = -100.0
        theta[7] = 0.0
    return theta


def run_pe(args: argparse.Namespace) -> Path:
    bundle = Path(args.bundle)
    if not bundle.exists():
        raise FileNotFoundError(bundle)
    out = result_path_for_bundle(bundle, args.stage, args.nlive, args.fit_model)
    if result_complete(out) and not args.overwrite:
        log(f"PE result already complete: {out}")
        return out

    import bilby
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    log(f"JAX devices: {jax.devices()}")

    with h5py.File(bundle, "r") as h5:
        frequency = np.asarray(h5["frequency"][:], dtype=np.float64)
        obs_np = np.asarray(h5["data/obs_csd_star"][:], dtype=np.complex128)
        response_np = np.asarray(h5["tables/response"][:], dtype=np.complex128)
        acc_np = np.asarray(h5["tables/acc_transfer"][:], dtype=np.complex128)
        oms_np = np.asarray(h5["tables/oms_transfer"][:], dtype=np.complex128)
        truth = np.asarray(h5["parameters_true"][:], dtype=np.float64)
        factor = float(h5.attrs["likelihood_factor"])
        segments = int(h5.attrs["segments"])

    fit_names = fit_names_for_model(args.fit_model)
    include_ast = args.fit_model == "full8"
    frequency_j = jnp.asarray(frequency, dtype=jnp.float64)
    obs_j = jnp.asarray(obs_np, dtype=jnp.complex128)
    response_j = jnp.asarray(response_np, dtype=jnp.complex128)
    acc_j = jnp.asarray(acc_np, dtype=jnp.complex128)
    oms_j = jnp.asarray(oms_np, dtype=jnp.complex128)
    table_segment_independent = response_np.ndim == 3
    log_const = np.log(np.pi**3) + 3.0 * np.log(factor)
    cov_shape = (segments, frequency.size, 3, 3)

    @jax.jit
    def sh_model(theta):
        dwd = (
            10.0 ** theta[2]
            * (frequency_j / 0.016) ** theta[3]
            / (1.0 + 10.0 ** theta[4] * (frequency_j / 0.016) ** theta[5])
        )
        if include_ast:
            ast = 10.0 ** theta[6] * (frequency_j / 1.0e-3) ** theta[7]
        else:
            ast = 0.0
        omega = dwd + ast
        return 3.0 * H0**2 / (4.0 * jnp.pi**2) * omega / frequency_j**3

    @jax.jit
    def covariance_one(theta):
        n_acc = theta[0] * 1.0e-15
        delta_x = theta[1] * 1.0e-12
        cov = sh_model(theta)[:, None, None] * response_j + (n_acc**2) * acc_j + (delta_x**2) * oms_j
        return 0.5 * (cov + jnp.conjugate(jnp.swapaxes(cov, -1, -2)))

    @jax.jit
    def loglike(theta):
        cov1 = covariance_one(theta)
        sign, logabsdet = jnp.linalg.slogdet(cov1)
        cov = jnp.broadcast_to(cov1[None, :, :, :], cov_shape) if table_segment_independent else cov1
        solved = jnp.linalg.solve(cov, obs_j)
        quad = jnp.real(jnp.trace(solved, axis1=-2, axis2=-1))
        logabsdet_for_sum = jnp.real(logabsdet)[None, :] if table_segment_independent else jnp.real(logabsdet)
        value = -jnp.sum(log_const + logabsdet_for_sum + quad / factor)
        invalid = (
            jnp.any(jnp.real(sign) <= 0.0)
            | jnp.any(~jnp.isfinite(jnp.real(logabsdet)))
            | jnp.any(~jnp.isfinite(quad))
            | (~jnp.isfinite(value))
        )
        return jnp.where(invalid, -jnp.inf, value)

    theta_truth_for_stage = truth.copy()
    if args.fit_model == "dwd_only":
        theta_truth_for_stage[6] = -100.0
        theta_truth_for_stage[7] = 0.0
    start = stage_start(
        "PHASE=FD_EQUAL_PE_COMPILE "
        f"bundle={bundle} stage={args.stage} fit_model={args.fit_model} nlive={args.nlive}"
    )
    truth_loglike = float(loglike(jnp.asarray(theta_truth_for_stage, dtype=jnp.float64)))
    stage_end(start, f"Truth loglike={truth_loglike:.6e}")
    if args.check_only:
        return out

    priors = bilby.prior.PriorDict()
    for name in fit_names:
        lo, hi = para_range[name][0], para_range[name][1]
        priors[name] = bilby.prior.Uniform(lo, hi, name=name)

    class Ch32Likelihood(bilby.Likelihood):
        def __init__(self):
            super().__init__(parameters={name: para_range[name][2] for name in fit_names})

        def log_likelihood(self):
            theta = theta_from_fit_parameters(args.fit_model, self.parameters)
            return float(loglike(jnp.asarray(theta, dtype=jnp.float64)))

    label = out.stem
    start = stage_start(
        "PHASE=FD_EQUAL_PE_RUN "
        f"stage={args.stage} fit_model={args.fit_model} nlive={args.nlive} dlogz={args.dlogz} walks={args.walks}"
    )
    result = bilby.run_sampler(
        likelihood=Ch32Likelihood(),
        priors=priors,
        sampler="dynesty",
        nlive=args.nlive,
        walks=args.walks,
        npool=1,
        outdir=str(bilby_dir_for_bundle(bundle)),
        label=label,
        clean=not bool(getattr(args, "resume", False)),
        resume=bool(getattr(args, "resume", False)),
        dlogz=args.dlogz,
        update_interval=100,
    )
    stage_end(start, f"Bilby.dynesty completed label={label}")

    trace = np.column_stack([np.asarray(result.posterior[name], dtype=np.float64) for name in fit_names])
    truth_fit = np.asarray([truth[FIT_NAMES_FULL.index(name)] for name in fit_names], dtype=np.float64)
    mean = trace.mean(axis=0)
    median = np.median(trace, axis=0)
    std = trace.std(axis=0)
    bias = mean - truth_fit
    rel_unc = std / np.maximum(np.abs(truth_fit), 1.0e-12)

    ensure_parent(out)
    tmp = fresh_tmp_path(out)
    with h5py.File(bundle, "r") as src, h5py.File(tmp, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for name in src:
            src.copy(name, dst, name=name)
        dst.attrs["pe_source_bundle"] = repo_path(bundle)
        dst.attrs["pe_stage"] = args.stage
        dst.attrs["pe_fit_model"] = args.fit_model
        group = dst.create_group("pe")
        group.create_dataset("trace", data=trace)
        group.create_dataset("fit_parameter_names", data=np.asarray(fit_names, dtype="S32"))
        group.create_dataset("truth_fit", data=truth_fit)
        group.create_dataset("posterior_mean", data=mean)
        group.create_dataset("posterior_median", data=median)
        group.create_dataset("posterior_std", data=std)
        group.create_dataset("bias", data=bias)
        group.create_dataset("relative_uncertainty", data=rel_unc)
        group.create_dataset("evidence", data=float(result.log_evidence))
        group.create_dataset("truth_loglike", data=truth_loglike)
        group.attrs["sampler"] = "Bilby.dynesty"
        group.attrs["nlive"] = int(args.nlive)
        group.attrs["dlogz"] = float(args.dlogz)
        group.attrs["walks"] = int(args.walks)
        group.attrs["label"] = label
        group.attrs["stage"] = args.stage
        group.attrs["fit_model"] = args.fit_model
        group.attrs["include_ast"] = include_ast
        group.attrs["frequency_bins_kept"] = int(frequency.size)
        group.attrs["segments"] = int(segments)
        group.attrs["likelihood_log_const"] = float(log_const)
        group.attrs["likelihood_factor"] = float(factor)
        group.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    tmp.replace(out)
    log(f"FILES_CREATED pe_result={out} trace_shape={trace.shape} evidence={float(result.log_evidence):.6e}")
    return out


def inspect_pe(args: argparse.Namespace) -> dict[str, object]:
    result_path = Path(args.result)
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    with h5py.File(result_path, "r+") as h5:
        group = h5["pe"]
        fit_names = [name.decode("utf-8") for name in group["fit_parameter_names"][:]]
        truth_fit = np.asarray(group["truth_fit"][:], dtype=np.float64)
        mean = np.asarray(group["posterior_mean"][:], dtype=np.float64)
        median = np.asarray(group["posterior_median"][:], dtype=np.float64)
        evidence = float(group["evidence"][()])
        report: dict[str, object] = {
            "result": str(result_path),
            "stage": str(group.attrs.get("stage", "")),
            "fit_model": str(group.attrs.get("fit_model", "")),
            "fit_parameter_names": fit_names,
            "posterior_mean": {name: float(value) for name, value in zip(fit_names, mean)},
            "posterior_median": {name: float(value) for name, value in zip(fit_names, median)},
            "truth_fit": {name: float(value) for name, value in zip(fit_names, truth_fit)},
            "log_evidence": evidence,
        }
        if "log_Omega_ast" in fit_names:
            idx = fit_names.index("log_Omega_ast")
            recovered = float(mean[idx])
            injected = float(truth_fit[idx])
            delta = recovered - injected
            accepted = abs(delta) <= args.gate_tolerance
            report.update(
                {
                    "gate_parameter": "log_Omega_ast",
                    "gate_recovered_mean": recovered,
                    "gate_injected": injected,
                    "gate_delta": float(delta),
                    "gate_tolerance": float(args.gate_tolerance),
                    "gate_accepted": bool(accepted),
                }
            )
            if "gate" in h5:
                del h5["gate"]
            gate = h5.create_group("gate")
            gate.attrs["parameter"] = "log_Omega_ast"
            gate.attrs["recovered_mean"] = recovered
            gate.attrs["injected"] = injected
            gate.attrs["delta"] = float(delta)
            gate.attrs["tolerance"] = float(args.gate_tolerance)
            gate.attrs["accepted"] = bool(accepted)
            gate.attrs["evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        else:
            report["gate_accepted"] = None

    if args.report:
        out = Path(args.report)
    else:
        out = LOG_ROOT / f"{result_path.stem}_gate.json"
    ensure_parent(out)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log(f"FILES_CREATED gate_report={out}")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def combine_bayes_factor(args: argparse.Namespace) -> dict[str, object]:
    full = Path(args.full_result)
    bf = Path(args.bf_result)
    if not full.exists() or not bf.exists():
        raise FileNotFoundError(f"Missing full or BF result: {full}, {bf}")
    with h5py.File(full, "r") as f1, h5py.File(bf, "r") as f0:
        evidence_full = float(f1["pe/evidence"][()])
        evidence_bf = float(f0["pe/evidence"][()])
    report = {
        "full_result": str(full),
        "bf_result": str(bf),
        "log_evidence_full8": evidence_full,
        "log_evidence_dwd_only": evidence_bf,
        "ln_bayes_factor_full8_over_dwd_only": evidence_full - evidence_bf,
    }
    out = Path(args.report) if args.report else LOG_ROOT / f"{full.stem}_lnbf.json"
    ensure_parent(out)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log(f"FILES_CREATED bf_report={out} lnBF={report['ln_bayes_factor_full8_over_dwd_only']:.6e}")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def print_summary() -> None:
    summary = {
        "campaign": CAMPAIGN,
        "run_tag": RUN_TAG,
        "data_root": str(DATA_ROOT),
        "log_root": str(LOG_ROOT),
        "figure_root": str(FIG_ROOT),
        "fd_root": str(FD_ROOT),
        "td_equal_root": str(TD_EQUAL_ROOT),
        "td_unequal_root": str(TD_UNEQUAL_ROOT),
        "td_fullfreq_precompute": str(TD_FULLFREQ_PRECOMPUTE),
        "frequency_source_default": str(FREQUENCY_SOURCE),
        "model_full": MODEL_FULL,
        "model_bf": MODEL_BF,
        "fit_names_full": FIT_NAMES_FULL,
        "truth_full": TRUTH_FULL.tolist(),
        "prior_bounds_full": {
            name: [float(lo), float(hi)] for name, (lo, hi) in zip(FIT_NAMES_FULL, PRIOR_BOUNDS_FULL)
        },
        "default_mask": {
            "f_min": DEFAULT_F_MIN,
            "f_max": DEFAULT_F_MAX,
            "notch_half_width": DEFAULT_NOTCH_WIDTH,
            "notch_center_mode": "fixed_n_c_over_4L_equal_arm",
        },
        "default_segments": DEFAULT_SEGMENTS,
        "default_segment_days": DEFAULT_SEGMENT_DAYS,
        "default_fsample": DEFAULT_FSAMPLE,
        "default_orbit_chunks": DEFAULT_ORBIT_CHUNKS,
        "default_days_per_orbit_chunk": DEFAULT_DAYS_PER_ORBIT_CHUNK,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("summary")

    p_build = sub.add_parser("build-fd-equal")
    p_build.add_argument("--realization", type=int, default=0)
    p_build.add_argument("--seed", type=int, default=DEFAULT_DATA_SEED)
    p_build.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS)
    p_build.add_argument("--segment-days", type=float, default=DEFAULT_SEGMENT_DAYS)
    p_build.add_argument("--fsample", type=float, default=DEFAULT_FSAMPLE)
    p_build.add_argument("--f-min", type=float, default=DEFAULT_F_MIN)
    p_build.add_argument("--f-max", type=float, default=DEFAULT_F_MAX)
    p_build.add_argument("--notch-width", type=float, default=DEFAULT_NOTCH_WIDTH)
    p_build.add_argument("--response-seed", type=int, default=DEFAULT_RESPONSE_SEED)
    p_build.add_argument("--nsource", type=int, default=DEFAULT_N_SOURCE)
    p_build.add_argument("--frequency-source", type=Path, default=FREQUENCY_SOURCE)
    p_build.add_argument("--log-omega-ast", type=float, default=DEFAULT_LOG_OMEGA_AST)
    p_build.add_argument("--overwrite", action="store_true")

    def add_td_build_args(p: argparse.ArgumentParser, *, default_orbit_type: str) -> None:
        p.add_argument("--realization", type=int, default=0)
        p.add_argument("--seed", type=int, default=DEFAULT_DATA_SEED)
        p.add_argument("--orbit-chunks", type=int, default=DEFAULT_ORBIT_CHUNKS)
        p.add_argument("--days-per-orbit-chunk", type=int, default=DEFAULT_DAYS_PER_ORBIT_CHUNK)
        p.add_argument("--segment-days", type=float, default=DEFAULT_SEGMENT_DAYS)
        p.add_argument("--fsample", type=float, default=DEFAULT_FSAMPLE)
        p.add_argument("--f-min", type=float, default=DEFAULT_F_MIN)
        p.add_argument("--f-max", type=float, default=DEFAULT_F_MAX)
        p.add_argument("--notch-width", type=float, default=DEFAULT_NOTCH_WIDTH)
        p.add_argument("--response-seed", type=int, default=DEFAULT_RESPONSE_SEED)
        p.add_argument("--nsource", type=int, default=DEFAULT_N_SOURCE)
        p.add_argument("--max-parallel", type=int, default=1)
        p.add_argument("--log-omega-ast", type=float, default=DEFAULT_LOG_OMEGA_AST)
        p.add_argument("--orbit-type", choices=["equal", "unequal"], default=default_orbit_type)
        p.add_argument("--equal-orbit-dir", default=DEFAULT_EQUAL_ORBIT_DIR)
        p.add_argument("--equal-orbit-dir-template", default=DEFAULT_EQUAL_ORBIT_DIR_TEMPLATE)
        p.add_argument("--unequal-orbit-dir-template", default=DEFAULT_UNEQUAL_ORBIT_DIR_TEMPLATE)

    p_td_equal = sub.add_parser("build-td-equal")
    add_td_build_args(p_td_equal, default_orbit_type="equal")

    p_td_unequal = sub.add_parser("build-td-unequal")
    add_td_build_args(p_td_unequal, default_orbit_type="unequal")

    p_validate = sub.add_parser("validate-bundle")
    p_validate.add_argument("--bundle", required=True)
    p_validate.add_argument("--overwrite", action="store_true")

    p_pe = sub.add_parser("run-pe")
    p_pe.add_argument("--bundle", required=True)
    p_pe.add_argument("--stage", choices=["test", "formal", "bf"], default="test")
    p_pe.add_argument("--fit-model", choices=["full8", "dwd_only"], default="full8")
    p_pe.add_argument("--nlive", type=int, default=100)
    p_pe.add_argument("--dlogz", type=float, default=0.5)
    p_pe.add_argument("--walks", type=int, default=32)
    p_pe.add_argument("--overwrite", action="store_true")
    p_pe.add_argument("--resume", action="store_true")
    p_pe.add_argument("--check-only", action="store_true")

    p_inspect = sub.add_parser("inspect-pe")
    p_inspect.add_argument("--result", required=True)
    p_inspect.add_argument("--gate-tolerance", type=float, default=0.03)
    p_inspect.add_argument("--report")

    p_bf = sub.add_parser("combine-bf")
    p_bf.add_argument("--full-result", required=True)
    p_bf.add_argument("--bf-result", required=True)
    p_bf.add_argument("--report")

    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "summary":
        print_summary()
        return 0
    if args.command == "build-fd-equal":
        bundle = build_fd_equal_data_and_bundle(args)
        validate_bundle(bundle, overwrite=True)
        return 0
    if args.command in {"build-td-equal", "build-td-unequal"}:
        bundle = build_td_equal_bundle(args)
        validate_bundle(bundle, overwrite=False)
        return 0
    if args.command == "validate-bundle":
        validate_bundle(Path(args.bundle), overwrite=args.overwrite)
        return 0
    if args.command == "run-pe":
        run_pe(args)
        return 0
    if args.command == "inspect-pe":
        inspect_pe(args)
        return 0
    if args.command == "combine-bf":
        combine_bayes_factor(args)
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
