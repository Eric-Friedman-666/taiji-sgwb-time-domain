#!/usr/bin/env python3
"""Controller-side utilities for the 2026-05-08 AST4 control diagnostics.

This script is intentionally additive. It does not modify the formal paper
pipeline and it does not overwrite validated May 6-7 products. It builds
self-contained diagnostic bundles for the controlled ladder, validates their
truth log likelihoods, and runs/imports GPU PE results under the dated campaign
root.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
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
    p for p in Path(__file__).resolve().parents
    if (p / "py_code").exists() and (p / "data").exists()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from py_code.model import calculate_response_function, calculate_transfer_function  # noqa: E402
from py_code.parameters import H0, L, c, para_range  # noqa: E402
from py_code.triangle_code import DAY  # noqa: E402


RUN_TAG = "20260508"
CAMPAIGN = "ast4_control_diagnostics_20260508"
DATA_ROOT = REPO_ROOT / "data" / CAMPAIGN
LOG_ROOT = REPO_ROOT / "logs" / CAMPAIGN
FIG_ROOT = REPO_ROOT / "figure" / CAMPAIGN

TRUTH4 = np.array([3.0, 8.0, -11.5, 2.0 / 3.0], dtype=np.float64)
TRUTH_NAMES = ("N_acc_amp", "delta_x_amp", "log_Omega_ast", "alpha_ast")
LADDER_CHOICES = ("noise_only", "signal_only", "signal_fixed_noise", "full4")
LADDER_FIT_NAMES = {
    "noise_only": ("N_acc_amp", "delta_x_amp"),
    "signal_only": ("log_Omega_ast", "alpha_ast"),
    "signal_fixed_noise": ("log_Omega_ast", "alpha_ast"),
    "full4": TRUTH_NAMES,
}
LADDER_DATA_COMPONENTS = {
    "noise_only": {"noise": True, "signal": False},
    "signal_only": {"noise": False, "signal": True},
    "signal_fixed_noise": {"noise": True, "signal": True},
    "full4": {"noise": True, "signal": True},
}
LADDER_LIKELIHOOD_COMPONENTS = {
    "noise_only": {"noise": True, "signal": False},
    "signal_only": {"noise": True, "signal": True},
    "signal_fixed_noise": {"noise": True, "signal": True},
    "full4": {"noise": True, "signal": True},
}
LADDER_SEED_OFFSET = {
    "noise_only": 11000000,
    "signal_only": 12000000,
    "signal_fixed_noise": 12500000,
    "full4": 13000000,
}
FSAMPLE = 0.4
PE_F_MIN = 3.0e-5
PE_F_MAX = 0.15
PE_NOTCH_WIDTH = 0.001
N_SOURCE = 1024
RESPONSE_SEED = 20260508

U_AET_FROM_XYZ = np.array(
    [
        [-1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
        [1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)],
        [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
    ],
    dtype=np.float64,
)
CHANNEL_INDEX = {"A": 0, "E": 1, "T": 2}


@dataclass(frozen=True)
class TreatmentConfig:
    name: str
    segments: int
    segment_days: float
    response_day_offsets: tuple[float, ...]
    noise_day_offsets: tuple[float, ...]
    source_data_template: str
    table_path: str
    mask_path: str
    orbit_dir_template: str


TREATMENTS = {
    "tenseg_1day": TreatmentConfig(
        name="tenseg_1day",
        segments=10,
        segment_days=1.0,
        response_day_offsets=(0.5,),
        noise_day_offsets=(0.5,),
        source_data_template="data/xyz4_10seg_1day_parts_20260506/xyz4_10seg_1day_r{index:02d}_10d_allfreq_20260506.h5",
        table_path="data/ast4_control_diagnostics_20260508/td_tables/td_tenseg_1day_fullfreq_response_noise_20260508.h5",
        mask_path="data/ast4_control_diagnostics_20260508/td_masks/td_tenseg_1day_orbit_n4tau_w001_20260508.h5",
        orbit_dir_template="data/orbit_unequal_dayseg_20260506/day{segment1:02d}",
    ),
    "oneseg_10day_dailyavg": TreatmentConfig(
        name="oneseg_10day_dailyavg",
        segments=1,
        segment_days=10.0,
        response_day_offsets=(0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5),
        noise_day_offsets=(0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5),
        source_data_template="data/xyz4_oneseg_dailyavg_parts_20260506/xyz4_oneseg_dailyavg_r{index:02d}_10d_allfreq_20260506.h5",
        table_path="data/ast4_control_diagnostics_20260508/td_tables/td_oneseg_10day_dailyavg_fullfreq_response_noise_20260508.h5",
        mask_path="data/ast4_control_diagnostics_20260508/td_masks/td_oneseg_10day_dailyavg_orbit_n4tau_w001_20260508.h5",
        orbit_dir_template=str(REPO_ROOT / "data/orbit_unequal_360day/seg{segment1}"),
    ),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def stage_start(message: str) -> float:
    log(message)
    return time.perf_counter()


def stage_end(start: float, message: str) -> None:
    log(f"{message} in {time.perf_counter() - start:.2f} s")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def repo_path(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def fresh_tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp_{os.getpid()}_{int(time.time())}")


def source_data_path(treatment: TreatmentConfig, realization: int) -> Path:
    return REPO_ROOT / treatment.source_data_template.format(index=realization)


def controlled_td_data_path(treatment: TreatmentConfig, ladder: str, realization: int) -> Path:
    return (
        DATA_ROOT
        / "td_data"
        / f"td_{treatment.name}_{ladder}_r{realization:02d}_data_{RUN_TAG}.h5"
    )


def td_table_path(treatment: TreatmentConfig) -> Path:
    return REPO_ROOT / treatment.table_path


def td_mask_path(treatment: TreatmentConfig) -> Path:
    return REPO_ROOT / treatment.mask_path


def bundle_path(method: str, treatment: TreatmentConfig, ladder: str, realization: int) -> Path:
    return (
        DATA_ROOT
        / "bundles"
        / f"{method}_{treatment.name}_{ladder}_r{realization:02d}_bundle_{RUN_TAG}.h5"
    )


def result_path(method: str, treatment: TreatmentConfig, ladder: str, realization: int, nlive: int) -> Path:
    return (
        DATA_ROOT
        / "results"
        / f"{method}_{treatment.name}_{ladder}_r{realization:02d}_nlive{nlive}_{RUN_TAG}.h5"
    )


def copy_h5_tree(src: h5py.File, dst: h5py.File) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for name in src:
        src.copy(name, dst, name=name)


def truth_theta10() -> np.ndarray:
    return np.array([TRUTH4[0], TRUTH4[1], TRUTH4[2], TRUTH4[3], -100.0, 0.0, -100.0, 0.0, -100.0, 0.0], dtype=np.float64)


def theta4_for_ladder(ladder: str, values: dict[str, float]) -> np.ndarray:
    theta = TRUTH4.copy()
    if ladder == "noise_only":
        theta[0] = float(values["N_acc_amp"])
        theta[1] = float(values["delta_x_amp"])
    elif ladder in {"signal_only", "signal_fixed_noise"}:
        theta[2] = float(values["log_Omega_ast"])
        theta[3] = float(values["alpha_ast"])
    elif ladder == "full4":
        theta[:] = [
            float(values["N_acc_amp"]),
            float(values["delta_x_amp"]),
            float(values["log_Omega_ast"]),
            float(values["alpha_ast"]),
        ]
    else:
        raise ValueError(f"Unsupported ladder: {ladder}")
    return theta


def theta10_from_theta4(theta4: np.ndarray) -> np.ndarray:
    return np.array([theta4[0], theta4[1], theta4[2], theta4[3], -100.0, 0.0, -100.0, 0.0, -100.0, 0.0], dtype=np.float64)


def sh_ast(frequency: np.ndarray, log_omega: float, epsilon: float) -> np.ndarray:
    omega = 10.0**log_omega * (frequency / 1.0e-3) ** epsilon
    return 3.0 * H0**2 / (4.0 * np.pi**2) * omega / frequency**3


def rotate_to_aet(matrix: np.ndarray) -> np.ndarray:
    return np.einsum("ai,...ij,bj->...ab", U_AET_FROM_XYZ, matrix, U_AET_FROM_XYZ, optimize=True)


def load_xyz_base(treatment: TreatmentConfig):
    os.environ["SEGMENTS"] = str(treatment.segments)
    os.environ["SEGMENT_DAYS"] = str(treatment.segment_days)
    os.environ["FS"] = str(FSAMPLE)
    os.environ["F_MIN"] = str(PE_F_MIN)
    os.environ["F_MAX"] = str(PE_F_MAX)
    os.environ["NOTCH_WIDTH"] = str(PE_NOTCH_WIDTH)
    base_script = REPO_ROOT / "scripts/astro/xyz4_direct_fullcov_pe_dailyavg_20260506.py"
    module_name = f"xyz4_control_base_{treatment.name}"
    spec = importlib.util.spec_from_file_location(module_name, base_script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_frequency_from_source(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return np.asarray(h5["frequency"][:], dtype=np.float64)


def build_td_table(treatment: TreatmentConfig, source_realization: int, recompute: bool) -> Path:
    table = td_table_path(treatment)
    if table.exists() and not recompute:
        log(f"TD table already exists: {table}")
        return table

    source_path = source_data_path(treatment, source_realization)
    frequency = load_frequency_from_source(source_path)
    base = load_xyz_base(treatment)

    ensure_parent(table)
    tmp = fresh_tmp_path(table)

    start = stage_start(
        f"Computing full-frequency TD table for {treatment.name}: "
        f"segments={treatment.segments}, bins={len(frequency)}, nsource={N_SOURCE}"
    )
    with h5py.File(tmp, "w") as h5:
        h5.attrs["description"] = "AST4 control-diagnostic full-frequency TD XYZ response/noise table"
        h5.attrs["campaign"] = CAMPAIGN
        h5.attrs["segment_treatment"] = treatment.name
        h5.attrs["ladder"] = "full4_source_bundle"
        h5.attrs["tdi_generation"] = 2
        h5.attrs["channels"] = "X,Y,Z"
        h5.attrs["orientation"] = "left-conjugated CSD/model covariance"
        h5.attrs["fsample"] = FSAMPLE
        h5.attrs["segment_days"] = treatment.segment_days
        h5.attrs["segments"] = treatment.segments
        h5.attrs["n_source"] = N_SOURCE
        h5.attrs["response_seed"] = RESPONSE_SEED
        h5.attrs["table_has_no_pe_mask"] = True
        h5.create_dataset("frequency", data=frequency)
        h5.create_dataset("segment_index", data=np.arange(1, treatment.segments + 1, dtype=np.int64))
        h5.create_dataset("response_day_offsets", data=np.asarray(treatment.response_day_offsets, dtype=np.float64))
        h5.create_dataset("noise_day_offsets", data=np.asarray(treatment.noise_day_offsets, dtype=np.float64))
        for name in ("response", "acc_transfer", "oms_transfer"):
            h5.create_dataset(
                name,
                shape=(treatment.segments, len(frequency), 3, 3),
                dtype=np.complex128,
                chunks=(1, min(len(frequency), 4096), 3, 3),
                compression="gzip",
                compression_opts=2,
            )

        for seg_index in range(1, treatment.segments + 1):
            seg_start = stage_start(f"Table segment {seg_index:02d}/{treatment.segments}")
            response = base.calculate_xyz_response_averaged_for_segment(
                frequency,
                seg_index,
                nsource=N_SOURCE,
                seed=RESPONSE_SEED,
                day_offsets=treatment.response_day_offsets,
            )
            acc, oms = base.calculate_xyz_noise_transfer_for_segment(
                frequency,
                seg_index,
                day_offsets=treatment.noise_day_offsets,
            )
            h5["response"][seg_index - 1] = response
            h5["acc_transfer"][seg_index - 1] = acc
            h5["oms_transfer"][seg_index - 1] = oms
            h5.flush()
            stage_end(seg_start, f"Stored table segment {seg_index:02d}")

    tmp.replace(table)
    stage_end(start, f"Saved TD table {table}")
    return table


def build_td_orbit_mask(treatment: TreatmentConfig, source_realization: int, recompute: bool) -> Path:
    mask_path = td_mask_path(treatment)
    if mask_path.exists() and not recompute:
        log(f"TD mask already exists: {mask_path}")
        return mask_path

    base = load_xyz_base(treatment)
    frequency = load_frequency_from_source(source_data_path(treatment, source_realization))
    unique_offsets = tuple(dict.fromkeys(treatment.response_day_offsets + treatment.noise_day_offsets))
    mask = (frequency >= PE_F_MIN) & (frequency <= PE_F_MAX)
    rows = []

    start = stage_start(
        f"Building orbit-derived TD PE mask for {treatment.name}: "
        f"segments={treatment.segments}, bins={len(frequency)}, width={PE_NOTCH_WIDTH}"
    )
    for seg_index in range(1, treatment.segments + 1):
        for day_offset in unique_offsets:
            sens = base.build_orbit_sensitivity(seg_index, day_offset=day_offset)
            for link in base.MOSA_labels:
                tau = float(sens.dij[link])
                max_n = int(math.ceil((PE_F_MAX + PE_NOTCH_WIDTH) * 4.0 * tau)) + 1
                for n in range(1, max_n + 1):
                    center = n / (4.0 * tau)
                    if not (PE_F_MIN - PE_NOTCH_WIDTH <= center <= PE_F_MAX + PE_NOTCH_WIDTH):
                        continue
                    mask &= ~((frequency >= center - PE_NOTCH_WIDTH) & (frequency <= center + PE_NOTCH_WIDTH))
                    rows.append((seg_index, float(day_offset), link.encode("ascii"), n, center))

    centers_dtype = np.dtype(
        [
            ("segment", np.int32),
            ("day_offset", np.float64),
            ("link", "S3"),
            ("n", np.int16),
            ("center_hz", np.float64),
        ]
    )
    centers = np.asarray(rows, dtype=centers_dtype)
    ensure_parent(mask_path)
    tmp = fresh_tmp_path(mask_path)
    with h5py.File(tmp, "w") as h5:
        h5.attrs["description"] = "AST4 control-diagnostic TD PE mask with orbit-derived n/(4 tau_ij) notches"
        h5.attrs["campaign"] = CAMPAIGN
        h5.attrs["segment_treatment"] = treatment.name
        h5.attrs["mode"] = "orbit_n_over_4tau"
        h5.attrs["pe_f_min"] = PE_F_MIN
        h5.attrs["pe_f_max"] = PE_F_MAX
        h5.attrs["pe_notch_width"] = PE_NOTCH_WIDTH
        h5.attrs["kept_bins"] = int(np.sum(mask))
        h5.attrs["total_bins"] = int(mask.size)
        h5.create_dataset("frequency", data=frequency)
        h5.create_dataset("mask", data=mask)
        h5.create_dataset("centers", data=centers)
        h5.create_dataset("day_offsets", data=np.asarray(unique_offsets, dtype=np.float64))
    tmp.replace(mask_path)
    stage_end(start, f"Saved TD mask {mask_path}")
    return mask_path


def orbit_dir_for_treatment(treatment: TreatmentConfig, segment: int) -> str:
    return treatment.orbit_dir_template.format(
        treatment=treatment.name,
        segment=segment,
        segment1=segment + 1,
    )


def generate_td_controlled_data(treatment: TreatmentConfig, ladder: str, realization: int, overwrite: bool) -> Path:
    if ladder not in {"noise_only", "signal_only"}:
        raise ValueError("Controlled TD generation is only needed for noise_only or signal_only ladders.")
    out = controlled_td_data_path(treatment, ladder, realization)
    if out.exists() and not overwrite:
        log(f"Controlled TD data already exists: {out}")
        return out

    from py_code.lisa_generate import generate_TDI_data

    base = load_xyz_base(treatment)
    include_noise = LADDER_DATA_COMPONENTS[ladder]["noise"]
    include_signal = LADDER_DATA_COMPONENTS[ladder]["signal"]
    signal_type = "noise+ast" if include_signal else None
    seed_base = 20260508 + LADDER_SEED_OFFSET[ladder] + 100000 * realization

    ensure_parent(out)
    tmp = fresh_tmp_path(out)

    start = stage_start(
        f"Generating controlled TD data {ladder}/{treatment.name}/r{realization:02d}: "
        f"segments={treatment.segments}, segment_days={treatment.segment_days}, "
        f"noise={include_noise}, signal={include_signal}"
    )
    frequency_ref = None
    obs_segments = []
    seeds = []
    for segment in range(treatment.segments):
        seed = seed_base + segment
        orbit_dir = orbit_dir_for_treatment(treatment, segment)
        seg_start = stage_start(
            f"TD controlled segment {segment + 1:02d}/{treatment.segments}: seed={seed}, orbit={orbit_dir}"
        )
        np.random.seed(seed)
        tdi = generate_TDI_data(
            noise=include_noise,
            signal_type=signal_type,
            day_number=treatment.segment_days,
            fsample=FSAMPLE,
            orbit_type="unequal",
            orbit_dir=orbit_dir,
            parameters=TRUTH4,
        )
        try:
            frequency, matrix = base.calculate_xyz_csd_single_segment(
                tdi.measurements["X2"],
                tdi.measurements["Y2"],
                tdi.measurements["Z2"],
                FSAMPLE,
                window_type="hann",
            )
        finally:
            del tdi
        positive = frequency > 0.0
        frequency_pos = np.asarray(frequency[positive], dtype=np.float64)
        matrix_pos = np.asarray(matrix[positive], dtype=np.complex128)
        if frequency_ref is None:
            frequency_ref = frequency_pos
        elif not np.array_equal(frequency_ref, frequency_pos):
            raise RuntimeError("Controlled TD frequency grid changed across segments.")
        obs_segments.append(matrix_pos)
        seeds.append(seed)
        stage_end(seg_start, f"Stored controlled segment {segment + 1:02d}")

    obs = np.asarray(obs_segments, dtype=np.complex128)
    with h5py.File(tmp, "w") as h5:
        h5.attrs["description"] = "AST4 controlled diagnostic TD XYZ data"
        h5.attrs["campaign"] = CAMPAIGN
        h5.attrs["segment_treatment"] = treatment.name
        h5.attrs["ladder"] = ladder
        h5.attrs["channels"] = "X,Y,Z"
        h5.attrs["orientation"] = "left-conjugated CSD/model covariance"
        h5.attrs["model"] = "controlled_noise_only" if ladder == "noise_only" else "controlled_signal_only"
        h5.attrs["data_noise_component"] = include_noise
        h5.attrs["data_signal_component"] = include_signal
        h5.attrs["realization_index"] = realization
        h5.attrs["tdi_generation"] = 2
        h5.attrs["fsample"] = FSAMPLE
        h5.attrs["segment_days"] = treatment.segment_days
        h5.attrs["segments"] = treatment.segments
        h5.attrs["seed_base"] = seed_base
        h5.create_dataset("frequency", data=frequency_ref)
        h5.create_dataset("obs_csd_star", data=obs, compression="gzip", compression_opts=2)
        h5.create_dataset("segment_complete", data=np.ones(treatment.segments, dtype=bool))
        h5.create_dataset("segment_seed", data=np.asarray(seeds, dtype=np.int64))
        h5.create_dataset(
            "orbit_dir",
            data=np.asarray([orbit_dir_for_treatment(treatment, i).encode("utf-8") for i in range(treatment.segments)]),
        )
        params = h5.create_group("parameters")
        params.create_dataset("parameters_true", data=TRUTH4)
        params.create_dataset("segments", data=treatment.segments)
        params.create_dataset("segments_duration_days", data=treatment.segment_days)
        params.create_dataset("fsample", data=FSAMPLE)
        params.create_dataset("observation_days", data=treatment.segments * treatment.segment_days)
    tmp.replace(out)
    stage_end(start, f"Saved controlled TD data {out}")
    return out


def bundle_td_full4(method: str, treatment: TreatmentConfig, realization: int, ladder: str, overwrite: bool) -> Path:
    if ladder not in LADDER_CHOICES:
        raise ValueError(f"Unsupported ladder: {ladder}")
    if method not in {"td_full_xyz", "td_diag_aet", "td_diag_ae"}:
        raise ValueError(f"Unsupported TD method: {method}")

    if ladder in {"full4", "signal_fixed_noise"}:
        src_data = source_data_path(treatment, realization)
        source_case = "existing_validated_full4_td_data"
    else:
        src_data = controlled_td_data_path(treatment, ladder, realization)
        source_case = f"controlled_td_time_domain_{ladder}"
        if not src_data.exists():
            raise FileNotFoundError(
                f"Missing controlled TD data {src_data}. Run generate-td-controlled first."
            )
    src_table = td_table_path(treatment)
    src_mask = td_mask_path(treatment)
    out = bundle_path(method, treatment, ladder, realization)
    if out.exists() and not overwrite:
        log(f"Bundle already exists: {out}")
        return out

    ensure_parent(out)
    tmp = fresh_tmp_path(out)

    with h5py.File(src_data, "r") as data_h5, h5py.File(src_table, "r") as table_h5, h5py.File(src_mask, "r") as mask_h5:
        frequency_full = np.asarray(data_h5["frequency"][:], dtype=np.float64)
        obs_xyz_full = np.asarray(data_h5["obs_csd_star"][:], dtype=np.complex128)
        truth = np.asarray(data_h5["parameters/parameters_true"][:], dtype=np.float64)
        table_frequency = np.asarray(table_h5["frequency"][:], dtype=np.float64)
        mask_frequency = np.asarray(mask_h5["frequency"][:], dtype=np.float64)
        pe_mask = np.asarray(mask_h5["mask"][:], dtype=bool)
        centers = np.asarray(mask_h5["centers"][:])
        if not np.array_equal(frequency_full, table_frequency) or not np.array_equal(frequency_full, mask_frequency):
            raise RuntimeError("TD source data, table, and mask frequency grids do not match.")
        response_xyz_full = np.asarray(table_h5["response"][:], dtype=np.complex128)
        acc_xyz_full = np.asarray(table_h5["acc_transfer"][:], dtype=np.complex128)
        oms_xyz_full = np.asarray(table_h5["oms_transfer"][:], dtype=np.complex128)

    frequency = frequency_full[pe_mask]
    response_xyz = response_xyz_full[:, pe_mask]
    acc_xyz = acc_xyz_full[:, pe_mask]
    oms_xyz = oms_xyz_full[:, pe_mask]
    obs_xyz = obs_xyz_full[:, pe_mask]
    fit_channels = "XYZ"

    if method == "td_full_xyz":
        obs_for_fit = obs_xyz
        response_for_fit = response_xyz
        acc_for_fit = acc_xyz
        oms_for_fit = oms_xyz
        data_dataset_name = "obs_csd_star"
    else:
        obs_aet = rotate_to_aet(obs_xyz)
        response_aet = rotate_to_aet(response_xyz)
        acc_aet = rotate_to_aet(acc_xyz)
        oms_aet = rotate_to_aet(oms_xyz)
        if method == "td_diag_aet":
            channel_names = ("A", "E", "T")
        else:
            channel_names = ("A", "E")
        indices = np.array([CHANNEL_INDEX[ch] for ch in channel_names], dtype=int)
        obs_for_fit = np.real(np.diagonal(obs_aet, axis1=-2, axis2=-1)[:, :, indices])
        response_for_fit = np.real(np.diagonal(response_aet, axis1=-2, axis2=-1)[:, :, indices])
        acc_for_fit = np.real(np.diagonal(acc_aet, axis1=-2, axis2=-1)[:, :, indices])
        oms_for_fit = np.real(np.diagonal(oms_aet, axis1=-2, axis2=-1)[:, :, indices])
        fit_channels = "".join(channel_names)
        data_dataset_name = "obs_diag"

    with h5py.File(tmp, "w") as h5:
        h5.attrs["campaign"] = CAMPAIGN
        h5.attrs["run_tag"] = RUN_TAG
        h5.attrs["method"] = method
        h5.attrs["segment_treatment"] = treatment.name
        h5.attrs["ladder"] = ladder
        h5.attrs["source_case"] = source_case
        h5.attrs["data_noise_component"] = bool(LADDER_DATA_COMPONENTS[ladder]["noise"])
        h5.attrs["data_signal_component"] = bool(LADDER_DATA_COMPONENTS[ladder]["signal"])
        h5.attrs["likelihood_noise_component"] = bool(LADDER_LIKELIHOOD_COMPONENTS[ladder]["noise"])
        h5.attrs["likelihood_signal_component"] = bool(LADDER_LIKELIHOOD_COMPONENTS[ladder]["signal"])
        h5.attrs["realization_index"] = realization
        h5.attrs["tdi_generation"] = 2
        h5.attrs["fsample"] = FSAMPLE
        h5.attrs["segment_days"] = treatment.segment_days
        h5.attrs["segments"] = treatment.segments
        h5.attrs["fit_channels"] = fit_channels
        h5.attrs["fit_parameter_names"] = ",".join(LADDER_FIT_NAMES[ladder])
        h5.attrs["pe_f_min"] = PE_F_MIN
        h5.attrs["pe_f_max"] = PE_F_MAX
        h5.attrs["pe_notch_width"] = PE_NOTCH_WIDTH
        h5.attrs["pe_notch_center_mode"] = "orbit_n_over_4tau"
        h5.attrs["source_data_path"] = repo_path(src_data)
        h5.attrs["source_table_path"] = repo_path(src_table)
        h5.attrs["source_mask_path"] = repo_path(src_mask)
        h5.create_dataset("frequency_full", data=frequency_full)
        h5.create_dataset("frequency", data=frequency)
        h5.create_dataset("pe_frequency_mask", data=pe_mask)
        h5.create_dataset("parameters_true", data=truth)
        h5.create_dataset("parameter_names", data=np.asarray(TRUTH_NAMES, dtype="S32"))
        h5.create_dataset("segment_index", data=np.arange(1, treatment.segments + 1, dtype=np.int64))
        h5.create_dataset("notch_centers", data=centers)
        h5.create_dataset("response_day_offsets", data=np.asarray(treatment.response_day_offsets, dtype=np.float64))
        h5.create_dataset("noise_day_offsets", data=np.asarray(treatment.noise_day_offsets, dtype=np.float64))
        data_group = h5.create_group("data")
        tables_group = h5.create_group("tables")
        data_group.create_dataset(data_dataset_name, data=obs_for_fit, compression="gzip", compression_opts=2)
        tables_group.create_dataset("response", data=response_for_fit, compression="gzip", compression_opts=2)
        tables_group.create_dataset("acc_transfer", data=acc_for_fit, compression="gzip", compression_opts=2)
        tables_group.create_dataset("oms_transfer", data=oms_for_fit, compression="gzip", compression_opts=2)
    tmp.replace(out)
    log(f"Saved bundle {out}")
    return out


def build_fd_mask(frequency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (frequency >= PE_F_MIN) & (frequency <= PE_F_MAX)
    max_n = int(math.ceil((PE_F_MAX + PE_NOTCH_WIDTH) * 4.0 * L / c)) + 1
    centers = []
    for n in range(1, max_n + 1):
        center = n * c / (4.0 * L)
        if center > PE_F_MAX + PE_NOTCH_WIDTH:
            continue
        mask &= ~((frequency >= center - PE_NOTCH_WIDTH) & (frequency <= center + PE_NOTCH_WIDTH))
        centers.append((n, center))
    return mask, np.asarray(centers, dtype=np.dtype([("n", np.int16), ("center_hz", np.float64)]))


def bundle_fd_equal_full4(treatment: TreatmentConfig, realization: int, ladder: str, overwrite: bool) -> Path:
    if ladder not in LADDER_CHOICES:
        raise ValueError(f"Unsupported ladder: {ladder}")

    out = bundle_path("fd_equal", treatment, ladder, realization)
    if out.exists() and not overwrite:
        log(f"Bundle already exists: {out}")
        return out

    frequency_full = load_frequency_from_source(source_data_path(treatment, realization))
    pe_mask, centers = build_fd_mask(frequency_full)
    frequency = frequency_full[pe_mask]
    factor = treatment.segment_days * DAY * FSAMPLE**2 / 2.0

    start = stage_start(
        f"Computing FD equal-arm bundle inputs for {treatment.name}: "
        f"segments={treatment.segments}, bins={len(frequency)}"
    )
    np.random.seed(RESPONSE_SEED)
    ra, re, rt = calculate_response_function(frequency, 1, orbit_type="equal", generation_index=2, show_progress=True)
    acc_a, oms_a, acc_e, oms_e, acc_t, oms_t = calculate_transfer_function(frequency, 1, orbit_type="equal")
    response = np.stack([ra, re, rt], axis=-1)
    acc = np.stack([acc_a, acc_e, acc_t], axis=-1)
    oms = np.stack([oms_a, oms_e, oms_t], axis=-1)
    signal = sh_ast(frequency, TRUTH4[2], TRUTH4[3])[:, None] if LADDER_DATA_COMPONENTS[ladder]["signal"] else 0.0
    noise = (
        (TRUTH4[0] * 1.0e-15) ** 2 * acc
        + (TRUTH4[1] * 1.0e-12) ** 2 * oms
        if LADDER_DATA_COMPONENTS[ladder]["noise"]
        else 0.0
    )
    psd = signal * response + noise
    if np.any(psd < 0.0) or not np.all(np.isfinite(psd)):
        raise RuntimeError(f"Invalid FD simulation PSD for ladder={ladder}")
    sigma = np.sqrt(factor * psd / 2.0)
    rng = np.random.default_rng(20260508 + LADDER_SEED_OFFSET[ladder] + realization)
    real = rng.normal(0.0, sigma[None, :, :], size=(treatment.segments, len(frequency), 3))
    imag = rng.normal(0.0, sigma[None, :, :], size=(treatment.segments, len(frequency), 3))
    obs = real**2 + imag**2

    ensure_parent(out)
    tmp = fresh_tmp_path(out)
    with h5py.File(tmp, "w") as h5:
        h5.attrs["campaign"] = CAMPAIGN
        h5.attrs["run_tag"] = RUN_TAG
        h5.attrs["method"] = "fd_equal"
        h5.attrs["segment_treatment"] = treatment.name
        h5.attrs["ladder"] = ladder
        h5.attrs["source_case"] = "analytic_equal_arm_fd_reference"
        h5.attrs["data_noise_component"] = bool(LADDER_DATA_COMPONENTS[ladder]["noise"])
        h5.attrs["data_signal_component"] = bool(LADDER_DATA_COMPONENTS[ladder]["signal"])
        h5.attrs["likelihood_noise_component"] = bool(LADDER_LIKELIHOOD_COMPONENTS[ladder]["noise"])
        h5.attrs["likelihood_signal_component"] = bool(LADDER_LIKELIHOOD_COMPONENTS[ladder]["signal"])
        h5.attrs["realization_index"] = realization
        h5.attrs["tdi_generation"] = 2
        h5.attrs["fsample"] = FSAMPLE
        h5.attrs["segment_days"] = treatment.segment_days
        h5.attrs["segments"] = treatment.segments
        h5.attrs["fit_channels"] = "AET"
        h5.attrs["fit_parameter_names"] = ",".join(LADDER_FIT_NAMES[ladder])
        h5.attrs["orbit_type"] = "equal"
        h5.attrs["pe_f_min"] = PE_F_MIN
        h5.attrs["pe_f_max"] = PE_F_MAX
        h5.attrs["pe_notch_width"] = PE_NOTCH_WIDTH
        h5.attrs["pe_notch_center_mode"] = "fixed_n_over_4L_equal_arm"
        h5.create_dataset("frequency_full", data=frequency_full)
        h5.create_dataset("frequency", data=frequency)
        h5.create_dataset("pe_frequency_mask", data=pe_mask)
        h5.create_dataset("parameters_true", data=TRUTH4)
        h5.create_dataset("parameter_names", data=np.asarray(TRUTH_NAMES, dtype="S32"))
        h5.create_dataset("segment_index", data=np.arange(1, treatment.segments + 1, dtype=np.int64))
        h5.create_dataset("notch_centers", data=centers)
        data_group = h5.create_group("data")
        tables_group = h5.create_group("tables")
        data_group.create_dataset("obs_diag", data=obs, compression="gzip", compression_opts=2)
        tables_group.create_dataset("response", data=np.broadcast_to(response[None, :, :], (treatment.segments, len(frequency), 3)))
        tables_group.create_dataset("acc_transfer", data=np.broadcast_to(acc[None, :, :], (treatment.segments, len(frequency), 3)))
        tables_group.create_dataset("oms_transfer", data=np.broadcast_to(oms[None, :, :], (treatment.segments, len(frequency), 3)))
    tmp.replace(out)
    stage_end(start, f"Saved FD bundle {out}")
    return out


def validate_bundle(path: Path, overwrite: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r+") as h5:
        if "validation" in h5 and not overwrite:
            log(f"Validation already exists in {path}")
            return

        method = str(h5.attrs["method"])
        ladder = str(h5.attrs["ladder"])
        likelihood_signal = bool(h5.attrs.get("likelihood_signal_component", ladder != "noise_only"))
        likelihood_noise = bool(h5.attrs.get("likelihood_noise_component", True))
        frequency = np.asarray(h5["frequency"][:], dtype=np.float64)
        truth = np.asarray(h5["parameters_true"][:], dtype=np.float64)
        factor = float(h5.attrs["segment_days"]) * DAY * FSAMPLE**2 / 2.0
        if method == "td_full_xyz":
            obs = np.asarray(h5["data/obs_csd_star"][:], dtype=np.complex128)
            response = np.asarray(h5["tables/response"][:], dtype=np.complex128)
            acc = np.asarray(h5["tables/acc_transfer"][:], dtype=np.complex128)
            oms = np.asarray(h5["tables/oms_transfer"][:], dtype=np.complex128)
            signal = sh_ast(frequency, truth[2], truth[3])[None, :, None, None] if likelihood_signal else 0.0
            noise = (
                (truth[0] * 1.0e-15) ** 2 * acc + (truth[1] * 1.0e-12) ** 2 * oms
                if likelihood_noise
                else 0.0
            )
            cov = signal * response + noise
            cov_h = 0.5 * (cov + np.conjugate(np.swapaxes(cov, -1, -2)))
            eig = np.linalg.eigvalsh(cov_h)
            min_eig = float(np.min(np.real(eig)))
            hermitian_max = float(np.max(np.abs(cov - np.conjugate(np.swapaxes(cov, -1, -2)))))
            sign, logabsdet = np.linalg.slogdet(cov_h)
            solved = np.linalg.solve(cov_h, obs)
            quad = np.real(np.trace(solved, axis1=-2, axis2=-1))
            log_const = np.log(np.pi**3) + 3.0 * np.log(factor)
            truth_loglike = -float(np.sum(log_const + np.real(logabsdet) + quad / factor))
            ok = bool(np.all(sign > 0.0) and np.isfinite(truth_loglike) and min_eig > 0.0)
        else:
            obs = np.asarray(h5["data/obs_diag"][:], dtype=np.float64)
            response = np.asarray(h5["tables/response"][:], dtype=np.float64)
            acc = np.asarray(h5["tables/acc_transfer"][:], dtype=np.float64)
            oms = np.asarray(h5["tables/oms_transfer"][:], dtype=np.float64)
            signal = sh_ast(frequency, truth[2], truth[3])[None, :, None] if likelihood_signal else 0.0
            noise = (
                (truth[0] * 1.0e-15) ** 2 * acc + (truth[1] * 1.0e-12) ** 2 * oms
                if likelihood_noise
                else 0.0
            )
            psd = signal * response + noise
            min_psd = float(np.min(psd))
            truth_loglike = -float(np.sum(np.log(np.pi * factor) + np.log(psd) + obs / (factor * psd)))
            min_eig = min_psd
            hermitian_max = 0.0
            ok = bool(np.isfinite(truth_loglike) and min_psd > 0.0)

        if "validation" in h5:
            del h5["validation"]
        group = h5.create_group("validation")
        group.create_dataset("theta_truth", data=truth_theta10())
        group.create_dataset("truth_loglike", data=truth_loglike)
        group.create_dataset("min_model_eigenvalue_or_psd", data=min_eig)
        group.create_dataset("max_hermitian_residual", data=hermitian_max)
        group.attrs["ok"] = ok
        group.attrs["validated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        group.attrs["frequency_bins_kept"] = int(frequency.size)
        group.attrs["segments"] = int(h5.attrs["segments"])
    log(f"Validated bundle {path}: ok={ok}, truth_loglike={truth_loglike:.6e}, min_model={min_eig:.6e}")


def result_complete(path: Path) -> bool:
    if not path.exists():
        return False
    with h5py.File(path, "r") as h5:
        return "pe" in h5 and "trace" in h5["pe"] and h5["pe/trace"].shape[0] > 0


def import_bilby_json_result(bundle: Path, nlive: int, json_path: Optional[Path], overwrite: bool) -> Path:
    if not bundle.exists():
        raise FileNotFoundError(bundle)
    with h5py.File(bundle, "r") as src:
        method = str(src.attrs["method"])
        treatment_name = str(src.attrs["segment_treatment"])
        ladder = str(src.attrs["ladder"])
        realization = int(src.attrs["realization_index"])
        truth = np.asarray(src["parameters_true"][:], dtype=np.float64)
    treatment = TREATMENTS[treatment_name]
    out = result_path(method, treatment, ladder, realization, nlive)
    if result_complete(out) and not overwrite:
        log(f"PE result already complete: {out}")
        return out
    if out.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite incomplete existing result without --overwrite: {out}")

    if json_path is None:
        json_path = DATA_ROOT / "bilby" / f"{out.stem}_result.json"
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    with json_path.open("r") as fp:
        result_json = json.load(fp)

    posterior = result_json["posterior"]["content"]
    fit_names = tuple(name for name in LADDER_FIT_NAMES[ladder] if name in posterior)
    if not fit_names:
        fit_names = tuple(result_json.get("search_parameter_keys", []))
    if not fit_names:
        raise RuntimeError(f"No fitted parameter columns found in {json_path}")
    trace = np.column_stack([np.asarray(posterior[name], dtype=np.float64) for name in fit_names])
    truth_fit = np.asarray([truth[TRUTH_NAMES.index(name)] for name in fit_names], dtype=np.float64)
    mean = trace.mean(axis=0)
    std = trace.std(axis=0)
    bias = mean - truth_fit
    rel_unc = std / np.maximum(np.abs(truth_fit), 1.0e-12)

    ensure_parent(out)
    tmp = fresh_tmp_path(out)
    with h5py.File(bundle, "r") as src, h5py.File(tmp, "w") as dst:
        copy_h5_tree(src, dst)
        dst.attrs["pe_source_bundle"] = repo_path(bundle)
        dst.attrs["pe_imported_from_bilby_json"] = repo_path(json_path)
        group = dst.create_group("pe")
        group.create_dataset("trace", data=trace)
        group.create_dataset("fit_parameter_names", data=np.asarray(fit_names, dtype="S32"))
        group.create_dataset("truth_fit", data=truth_fit)
        group.create_dataset("evidence", data=float(result_json["log_evidence"]))
        group.create_dataset("evidence_err", data=float(result_json.get("log_evidence_err", np.nan)))
        group.create_dataset("posterior_mean", data=mean)
        group.create_dataset("posterior_std", data=std)
        group.create_dataset("bias", data=bias)
        group.create_dataset("relative_uncertainty", data=rel_unc)
        group.attrs["sampler"] = str(result_json.get("sampler", "Bilby.dynesty"))
        group.attrs["nlive"] = int(nlive)
        group.attrs["dlogz"] = float(result_json.get("sampler_kwargs", {}).get("dlogz", np.nan))
        group.attrs["walks"] = int(result_json.get("sampler_kwargs", {}).get("walks", -1))
        group.attrs["label"] = str(result_json.get("label", out.stem))
        group.attrs["imported_from_json"] = True
    tmp.replace(out)
    log(f"Imported Bilby JSON into self-contained PE result {out}")
    return out


def run_pe(bundle: Path, nlive: int, dlogz: float, walks: int, overwrite: bool) -> Path:
    if not bundle.exists():
        raise FileNotFoundError(bundle)

    with h5py.File(bundle, "r") as src:
        method = str(src.attrs["method"])
        treatment_name = str(src.attrs["segment_treatment"])
        ladder = str(src.attrs["ladder"])
        realization = int(src.attrs["realization_index"])
        fit_names = LADDER_FIT_NAMES[ladder]
    treatment = TREATMENTS[treatment_name]
    out = result_path(method, treatment, ladder, realization, nlive)
    if result_complete(out) and not overwrite:
        log(f"PE result already complete: {out}")
        return out

    import bilby
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    with h5py.File(bundle, "r") as h5:
        frequency = np.asarray(h5["frequency"][:], dtype=np.float64)
        truth = np.asarray(h5["parameters_true"][:], dtype=np.float64)
        factor = float(h5.attrs["segment_days"]) * DAY * FSAMPLE**2 / 2.0
        likelihood_signal = bool(h5.attrs.get("likelihood_signal_component", ladder != "noise_only"))
        likelihood_noise = bool(h5.attrs.get("likelihood_noise_component", True))

        if method == "td_full_xyz":
            obs = jnp.asarray(np.asarray(h5["data/obs_csd_star"][:], dtype=np.complex128))
            response = jnp.asarray(np.asarray(h5["tables/response"][:], dtype=np.complex128))
            acc = jnp.asarray(np.asarray(h5["tables/acc_transfer"][:], dtype=np.complex128))
            oms = jnp.asarray(np.asarray(h5["tables/oms_transfer"][:], dtype=np.complex128))
            frequency_j = jnp.asarray(frequency, dtype=jnp.float64)
            log_const = np.log(np.pi**3) + 3.0 * np.log(factor)

            @jax.jit
            def loglike(theta):
                if likelihood_signal:
                    omega = 10.0 ** theta[2] * (frequency_j / 1.0e-3) ** theta[3]
                    signal = 3.0 * H0**2 / (4.0 * jnp.pi**2) * omega / frequency_j**3
                    signal_part = signal[None, :, None, None] * response
                else:
                    signal_part = 0.0
                if likelihood_noise:
                    noise_part = (theta[0] * 1.0e-15) ** 2 * acc + (theta[1] * 1.0e-12) ** 2 * oms
                else:
                    noise_part = 0.0
                cov = signal_part + noise_part
                _, logabsdet = jnp.linalg.slogdet(cov)
                solved = jnp.linalg.solve(cov, obs)
                quad = jnp.real(jnp.trace(solved, axis1=-2, axis2=-1))
                value = -jnp.sum(log_const + jnp.real(logabsdet) + quad / factor)
                return jnp.where(jnp.isfinite(value), value, -jnp.inf)

        else:
            obs = jnp.asarray(np.asarray(h5["data/obs_diag"][:], dtype=np.float64))
            response = jnp.asarray(np.asarray(h5["tables/response"][:], dtype=np.float64))
            acc = jnp.asarray(np.asarray(h5["tables/acc_transfer"][:], dtype=np.float64))
            oms = jnp.asarray(np.asarray(h5["tables/oms_transfer"][:], dtype=np.float64))
            frequency_j = jnp.asarray(frequency, dtype=jnp.float64)
            log_prefactor = np.log(np.pi * factor)

            @jax.jit
            def loglike(theta):
                if likelihood_signal:
                    omega = 10.0 ** theta[2] * (frequency_j / 1.0e-3) ** theta[3]
                    signal = 3.0 * H0**2 / (4.0 * jnp.pi**2) * omega / frequency_j**3
                    signal_part = signal[None, :, None] * response
                else:
                    signal_part = 0.0
                if likelihood_noise:
                    noise_part = (theta[0] * 1.0e-15) ** 2 * acc + (theta[1] * 1.0e-12) ** 2 * oms
                else:
                    noise_part = 0.0
                psd = signal_part + noise_part
                invalid = jnp.any(psd <= 0.0) | jnp.any(~jnp.isfinite(psd))
                value = -jnp.sum(log_prefactor + jnp.log(psd) + obs / (factor * psd))
                return jnp.where(invalid | (~jnp.isfinite(value)), -jnp.inf, value)

    start = stage_start(f"Compiling truth loglike for {bundle.name}")
    truth_value = float(loglike(jnp.asarray(theta10_from_theta4(truth), dtype=jnp.float64)))
    stage_end(start, f"Truth loglike={truth_value:.6e}")

    label = out.stem
    priors = bilby.prior.PriorDict()
    for name in fit_names:
        prior_min, prior_max = para_range[name][0], para_range[name][1]
        priors[name] = bilby.prior.Uniform(prior_min, prior_max, name=name)

    class DiagnosticLikelihood(bilby.Likelihood):
        def __init__(self):
            super().__init__(parameters={name: para_range[name][2] for name in fit_names})

        def log_likelihood(self):
            theta4 = theta4_for_ladder(ladder, self.parameters)
            return float(loglike(jnp.asarray(theta10_from_theta4(theta4), dtype=jnp.float64)))

    result = bilby.run_sampler(
        likelihood=DiagnosticLikelihood(),
        priors=priors,
        sampler="dynesty",
        nlive=nlive,
        walks=walks,
        npool=1,
        outdir=str(DATA_ROOT / "bilby"),
        label=label,
        clean=True,
        resume=False,
        dlogz=dlogz,
        update_interval=100,
    )
    trace = np.column_stack([np.asarray(result.posterior[name], dtype=np.float64) for name in fit_names])
    truth_fit = np.asarray([truth[TRUTH_NAMES.index(name)] for name in fit_names], dtype=np.float64)
    mean = trace.mean(axis=0)
    std = trace.std(axis=0)
    bias = mean - truth_fit
    rel_unc = std / np.maximum(np.abs(truth_fit), 1.0e-12)

    ensure_parent(out)
    tmp = fresh_tmp_path(out)
    with h5py.File(bundle, "r") as src, h5py.File(tmp, "w") as dst:
        copy_h5_tree(src, dst)
        dst.attrs["pe_source_bundle"] = repo_path(bundle)
        group = dst.create_group("pe")
        group.create_dataset("trace", data=trace)
        group.create_dataset("fit_parameter_names", data=np.asarray(fit_names, dtype="S32"))
        group.create_dataset("truth_fit", data=truth_fit)
        group.create_dataset("evidence", data=float(result.log_evidence))
        group.create_dataset("posterior_mean", data=mean)
        group.create_dataset("posterior_std", data=std)
        group.create_dataset("bias", data=bias)
        group.create_dataset("relative_uncertainty", data=rel_unc)
        group.attrs["sampler"] = "Bilby.dynesty"
        group.attrs["nlive"] = int(nlive)
        group.attrs["dlogz"] = float(dlogz)
        group.attrs["walks"] = int(walks)
        group.attrs["label"] = label
        group.attrs["truth_loglike"] = truth_value
    tmp.replace(out)
    log(f"Stored PE result {out}")
    return out


def print_json_summary() -> None:
    summary = {
        "campaign": CAMPAIGN,
        "data_root": str(DATA_ROOT),
        "log_root": str(LOG_ROOT),
        "figure_root": str(FIG_ROOT),
        "treatments": {name: treatment.__dict__ for name, treatment in TREATMENTS.items()},
        "truth": dict(zip(TRUTH_NAMES, TRUTH4.tolist())),
        "pe": {"f_min_hz": PE_F_MIN, "f_max_hz": PE_F_MAX, "notch_width_hz": PE_NOTCH_WIDTH},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("summary")

    p_table = sub.add_parser("build-td-table")
    p_table.add_argument("--treatment", choices=sorted(TREATMENTS), required=True)
    p_table.add_argument("--source-realization", type=int, default=0)
    p_table.add_argument("--recompute", action="store_true")

    p_mask = sub.add_parser("build-td-mask")
    p_mask.add_argument("--treatment", choices=sorted(TREATMENTS), required=True)
    p_mask.add_argument("--source-realization", type=int, default=0)
    p_mask.add_argument("--recompute", action="store_true")

    p_gen_td = sub.add_parser("generate-td-controlled")
    p_gen_td.add_argument("--treatment", choices=sorted(TREATMENTS), required=True)
    p_gen_td.add_argument("--ladder", choices=["noise_only", "signal_only"], required=True)
    p_gen_td.add_argument("--realization", type=int, default=0)
    p_gen_td.add_argument("--overwrite", action="store_true")

    p_bundle_td_any = sub.add_parser("bundle-td")
    p_bundle_td_any.add_argument("--treatment", choices=sorted(TREATMENTS), required=True)
    p_bundle_td_any.add_argument("--method", choices=["td_full_xyz", "td_diag_aet", "td_diag_ae"], required=True)
    p_bundle_td_any.add_argument("--realization", type=int, default=0)
    p_bundle_td_any.add_argument("--ladder", choices=LADDER_CHOICES, required=True)
    p_bundle_td_any.add_argument("--overwrite", action="store_true")

    p_bundle_td = sub.add_parser("bundle-td-full4")
    p_bundle_td.add_argument("--treatment", choices=sorted(TREATMENTS), required=True)
    p_bundle_td.add_argument("--method", choices=["td_full_xyz", "td_diag_aet", "td_diag_ae"], required=True)
    p_bundle_td.add_argument("--realization", type=int, default=0)
    p_bundle_td.add_argument("--ladder", choices=LADDER_CHOICES, default="full4")
    p_bundle_td.add_argument("--overwrite", action="store_true")

    p_bundle_fd_any = sub.add_parser("bundle-fd")
    p_bundle_fd_any.add_argument("--treatment", choices=sorted(TREATMENTS), required=True)
    p_bundle_fd_any.add_argument("--realization", type=int, default=0)
    p_bundle_fd_any.add_argument("--ladder", choices=LADDER_CHOICES, required=True)
    p_bundle_fd_any.add_argument("--overwrite", action="store_true")

    p_bundle_fd = sub.add_parser("bundle-fd-full4")
    p_bundle_fd.add_argument("--treatment", choices=sorted(TREATMENTS), required=True)
    p_bundle_fd.add_argument("--realization", type=int, default=0)
    p_bundle_fd.add_argument("--ladder", choices=LADDER_CHOICES, default="full4")
    p_bundle_fd.add_argument("--overwrite", action="store_true")

    p_validate = sub.add_parser("validate-bundle")
    p_validate.add_argument("--bundle", required=True)
    p_validate.add_argument("--overwrite", action="store_true")

    p_import = sub.add_parser("import-bilby-json")
    p_import.add_argument("--bundle", required=True)
    p_import.add_argument("--nlive", type=int, default=100)
    p_import.add_argument("--json-path")
    p_import.add_argument("--overwrite", action="store_true")

    p_pe = sub.add_parser("run-pe")
    p_pe.add_argument("--bundle", required=True)
    p_pe.add_argument("--nlive", type=int, default=100)
    p_pe.add_argument("--dlogz", type=float, default=0.5)
    p_pe.add_argument("--walks", type=int, default=32)
    p_pe.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "summary":
        print_json_summary()
        return 0
    if args.command == "build-td-table":
        build_td_table(TREATMENTS[args.treatment], source_realization=args.source_realization, recompute=args.recompute)
        return 0
    if args.command == "build-td-mask":
        build_td_orbit_mask(TREATMENTS[args.treatment], source_realization=args.source_realization, recompute=args.recompute)
        return 0
    if args.command == "generate-td-controlled":
        generate_td_controlled_data(TREATMENTS[args.treatment], args.ladder, args.realization, args.overwrite)
        return 0
    if args.command in {"bundle-td", "bundle-td-full4"}:
        bundle_td_full4(args.method, TREATMENTS[args.treatment], args.realization, args.ladder, args.overwrite)
        return 0
    if args.command in {"bundle-fd", "bundle-fd-full4"}:
        bundle_fd_equal_full4(TREATMENTS[args.treatment], args.realization, args.ladder, args.overwrite)
        return 0
    if args.command == "validate-bundle":
        validate_bundle(Path(args.bundle), overwrite=args.overwrite)
        return 0
    if args.command == "import-bilby-json":
        json_path = Path(args.json_path) if args.json_path else None
        import_bilby_json_result(Path(args.bundle), nlive=args.nlive, json_path=json_path, overwrite=args.overwrite)
        return 0
    if args.command == "run-pe":
        run_pe(Path(args.bundle), nlive=args.nlive, dlogz=args.dlogz, walks=args.walks, overwrite=args.overwrite)
        return 0
    raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
