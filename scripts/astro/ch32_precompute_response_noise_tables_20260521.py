#!/usr/bin/env python3
"""Precompute Ch.3.2 full-frequency XYZ response/noise tables.

This CPU-only helper builds segment-indexed full-covariance XYZ response and
noise-transfer tables before any frequency mask is applied.  It is intentionally
thin: equal-arm tables reuse the formal Ch.3.2 equal-arm response/noise helper,
while unequal-arm segment tables reuse the existing daily full-covariance
prototype functions.
"""

from __future__ import annotations

from pathlib import Path
import gc
import importlib.util
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

from scripts.astro.ch32_astro_formal_20260520 import calculate_equal_arm_xyz_tables  # noqa: E402


CAMPAIGN = "ch32_astro_formal_20260520"
RUN_TAG = os.getenv("RUN_TAG", "ch32_response_noise_fullfreq_20260521")
MODE = os.getenv("MODE", "summary").strip().lower()
ORBIT_TYPE = os.getenv("ORBIT_TYPE", "unequal").strip().lower()
SEGMENTS = int(os.getenv("SEGMENTS", "360"))
SEGMENT_DAYS = float(os.getenv("SEGMENT_DAYS", "1"))
FSAMPLE = float(os.getenv("FS", "0.4"))
N_SOURCE = int(os.getenv("N_SOURCE", str(2**10)))
RESPONSE_SEED = int(os.getenv("RESPONSE_SEED", "20260520"))
START_SEG = int(os.getenv("START_SEG", "0"))
END_SEG = int(os.getenv("END_SEG", str(SEGMENTS)))
REGENERATE = os.getenv("REGENERATE", "0").strip().lower() in {"1", "true", "yes", "on"}
DEBUG_NF = int(os.getenv("DEBUG_NF", "0"))

DEFAULT_FREQ_SOURCE = "data/frequency_grid.h5"

FREQUENCY_SOURCE = Path(os.getenv("FREQUENCY_SOURCE", DEFAULT_FREQ_SOURCE))
FREQUENCY_DATASET = os.getenv("FREQUENCY_DATASET", "frequency_full")

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", f"data/{CAMPAIGN}/precomputed_tables/{RUN_TAG}"))
CHUNK_DIR = Path(os.getenv("CHUNK_DIR", str(OUTPUT_DIR / "chunks")))
FINAL_PATH = Path(os.getenv("FINAL_PATH", str(OUTPUT_DIR / f"{RUN_TAG}_equal_unequal_fullfreq_seg{SEGMENTS}.h5")))
EQUAL_PATH = Path(os.getenv("EQUAL_PATH", str(CHUNK_DIR / "equal_static_fullfreq.h5")))
CHUNK_TEMPLATE = os.getenv("CHUNK_TEMPLATE", "{orbit_type}_seg{start:03d}_{end:03d}.h5")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def segment_mid_days(indices0: np.ndarray) -> np.ndarray:
    return (indices0.astype(np.float64) * SEGMENT_DAYS) + 0.5 * SEGMENT_DAYS


def fresh_tmp(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    return tmp


def read_frequency() -> np.ndarray:
    if not FREQUENCY_SOURCE.exists():
        raise FileNotFoundError(f"Frequency source does not exist: {FREQUENCY_SOURCE}")
    with h5py.File(FREQUENCY_SOURCE, "r") as h5:
        if FREQUENCY_DATASET not in h5:
            raise KeyError(f"{FREQUENCY_DATASET!r} not found in {FREQUENCY_SOURCE}")
        frequency = np.asarray(h5[FREQUENCY_DATASET][:], dtype=np.float64)
    if DEBUG_NF > 0:
        frequency = frequency[:DEBUG_NF]
    if frequency.size == 0:
        raise RuntimeError("Frequency grid is empty.")
    return frequency


def load_unequal_base_module():
    os.environ["SEGMENTS"] = str(SEGMENTS)
    os.environ["SEGMENT_DAYS"] = str(SEGMENT_DAYS)
    os.environ["FS"] = str(FSAMPLE)
    os.environ.setdefault("GENERATE_ALL_FREQUENCIES", "1")
    base_script = REPO_ROOT / "scripts/astro/xyz4_direct_fullcov_pe_dailyavg_20260506.py"
    spec = importlib.util.spec_from_file_location("ch32_unequal_fullcov_base_20260521", base_script)
    base = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(base)
    return base


def chunk_path(orbit_type: str, start: int, end: int) -> Path:
    return CHUNK_DIR / CHUNK_TEMPLATE.format(orbit_type=orbit_type, start=start, end=end)


def chunk_glob(orbit_type: str):
    return CHUNK_DIR.glob(f"{orbit_type}_seg*.h5")


def create_matrix_datasets(h5: h5py.File, nseg: int, nfreq: int) -> None:
    chunk_nfreq = min(nfreq, 4096)
    for name in ("response", "acc_transfer", "oms_transfer"):
        h5.create_dataset(
            name,
            shape=(nseg, nfreq, 3, 3),
            dtype=np.complex128,
            chunks=(1, chunk_nfreq, 3, 3),
            compression="gzip",
            compression_opts=2,
        )


def stamp_common_attrs(h5: h5py.File, *, orbit_type: str) -> None:
    h5.attrs["description"] = "Ch.3.2 full-frequency full-covariance XYZ response/noise precompute"
    h5.attrs["campaign"] = CAMPAIGN
    h5.attrs["run_tag"] = RUN_TAG
    h5.attrs["orbit_type"] = orbit_type
    h5.attrs["tdi_generation"] = 2
    h5.attrs["channels"] = "X,Y,Z"
    h5.attrs["segments"] = int(SEGMENTS)
    h5.attrs["segment_days"] = float(SEGMENT_DAYS)
    h5.attrs["frequency_source"] = str(FREQUENCY_SOURCE)
    h5.attrs["frequency_dataset"] = FREQUENCY_DATASET
    h5.attrs["frequency_mask_applied"] = False
    h5.attrs["full_frequency"] = True
    h5.attrs["response_seed"] = int(RESPONSE_SEED)
    h5.attrs["n_source"] = int(N_SOURCE)
    h5.attrs["orientation"] = "left-conjugated covariance: matrix[i,j] = <d_i^* d_j>"
    h5.attrs["basis"] = "full XYZ 3x3 covariance with off-diagonal terms"


def build_equal_static() -> Path:
    if EQUAL_PATH.exists() and not REGENERATE:
        log(f"Equal-arm static table already exists: {EQUAL_PATH}")
        return EQUAL_PATH
    frequency = read_frequency()
    log(f"Building equal-arm static full-frequency table: nfreq={frequency.size}")
    response, acc, oms, response_aet, acc_aet, oms_aet = calculate_equal_arm_xyz_tables(
        frequency,
        response_seed=RESPONSE_SEED,
        nsource=N_SOURCE,
    )
    tmp = fresh_tmp(EQUAL_PATH)
    with h5py.File(tmp, "w") as h5:
        stamp_common_attrs(h5, orbit_type="equal")
        h5.attrs["equal_arm_static_reused_for_all_segments"] = True
        h5.create_dataset("frequency", data=frequency)
        h5.create_dataset("segment_index", data=np.arange(SEGMENTS, dtype=np.int64))
        h5.create_dataset("segment_mid_time_days", data=segment_mid_days(np.arange(SEGMENTS)))
        h5.create_dataset("response_single", data=response, compression="gzip", compression_opts=2)
        h5.create_dataset("acc_transfer_single", data=acc, compression="gzip", compression_opts=2)
        h5.create_dataset("oms_transfer_single", data=oms, compression="gzip", compression_opts=2)
        h5.create_dataset("response_aet_diag", data=response_aet, compression="gzip", compression_opts=2)
        h5.create_dataset("acc_transfer_aet_diag", data=acc_aet, compression="gzip", compression_opts=2)
        h5.create_dataset("oms_transfer_aet_diag", data=oms_aet, compression="gzip", compression_opts=2)
    tmp.replace(EQUAL_PATH)
    log(f"Saved equal-arm static table: {EQUAL_PATH}")
    return EQUAL_PATH


def build_unequal_chunk(start_seg: int, end_seg: int) -> Path:
    start_seg = max(0, start_seg)
    end_seg = min(SEGMENTS, end_seg)
    if start_seg >= end_seg:
        raise ValueError(f"Invalid segment range [{start_seg}, {end_seg})")
    path = chunk_path("unequal", start_seg, end_seg)
    if path.exists() and not REGENERATE:
        log(f"Unequal chunk already exists: {path}")
        return path

    frequency = read_frequency()
    base = load_unequal_base_module()
    indices = np.arange(start_seg, end_seg, dtype=np.int64)
    tmp = fresh_tmp(path)
    with h5py.File(tmp, "w") as h5:
        stamp_common_attrs(h5, orbit_type="unequal")
        h5.attrs["segment_range_start_inclusive"] = int(start_seg)
        h5.attrs["segment_range_end_exclusive"] = int(end_seg)
        h5.attrs["response_day_offset"] = 0.5
        h5.attrs["noise_day_offsets"] = "0.5"
        h5.create_dataset("frequency", data=frequency)
        h5.create_dataset("segment_index", data=indices)
        h5.create_dataset("segment_mid_time_days", data=segment_mid_days(indices))
        create_matrix_datasets(h5, len(indices), frequency.size)
        for out_i, seg0 in enumerate(indices):
            seg_index = int(seg0) + 1
            log(f"PHASE=UNEQUAL_TABLE_SEG seg={seg_index:03d}/{SEGMENTS} nfreq={frequency.size}")
            response = base.calculate_xyz_response_for_segment(
                frequency,
                seg_index,
                nsource=N_SOURCE,
                seed=RESPONSE_SEED,
                day_offset=0.5,
            )
            acc, oms = base.calculate_xyz_noise_transfer_for_segment(
                frequency,
                seg_index,
                day_offsets=(0.5,),
            )
            h5["response"][out_i] = response
            h5["acc_transfer"][out_i] = acc
            h5["oms_transfer"][out_i] = oms
            h5.flush()
            del response, acc, oms
            gc.collect()
    tmp.replace(path)
    log(f"Saved unequal chunk: {path}")
    return path


def copy_equal_group(out: h5py.File, frequency: np.ndarray) -> None:
    if not EQUAL_PATH.exists():
        build_equal_static()
    with h5py.File(EQUAL_PATH, "r") as src:
        if not np.array_equal(src["frequency"][:], frequency):
            raise RuntimeError("Equal-arm static frequency grid does not match final grid.")
        grp = out.create_group("equal")
        stamp_common_attrs(grp, orbit_type="equal")
        grp.attrs["equal_arm_static_reused_for_all_segments"] = True
        grp.create_dataset("frequency", data=frequency)
        grp.create_dataset("segment_index", data=np.arange(SEGMENTS, dtype=np.int64))
        grp.create_dataset("segment_mid_time_days", data=segment_mid_days(np.arange(SEGMENTS)))
        create_matrix_datasets(grp, SEGMENTS, frequency.size)
        response = src["response_single"][:]
        acc = src["acc_transfer_single"][:]
        oms = src["oms_transfer_single"][:]
        for seg0 in range(SEGMENTS):
            grp["response"][seg0] = response
            grp["acc_transfer"][seg0] = acc
            grp["oms_transfer"][seg0] = oms
        for name in ("response_aet_diag", "acc_transfer_aet_diag", "oms_transfer_aet_diag"):
            src.copy(name, grp)


def copy_unequal_group(out: h5py.File, frequency: np.ndarray) -> None:
    grp = out.create_group("unequal")
    stamp_common_attrs(grp, orbit_type="unequal")
    grp.create_dataset("frequency", data=frequency)
    grp.create_dataset("segment_index", data=np.arange(SEGMENTS, dtype=np.int64))
    grp.create_dataset("segment_mid_time_days", data=segment_mid_days(np.arange(SEGMENTS)))
    create_matrix_datasets(grp, SEGMENTS, frequency.size)
    filled = np.zeros(SEGMENTS, dtype=bool)
    for path in sorted(chunk_glob("unequal")):
        with h5py.File(path, "r") as src:
            if not np.array_equal(src["frequency"][:], frequency):
                raise RuntimeError(f"Frequency mismatch in {path}")
            indices = src["segment_index"][:].astype(int)
            for name in ("response", "acc_transfer", "oms_transfer"):
                grp[name][indices] = src[name][:]
            filled[indices] = True
            out.flush()
    if not np.all(filled):
        missing = np.where(~filled)[0].tolist()
        raise RuntimeError(f"Missing unequal table segments: {missing[:20]}{'...' if len(missing) > 20 else ''}")


def combine() -> Path:
    frequency = read_frequency()
    if FINAL_PATH.exists() and not REGENERATE:
        log(f"Final precompute table already exists: {FINAL_PATH}")
        return FINAL_PATH
    tmp = fresh_tmp(FINAL_PATH)
    with h5py.File(tmp, "w") as out:
        stamp_common_attrs(out, orbit_type="equal+unequal")
        out.attrs["description"] = "Ch.3.2 full-frequency equal/unequal full-covariance XYZ response/noise tables"
        out.create_dataset("frequency", data=frequency)
        copy_equal_group(out, frequency)
        copy_unequal_group(out, frequency)
    tmp.replace(FINAL_PATH)
    log(f"Combined final precompute table: {FINAL_PATH}")
    return FINAL_PATH


def summary() -> None:
    frequency = read_frequency()
    chunks = sorted(chunk_glob("unequal"))
    filled = np.zeros(SEGMENTS, dtype=bool)
    for path in chunks:
        try:
            with h5py.File(path, "r") as h5:
                filled[h5["segment_index"][:].astype(int)] = True
        except Exception:
            pass
    print(f"RUN_TAG={RUN_TAG}")
    print(f"frequency_source={FREQUENCY_SOURCE}:{FREQUENCY_DATASET} nfreq={frequency.size}")
    print(f"equal_static_exists={EQUAL_PATH.exists()} path={EQUAL_PATH}")
    print(f"unequal_chunks={len(chunks)} segments_done={int(filled.sum())}/{SEGMENTS}")
    print(f"final_exists={FINAL_PATH.exists()} path={FINAL_PATH}")
    if FINAL_PATH.exists():
        with h5py.File(FINAL_PATH, "r") as h5:
            print(f"final frequency={h5['frequency'].shape}")
            for orbit in ("equal", "unequal"):
                if orbit in h5:
                    print(f"{orbit} response={h5[orbit]['response'].shape}")


def main() -> None:
    if MODE == "summary":
        summary()
    elif MODE == "equal":
        build_equal_static()
    elif MODE == "chunk":
        if ORBIT_TYPE != "unequal":
            raise ValueError("MODE=chunk currently supports ORBIT_TYPE=unequal.")
        build_unequal_chunk(START_SEG, END_SEG)
    elif MODE == "combine":
        combine()
    else:
        raise ValueError("MODE must be summary, equal, chunk, or combine.")


if __name__ == "__main__":
    main()
