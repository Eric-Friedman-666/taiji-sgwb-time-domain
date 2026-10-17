#!/usr/bin/env python3
"""Generate and fit multiple independent XYZ full-covariance realizations.

This is a comparison-only pipeline. It reuses the validated direct XYZ
full-covariance likelihood from ``xyz4_direct_fullcov_pe_debug.py`` but keeps
the data from many independent realizations in one HDF5 container.

Typical use:
  MODE=generate python scripts/astro/xyz4_multirealization_fullcov.py
  MODE=pe PE_REALIZATION_INDEX=0 CUDA_VISIBLE_DEVICES=0 python scripts/astro/xyz4_multirealization_fullcov.py
"""

from __future__ import annotations

from pathlib import Path
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
from tqdm import tqdm


REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents
    if (p / "py_code").exists() and (p / "data").exists()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

# Defaults for the paper's ten independent ten-day validation data sets.
os.environ.setdefault("SEGMENTS", "10")
os.environ.setdefault("SEGMENT_DAYS", "1")
os.environ.setdefault("GENERATE_ALL_FREQUENCIES", "1")

BASE_SCRIPT = Path(__file__).resolve().with_name(
    os.getenv("XYZ4_BASE_SCRIPT", "xyz4_direct_fullcov_pe_dailyavg_20260506.py")
)
spec = importlib.util.spec_from_file_location("xyz4_direct_fullcov_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

from py_code.parameters import H0, input_parameters, para_range  # noqa: E402
from py_code.sample import Likelihood, get_priors  # noqa: E402


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


MODE = os.getenv("MODE", "summary").strip().lower()
MULTI_DATA_PATH = Path(
    os.getenv(
        "MULTI_DATA_PATH",
        "./data/xyz4_multirealization_1yr_allfreq_20260426.h5",
    )
)
RESULT_DIR = Path(os.getenv("RESULT_DIR", "./data/xyz4_multirealization_pe_20260426"))
PART_DIR = Path(os.getenv("PART_DIR", "./data/xyz4_10seg_1day_parts_20260506"))
PART_FILE_TEMPLATE = os.getenv(
    "PART_FILE_TEMPLATE",
    "xyz4_10seg_1day_r{index:02d}_10d_allfreq_20260506.h5",
)
PART_GLOB_PATTERN = os.getenv(
    "PART_GLOB_PATTERN",
    "xyz4_10seg_1day_r*_10d_allfreq_20260506.h5",
)
RESULT_GROUP = os.getenv("RESULT_GROUP", "fullcov_xyz_pe")
LABEL_PREFIX = os.getenv(
    "LABEL_PREFIX",
    "xyz4_multireal_fmin1e4_fmax015_w001_nlive1000_20260426",
)
N_REALIZATIONS = int(os.getenv("N_REALIZATIONS", "10"))
PE_REALIZATION_INDEX = int(os.getenv("PE_REALIZATION_INDEX", "0"))
GEN_REALIZATION_INDEX = int(os.getenv("GEN_REALIZATION_INDEX", str(PE_REALIZATION_INDEX)))
REGENERATE = env_bool("REGENERATE", False)
BASE_SEED = int(os.getenv("MULTI_BASE_SEED", "20260506"))
ORBIT_DIR_TEMPLATE = os.getenv("ORBIT_DIR_TEMPLATE", "")


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def log_step(message: str) -> float:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)
    return time.perf_counter()


def finish_step(start: float, message: str) -> None:
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message} in {format_seconds(time.perf_counter() - start)}",
        flush=True,
    )


def orbit_dir_for_segment(realization: int, segment: int) -> str:
    if ORBIT_DIR_TEMPLATE:
        return ORBIT_DIR_TEMPLATE.format(
            realization=realization,
            realization1=realization + 1,
            segment=segment,
            segment1=segment + 1,
        )
    return str(REPO_ROOT / f"data/orbit_unequal_dayseg_20260506/day{segment + 1:02d}")


def generate_one_segment(realization: int, segment: int):
    from py_code.lisa_generate import generate_TDI_data

    seed = BASE_SEED + 100000 * realization + segment
    np.random.seed(seed)
    orbit_dir = orbit_dir_for_segment(realization, segment)
    tdi = generate_TDI_data(
        noise=True,
        signal_type=base.MODEL,
        day_number=base.SEGMENT_DAYS,
        fsample=base.FSAMPLE,
        orbit_type="unequal",
        orbit_dir=orbit_dir,
        parameters=np.asarray(input_parameters(model=base.MODEL), dtype=np.float64),
    )
    try:
        frequency, matrix = base.calculate_xyz_csd_single_segment(
            tdi.measurements["X2"],
            tdi.measurements["Y2"],
            tdi.measurements["Z2"],
            base.FSAMPLE,
            window_type="hann",
        )
    finally:
        del tdi
    return seed, frequency, matrix


def initialize_multi_h5(frequency: np.ndarray, mask: np.ndarray) -> None:
    MULTI_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    parameters_true = np.asarray(input_parameters(model=base.MODEL), dtype=np.float64)
    frequency_filtered = np.asarray(frequency[mask], dtype=np.float64)
    with h5py.File(MULTI_DATA_PATH, "w") as h5:
        h5.attrs["description"] = "Ten independent unequal-arm 2nd TDI XYZ full-covariance realizations"
        h5.attrs["orientation"] = (
            "left-conjugated CSD: matrix[i,j] = <d_i^* d_j>; "
            "trace(inv(P_star) @ obs_csd_star) equals d^dagger P^{-1} d"
        )
        h5.attrs["channels"] = ",".join(base.CHANNELS)
        h5.attrs["model"] = base.MODEL
        h5.attrs["generation"] = "all positive FFT frequencies; PE cuts are applied later"
        h5.create_dataset("frequency", data=frequency_filtered)
        h5.create_dataset(
            "obs_csd_star",
            shape=(N_REALIZATIONS, base.SEGMENTS, len(frequency_filtered), 3, 3),
            dtype=np.complex128,
            chunks=(1, 1, min(len(frequency_filtered), 4096), 3, 3),
            compression="gzip",
            compression_opts=2,
        )
        h5.create_dataset("realization_complete", data=np.zeros(N_REALIZATIONS, dtype=bool))
        h5.create_dataset("segment_complete", data=np.zeros((N_REALIZATIONS, base.SEGMENTS), dtype=bool))
        h5.create_dataset("segment_seed", data=np.zeros((N_REALIZATIONS, base.SEGMENTS), dtype=np.int64))
        params = h5.create_group("parameters")
        params.create_dataset("parameters_true", data=parameters_true)
        params.create_dataset("segments", data=base.SEGMENTS)
        params.create_dataset("segments_duration_days", data=base.SEGMENT_DAYS)
        params.create_dataset("fsample", data=base.FSAMPLE)
        params.create_dataset("observation_days", data=base.OBSERVATION_DAYS)


def part_path(realization: int) -> Path:
    return PART_DIR / PART_FILE_TEMPLATE.format(index=realization)


def initialize_part_h5(
    path: Path,
    realization: int,
    frequency: np.ndarray,
    *,
    obs: np.ndarray | None = None,
    segment_complete: np.ndarray | None = None,
    segment_seed: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters_true = np.asarray(input_parameters(model=base.MODEL), dtype=np.float64)
    with h5py.File(path, "w") as h5:
        h5.attrs["description"] = "One independent unequal-arm 2nd TDI XYZ full-covariance realization"
        h5.attrs["orientation"] = (
            "left-conjugated CSD: matrix[i,j] = <d_i^* d_j>; "
            "trace(inv(P_star) @ obs_csd_star) equals d^dagger P^{-1} d"
        )
        h5.attrs["channels"] = ",".join(base.CHANNELS)
        h5.attrs["model"] = base.MODEL
        h5.attrs["generation"] = "all positive FFT frequencies; PE cuts are applied later"
        h5.attrs["realization_index"] = realization
        h5.attrs["complete"] = False
        h5.create_dataset("frequency", data=np.asarray(frequency, dtype=np.float64))
        if obs is None:
            obs = np.zeros((base.SEGMENTS, len(frequency), 3, 3), dtype=np.complex128)
        if segment_complete is None:
            segment_complete = np.zeros(base.SEGMENTS, dtype=bool)
        if segment_seed is None:
            segment_seed = np.zeros(base.SEGMENTS, dtype=np.int64)
        h5.create_dataset(
            "obs_csd_star",
            data=np.asarray(obs, dtype=np.complex128),
            chunks=(1, min(len(frequency), 4096), 3, 3),
            compression="gzip",
            compression_opts=2,
        )
        h5.create_dataset("segment_complete", data=np.asarray(segment_complete, dtype=bool))
        h5.create_dataset("segment_seed", data=np.asarray(segment_seed, dtype=np.int64))
        params = h5.create_group("parameters")
        params.create_dataset("parameters_true", data=parameters_true)
        params.create_dataset("segments", data=base.SEGMENTS)
        params.create_dataset("segments_duration_days", data=base.SEGMENT_DAYS)
        params.create_dataset("fsample", data=base.FSAMPLE)
        params.create_dataset("observation_days", data=base.OBSERVATION_DAYS)


def initialize_part_from_master_if_possible(path: Path, realization: int) -> bool:
    if not MULTI_DATA_PATH.exists():
        return False
    with h5py.File(MULTI_DATA_PATH, "r") as master:
        if "frequency" not in master or "obs_csd_star" not in master:
            return False
        frequency = master["frequency"][:]
        obs_all = master["obs_csd_star"]
        if obs_all.ndim != 5 or realization >= obs_all.shape[0]:
            return False
        segment_complete = master["segment_complete"][realization][:]
        segment_seed = master["segment_seed"][realization][:]
        obs = np.zeros((base.SEGMENTS, len(frequency), 3, 3), dtype=np.complex128)
        done = np.where(segment_complete)[0]
        if len(done):
            obs[done] = obs_all[realization, done]
    initialize_part_h5(
        path,
        realization,
        frequency,
        obs=obs,
        segment_complete=segment_complete,
        segment_seed=segment_seed,
    )
    print(
        f"Initialized part file from existing master for realization {realization:02d}; "
        f"copied {int(segment_complete.sum())}/{base.SEGMENTS} segments.",
        flush=True,
    )
    return True


def generate_one_realization_data(realization: int) -> None:
    path = part_path(realization)
    if path.exists() and REGENERATE:
        path.unlink()
    if not path.exists():
        if not initialize_part_from_master_if_possible(path, realization):
            start = log_step(f"Bootstrapping realization {realization:02d} part file")
            seed, frequency, matrix = generate_one_segment(realization, 0)
            mask = base.build_generation_mask(frequency)
            initialize_part_h5(path, realization, frequency[mask])
            with h5py.File(path, "r+") as h5:
                h5["obs_csd_star"][0] = matrix[mask]
                h5["segment_complete"][0] = True
                h5["segment_seed"][0] = seed
                h5.flush()
            finish_step(start, f"Initialized {path}")

    with h5py.File(path, "r+") as h5:
        frequency = h5["frequency"][:]
        obs = h5["obs_csd_star"]
        complete = h5["segment_complete"]
        seeds = h5["segment_seed"]
        if bool(h5.attrs.get("complete", False)) and not REGENERATE:
            print(f"Realization {realization:02d} already complete: {path}", flush=True)
            return
        start_real = log_step(f"Generating realization {realization:02d} into {path}")
        for segment in tqdm(range(base.SEGMENTS), desc=f"realization {realization:02d}"):
            if bool(complete[segment]) and not REGENERATE:
                continue
            seed, seg_frequency, matrix = generate_one_segment(realization, segment)
            seg_mask = base.build_generation_mask(seg_frequency)
            if not np.array_equal(seg_frequency[seg_mask], frequency):
                raise RuntimeError("Frequency grid changed across Triangle segments.")
            obs[segment] = matrix[seg_mask]
            complete[segment] = True
            seeds[segment] = seed
            h5.flush()
        h5.attrs["complete"] = True
        h5.flush()
        finish_step(start_real, f"Finished realization {realization:02d}")


def generate_multi_data() -> None:
    if MULTI_DATA_PATH.exists() and REGENERATE:
        MULTI_DATA_PATH.unlink()

    first_frequency = None
    mask = None
    if not MULTI_DATA_PATH.exists():
        start = log_step("Bootstrapping HDF5 layout from realization 0 segment 0")
        seed, first_frequency, first_matrix = generate_one_segment(0, 0)
        mask = base.build_generation_mask(first_frequency)
        initialize_multi_h5(first_frequency, mask)
        with h5py.File(MULTI_DATA_PATH, "r+") as h5:
            h5["obs_csd_star"][0, 0] = first_matrix[mask]
            h5["segment_complete"][0, 0] = True
            h5["segment_seed"][0, 0] = seed
            h5.flush()
        finish_step(start, f"Initialized {MULTI_DATA_PATH}")

    with h5py.File(MULTI_DATA_PATH, "r+") as h5:
        frequency = h5["frequency"][:]
        complete = h5["segment_complete"]
        obs = h5["obs_csd_star"]
        seeds = h5["segment_seed"]
        realization_complete = h5["realization_complete"]
        for realization in range(N_REALIZATIONS):
            if bool(realization_complete[realization]) and not REGENERATE:
                print(f"Realization {realization:02d} already complete; skipping.", flush=True)
                continue
            start_real = log_step(f"Generating realization {realization:02d}")
            for segment in tqdm(range(base.SEGMENTS), desc=f"realization {realization:02d}"):
                if bool(complete[realization, segment]) and not REGENERATE:
                    continue
                seed, seg_frequency, matrix = generate_one_segment(realization, segment)
                if first_frequency is None:
                    # Rebuild the full-frequency mask from the segment grid, then compare
                    # to the already stored filtered grid.
                    seg_mask = base.build_generation_mask(seg_frequency)
                    if not np.array_equal(seg_frequency[seg_mask], frequency):
                        raise RuntimeError("Frequency grid changed across Triangle segments.")
                else:
                    seg_mask = mask
                obs[realization, segment] = matrix[seg_mask]
                complete[realization, segment] = True
                seeds[realization, segment] = seed
                h5.flush()
            realization_complete[realization] = True
            h5.flush()
            finish_step(start_real, f"Finished realization {realization:02d}")


def load_realization(realization: int):
    with h5py.File(MULTI_DATA_PATH, "r") as h5:
        frequency = np.asarray(h5["frequency"][:], dtype=np.float64)
        obs_dataset = h5["obs_csd_star"]
        if obs_dataset.ndim == 5:
            if not bool(h5["realization_complete"][realization]):
                raise RuntimeError(f"Realization {realization:02d} is not complete yet.")
            obs = np.asarray(obs_dataset[realization], dtype=np.complex128)
        elif obs_dataset.ndim == 4:
            if "segment_complete" in h5 and not bool(np.all(h5["segment_complete"][:])):
                raise RuntimeError(f"Single-realization file {MULTI_DATA_PATH} is not complete yet.")
            if not bool(h5.attrs.get("complete", True)):
                raise RuntimeError(f"Single-realization file {MULTI_DATA_PATH} is not complete yet.")
            obs = np.asarray(obs_dataset, dtype=np.complex128)
        else:
            raise RuntimeError(f"Unexpected obs_csd_star ndim={obs_dataset.ndim}")
        parameters_true = np.asarray(h5["parameters/parameters_true"][:], dtype=np.float64)
    pe_mask = base.build_pe_frequency_mask(frequency)
    frequency = frequency[pe_mask]
    obs = obs[:, pe_mask]
    print(
        "PE frequency bins: "
        f"{len(frequency)}; range=[{frequency[0]:.6e}, {frequency[-1]:.6e}], "
        f"PE_F_RANGE={base.PE_F_RANGE}, PE_NOTCH_WIDTH={base.PE_NOTCH_WIDTH_VALUE}",
        flush=True,
    )
    return frequency, obs, parameters_true


def consolidate_parts() -> None:
    first = part_path(0)
    if not first.exists():
        raise FileNotFoundError(f"Missing first part file: {first}")
    with h5py.File(first, "r") as f0:
        frequency = f0["frequency"][:]
    initialize_multi_h5(frequency, np.ones_like(frequency, dtype=bool))
    with h5py.File(MULTI_DATA_PATH, "r+") as master:
        for realization in range(N_REALIZATIONS):
            path = part_path(realization)
            if not path.exists():
                raise FileNotFoundError(f"Missing part file: {path}")
            with h5py.File(path, "r") as part:
                if not bool(part.attrs.get("complete", False)):
                    raise RuntimeError(f"Part file is not complete: {path}")
                if not np.array_equal(part["frequency"][:], frequency):
                    raise RuntimeError(f"Frequency mismatch in {path}")
                master["obs_csd_star"][realization] = part["obs_csd_star"][:]
                master["segment_complete"][realization] = part["segment_complete"][:]
                master["segment_seed"][realization] = part["segment_seed"][:]
                master["realization_complete"][realization] = True
                master.flush()
            print(f"Consolidated realization {realization:02d}", flush=True)
    print(f"Consolidated all parts into {MULTI_DATA_PATH}", flush=True)


def run_pe_for_realization(realization: int) -> None:
    import bilby
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    print("JAX devices:", jax.devices(), flush=True)

    frequency, obs, parameters_true = load_realization(realization)
    response, acc_transfer, oms_transfer = base.prepare_model_tables(frequency)

    frequency_j = jnp.asarray(frequency, dtype=jnp.float64)
    obs_j = jnp.asarray(obs, dtype=jnp.complex128)
    response_j = jnp.asarray(response, dtype=jnp.complex128)
    acc_j = jnp.asarray(acc_transfer, dtype=jnp.complex128)
    oms_j = jnp.asarray(oms_transfer, dtype=jnp.complex128)

    log_const = np.log((np.pi**3) * (base.T_SEG**3 * base.FSAMPLE**6) / 8.0)
    prefactor = 2.0 / (base.T_SEG * base.FSAMPLE**2)

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
        _, logabsdet = jnp.linalg.slogdet(cov)
        # obs_csd_star and cov use Q_star[i,j]=d_i^* d_j. With the model
        # covariance transposed into the same orientation, trace(P_star^-1 Q_star)
        # is identical to the standard quadratic form d^\dagger P^{-1} d.
        solved = jnp.linalg.solve(cov, obs_j)
        quad = jnp.real(jnp.trace(solved, axis1=-2, axis2=-1))
        return -jnp.sum(log_const + jnp.real(logabsdet) + prefactor * quad)

    theta_true_10 = jnp.asarray(
        [
            parameters_true[0],
            parameters_true[1],
            parameters_true[2],
            parameters_true[3],
            -100.0,
            0.0,
            -100.0,
            0.0,
            -100.0,
            0.0,
        ],
        dtype=jnp.float64,
    )
    start = log_step(f"Compiling/evaluating log-likelihood for realization {realization:02d}")
    injected_loglike = float(loglike(theta_true_10))
    finish_step(start, f"Injected log-likelihood = {injected_loglike:.6e}")

    priors = get_priors(model=base.MODEL)
    likelihood = Likelihood(params=para_range, model=base.MODEL, loglike=loglike)
    label = f"{LABEL_PREFIX}_r{realization:02d}"
    result_path = RESULT_DIR / f"{label}.h5"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    start = log_step(
        f"Running Bilby.dynesty PE for realization {realization:02d}: "
        f"nlive={base.SAMPLE_NUMBER}, dlogz={base.DLOGZ}, walks={base.WALKS}"
    )
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=base.SAMPLE_NUMBER,
        walks=base.WALKS,
        npool=1,
        outdir="./data/bilby",
        label=label,
        clean=True,
        resume=False,
        dlogz=base.DLOGZ,
        update_interval=100,
    )
    finish_step(start, f"Bilby.dynesty PE finished for realization {realization:02d}")

    trace = np.asarray(result.posterior)[:, : len(parameters_true)]
    with h5py.File(result_path, "w") as h5:
        group = h5.create_group(RESULT_GROUP)
        group.create_dataset("trace", data=trace)
        group.create_dataset("evidence", data=result.log_evidence)
        group.create_dataset("frequency", data=frequency)
        group.create_dataset("parameters_true", data=parameters_true)
        group.attrs["sampler"] = "Bilby.dynesty"
        group.attrs["nlive"] = base.SAMPLE_NUMBER
        group.attrs["dlogz"] = base.DLOGZ
        group.attrs["walks"] = base.WALKS
        group.attrs["label"] = label
        group.attrs["realization_index"] = realization
        group.attrs["multi_data_path"] = str(MULTI_DATA_PATH)
        group.attrs["table_path"] = str(base.TABLE_PATH)
        group.attrs["pe_f_min"] = "None" if base.PE_F_RANGE[0] is None else base.PE_F_RANGE[0]
        group.attrs["pe_f_max"] = "None" if base.PE_F_RANGE[1] is None else base.PE_F_RANGE[1]
        group.attrs["pe_notch_width"] = (
            "None" if base.PE_NOTCH_WIDTH_VALUE is None else base.PE_NOTCH_WIDTH_VALUE
        )
        group.attrs["noise_average_day_offsets"] = ",".join(str(x) for x in base.NOISE_AVERAGE_DAY_OFFSETS)
    print(f"Stored PE result in {result_path}: {RESULT_GROUP}", flush=True)


def print_summary() -> None:
    print(f"Repository: {REPO_ROOT}", flush=True)
    print(f"Mode: {MODE}", flush=True)
    print(f"Multi data path: {MULTI_DATA_PATH}", flush=True)
    print(f"Result dir: {RESULT_DIR}", flush=True)
    print(f"Realizations: {N_REALIZATIONS}", flush=True)
    if MULTI_DATA_PATH.exists():
        with h5py.File(MULTI_DATA_PATH, "r") as h5:
            print(f"frequency: {h5['frequency'].shape}", flush=True)
            print(f"obs_csd_star: {h5['obs_csd_star'].shape}", flush=True)
            if "realization_complete" in h5:
                complete = np.where(h5["realization_complete"][:])[0].tolist()
                print(f"complete: {complete}", flush=True)
            else:
                done = int(h5["segment_complete"][:].sum()) if "segment_complete" in h5 else -1
                complete = bool(h5.attrs.get("complete", False))
                print(f"part complete={complete}, segments={done}/{base.SEGMENTS}", flush=True)
    if RESULT_DIR.exists():
        for path in sorted(RESULT_DIR.glob(f"{LABEL_PREFIX}_r*.h5")):
            with h5py.File(path, "r") as h5:
                ok = RESULT_GROUP in h5 and "trace" in h5[RESULT_GROUP]
                shape = h5[RESULT_GROUP]["trace"].shape if ok else None
                print(f"{path.name}: ok={ok}, trace={shape}", flush=True)
    if PART_DIR.exists():
        for path in sorted(PART_DIR.glob(PART_GLOB_PATTERN)):
            with h5py.File(path, "r") as h5:
                done = int(h5["segment_complete"][:].sum()) if "segment_complete" in h5 else -1
                complete = bool(h5.attrs.get("complete", False))
                print(f"{path.name}: complete={complete}, segments={done}/{base.SEGMENTS}", flush=True)


def main() -> None:
    print_summary()
    if MODE == "generate":
        generate_multi_data()
    elif MODE == "generate_one":
        generate_one_realization_data(GEN_REALIZATION_INDEX)
    elif MODE == "pe":
        run_pe_for_realization(PE_REALIZATION_INDEX)
    elif MODE == "consolidate":
        consolidate_parts()
    elif MODE == "summary":
        return
    else:
        raise ValueError("MODE must be one of: generate, pe, summary")


if __name__ == "__main__":
    main()
