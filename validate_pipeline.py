#!/usr/bin/env python3
"""Bounded end-to-end checks against the original production data.

This checks the complete 360-segment FD observation generator, the likelihood
normalization, analytic Fisher derivatives, and (with --td) two complete
one-day Triangle simulations. It does not rerun 84 production samplers.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def error_metrics(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    return {"relative_frobenius_error": float(np.linalg.norm((a - b).ravel()) / np.linalg.norm(b.ravel())),
            "max_absolute_error": float(np.max(np.abs(a - b))), "entries": int(a.size)}


def fd_check(output):
    from scripts.astro import ch32_astro_formal_20260520 as core
    from infer import make_loglike
    core.FD_DATA_DIR = output / "fd_build/data"
    core.FD_BUNDLE_DIR = output / "fd_build/bundles"
    args = core.parse_args(["build-fd-equal"])
    with h5py.File(ROOT / "data/checks/fd_reference.h5", "r") as src:
        args.seed = int(src.attrs["seed"])
        args.segments = int(src.attrs["segments"])
        args.segment_days = float(src.attrs["segment_days"])
        args.fsample = float(src.attrs["fsample"])
        reference = src["obs_csd_star"][:]
        indices = src["segment_indices"][:]
        reference_loglike = float(src.attrs["truth_loglike"])
    bundle = core.build_fd_equal_data_and_bundle(args)
    with h5py.File(bundle, "r") as src:
        metrics = error_metrics(src["data/obs_csd_star"][indices], reference)
        arrays = {k: src[k][:] for k in ("frequency", "parameters_true", "data/obs_csd_star", "tables/response", "tables/acc_transfer", "tables/oms_transfer")}
        attrs = dict(src.attrs)
    value = float(make_loglike(arrays, attrs)(arrays["parameters_true"]))
    import jax.numpy as jnp
    jax_value = float(make_loglike(arrays, attrs, "jax")(jnp.asarray(arrays["parameters_true"])))
    metrics.update(truth_loglike=value, jax_truth_loglike=jax_value, reference_truth_loglike=reference_loglike,
                   likelihood_absolute_error=abs(value - reference_loglike),
                   jax_absolute_error=abs(jax_value - reference_loglike),
                   generated_segments=args.segments, comparison_segments=indices.tolist(),
                   frequency_bins=len(arrays["frequency"]),
                   scope="Full original 360-segment FD generator; observations checked at segments 0, 1 and 359; full-data likelihood checked")
    metrics["pass"] = bool(metrics["relative_frobenius_error"] < 1.0e-12 and
                           metrics["likelihood_absolute_error"] < 1.0e-5 and metrics["jax_absolute_error"] < 1.0e-5)
    return metrics


def derivative_check():
    from diagnostics import covariance_and_derivatives
    from paper_io import load_cases, read_model
    from scripts.astro import ch32_astro_formal_20260520 as astro
    from scripts.astro import ch33_sw10_fullxyz_20260531 as sw
    cases = load_cases()
    rows = []
    for group, core in (("astro_benchmark", astro), ("sw_benchmark", sw)):
        f, response, acc, oms, truth, _ = read_model(cases[group]["fd_equal"]["posterior"])
        pick = np.linspace(0, len(f) - 1, 31, dtype=int)
        f, response, acc, oms = f[pick], response[pick], acc[pick], oms[pick]
        analytic, derivatives, *_ = covariance_and_derivatives(f, response, acc, oms, truth)
        original = core.model_covariance_xyz_np(f, response, acc, oms, truth)
        covariance_error = error_metrics(analytic, original)
        for index, derivative in enumerate(derivatives):
            step = 1.0e-5 * max(1.0, abs(truth[index]))
            plus, minus = truth.copy(), truth.copy()
            plus[index] += step
            minus[index] -= step
            finite = (core.model_covariance_xyz_np(f, response, acc, oms, plus) -
                      core.model_covariance_xyz_np(f, response, acc, oms, minus)) / (2.0 * step)
            metric = error_metrics(derivative, finite)
            rows.append({"model_dimensions": len(truth), "parameter_index": index,
                         "covariance_relative_error": covariance_error["relative_frobenius_error"],
                         **metric, "pass": bool(metric["relative_frobenius_error"] < 1.0e-5 and
                                                covariance_error["relative_frobenius_error"] < 1.0e-12)})
    return rows


def td_check(label):
    if label == "td_equal_ast":
        from scripts.astro import ch32_astro_formal_20260520 as core
        orbit_type = "equal"
    else:
        from scripts.astro import ch33_sw10_fullxyz_20260531 as core
        orbit_type = "unequal"
    with h5py.File(ROOT / f"data/checks/{label}.h5", "r") as src:
        frequency = src["frequency"][:]
        reference = src["obs_csd_star"][:]
        truth = src["parameters_true"][:]
        seed = int(src.attrs["seed"])
    args = core.parse_args([f"build-td-{orbit_type}", "--seed", str(seed)])
    if len(truth) == 8:
        args.log_omega_ast = float(truth[6])
    else:
        args.log_omega_0 = float(truth[8])
    started = time.monotonic()
    measured = core.generate_td_equal_chunk(args, 0)
    full_frequency, cov = measured[0]
    indices = np.searchsorted(full_frequency, frequency)
    if not np.array_equal(full_frequency[indices], frequency):
        raise RuntimeError("Regenerated TD frequency grid differs")
    metrics = error_metrics(cov[indices], reference)
    metrics.update(seed=seed, orbit_type=orbit_type, segment=0, segment_days=1,
                   frequency_bins=len(frequency), seconds=time.monotonic() - started,
                   scope="Complete first one-day TD signal+noise simulation; every retained bin and all nine XYZ covariance entries",
                   tolerance=1.0e-7, pass_=bool(metrics["relative_frobenius_error"] < 1.0e-7))
    metrics["pass"] = metrics.pop("pass_")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "pipeline_checks")
    parser.add_argument("--td", action="store_true", help="Also run the two full-day TD simulations")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    started = time.monotonic()
    report = {"scope": __doc__, "production_inference_rerun": False}
    report["fd"] = fd_check(args.output)
    print("FD check:", json.dumps(report["fd"]), flush=True)
    report["fisher_derivatives"] = derivative_check()
    print("Fisher derivative checks:", len(report["fisher_derivatives"]), flush=True)
    if args.td:
        for label in ("td_equal_ast", "td_unequal_sw"):
            report[label] = td_check(label)
            print(label, json.dumps(report[label]), flush=True)
    checks = [report["fd"]] + report["fisher_derivatives"] + [report[label] for label in ("td_equal_ast", "td_unequal_sw") if label in report]
    report["status"] = "PASS" if all(row["pass"] for row in checks) else "FAIL"
    report["seconds"] = time.monotonic() - started
    (args.output / "pipeline_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
