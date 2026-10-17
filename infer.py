#!/usr/bin/env python3
"""Rerun any paper fit from the supplied observations, without saved chains.

The complex full-XYZ likelihood, priors and Bilby/dynesty settings match the
production drivers. Segment-independent tables use the exact sufficient
statistic sum(obs); this only changes the order of floating-point summation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import h5py
import numpy as np

from py_code.parameters import H0, para_range

ROOT = Path(__file__).resolve().parent
NAMES = ("N_acc_amp", "delta_x_amp", "log_A1", "alpha1", "log_A2", "alpha2",
         "log_Omega_ast", "alpha_ast", "log_Omega_0", "log_fp")
FOUR_NAMES = ("N_acc_amp", "delta_x_amp", "log_Omega_ast", "alpha_ast")


def manifest():
    path = ROOT / "reference/inference_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Unpack the inference-data archive at the package root first.")
    return json.loads(path.read_text())


def record_for(identifier):
    for record in manifest()["fits"]:
        if identifier in (record["id"], record["posterior"]):
            return record
    raise ValueError(f"Unknown fit {identifier!r}; run infer.py list")


def load_input(record, bundle=None):
    source = Path(bundle).resolve() if bundle else ROOT / manifest()["file"]
    with h5py.File(source, "r") as src:
        group = src if bundle else src[record["group"]]
        arrays = {name: group[name][:] for name in ("frequency", "parameters_true", "data/obs_csd_star",
                  "tables/response", "tables/acc_transfer", "tables/oms_transfer")}
        attrs = dict(group.attrs)
        if "reference/truth_loglike" in group:
            reference = float(group["reference/truth_loglike"][()])
        else:
            reference = record["sampler"].get("truth_loglike")
    if bundle:
        references = json.loads((ROOT / "reference/tested_all_inference_inputs.json").read_text())["checks"]
        reference = next(row["reference_truth_loglike"] for row in references if row["id"] == record["id"])
        if list(arrays["data/obs_csd_star"].shape) != record["observations_shape"]:
            raise ValueError("The bundle's observation shape does not match the selected paper case")
    return arrays, attrs, reference


def stage_truth(truth, fit_names):
    theta = np.asarray(truth, dtype=float).copy()
    if len(theta) >= 8 and "log_Omega_ast" not in fit_names:
        theta[6:8] = (-100.0, 0.0)
    if len(theta) == 10 and "log_Omega_0" not in fit_names:
        theta[8:10] = (-100.0, 0.0)
    return theta


def likelihood_tolerance(record):
    # These two deliberate detector-mismatch fits have condition numbers
    # around 2.5e7. The unmodified production likelihood itself differs across
    # CPU/GPU builds by about 1e-4 on a 3e8 total. Exact input-array equality
    # and the comparison with the original code are in mismatch_precision.json.
    return 1.0e-3 if "mismatch_tdunequaldata_equalarmmodel" in record["posterior"] else 1.0e-5


def make_loglike(arrays, attrs, backend="numpy"):
    if backend == "jax":
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as xp
    else:
        xp = np
    f = xp.asarray(arrays["frequency"], dtype=xp.float64)
    obs = xp.asarray(arrays["data/obs_csd_star"], dtype=xp.complex128)
    response, acc, oms = [xp.asarray(arrays["tables/" + name], dtype=xp.complex128)
                          for name in ("response", "acc_transfer", "oms_transfer")]
    factor = float(attrs.get("likelihood_factor", float(attrs["segment_days"]) * 86400.0 * float(attrs.get("fsample", 0.4))**2 / 2.0))
    nseg = int(obs.shape[0])
    static = response.ndim == 3
    if static:
        obs = xp.sum(obs, axis=0)
    constant = np.log(np.pi**3) + 3.0 * np.log(factor)
    conversion = 3.0 * H0**2 / (4.0 * np.pi**2 * f**3)

    def loglike(theta):
        if len(theta) == 4:
            omega = 10.0**theta[2] * (f / 1.0e-3)**theta[3]
        else:
            x = f / 0.016
            omega = 10.0**theta[2] * x**theta[3] / (1.0 + 10.0**theta[4] * x**theta[5])
            omega = omega + 10.0**theta[6] * (f / 1.0e-3)**theta[7]
            if len(theta) == 10:
                xsw = f / 10.0**theta[9]
                omega = omega + 10.0**theta[8] * xsw**3 * (7.0 / (4.0 + 3.0 * xsw**2))**3.5
        cov = (conversion * omega)[..., None, None] * response + (theta[0] * 1.0e-15)**2 * acc + (theta[1] * 1.0e-12)**2 * oms
        cov = 0.5 * (cov + xp.conjugate(xp.swapaxes(cov, -1, -2)))
        # Scaling preserves the likelihood while avoiding tiny-number LAPACK
        # underflow on some CPU builds (the PSD matrices are about 1e-40).
        scale = xp.max(xp.abs(cov), axis=(-2, -1))
        scaled_cov = cov / scale[..., None, None]
        sign, logdet = xp.linalg.slogdet(scaled_cov)
        logdet = logdet + 3.0 * xp.log(scale)
        solved = xp.linalg.solve(scaled_cov, obs / scale[..., None, None])
        quad = xp.real(xp.trace(solved, axis1=-2, axis2=-1))
        value = -xp.sum((constant + xp.real(logdet)) * (nseg if static else 1) + quad / factor)
        valid = xp.all(xp.real(sign) > 0) & xp.all(xp.isfinite(logdet)) & xp.all(xp.isfinite(quad)) & xp.isfinite(value)
        return xp.where(valid, value, -xp.inf)

    return jax.jit(loglike) if backend == "jax" else loglike


def check_inputs(records, output, bundle=None):
    started = time.monotonic()
    rows = []
    for record in records:
        arrays, attrs, reference = load_input(record, bundle)
        theta = stage_truth(arrays["parameters_true"], record["fit_names"])
        numpy_value = float(make_loglike(arrays, attrs)(theta))
        error = abs(numpy_value - reference) if reference is not None else None
        tolerance = likelihood_tolerance(record)
        row = {"id": record["id"], "posterior": record["posterior"],
               "observations_shape": list(arrays["data/obs_csd_star"].shape),
               "truth_loglike": numpy_value, "reference_truth_loglike": reference,
               "absolute_error": error, "tolerance": tolerance,
               "tolerance_reason": "Documented ill-conditioned mismatch CPU/GPU comparison" if tolerance > 1.0e-5 else "Ordinary float64 summation tolerance",
               "pass": bool(np.isfinite(numpy_value) and (error is None or error < tolerance)),
               "reference_available": reference is not None}
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {"status": "PASS" if all(r["pass"] for r in rows) else "FAIL", "fits_checked": len(rows),
              "reference_available": sum(r["reference_available"] for r in rows),
              "production_inference_rerun": False, "seconds": time.monotonic() - started, "checks": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 2


def run_fit(record, args):
    import bilby
    import jax.numpy as jnp

    arrays, attrs, reference = load_input(record, args.bundle)
    theta0 = stage_truth(arrays["parameters_true"], record["fit_names"])
    all_names = FOUR_NAMES if len(theta0) == 4 else NAMES[:len(theta0)]
    positions = [all_names.index(name) for name in record["fit_names"]]
    loglike = make_loglike(arrays, attrs, "jax")
    value = float(loglike(jnp.asarray(theta0)))
    if not np.isfinite(value) or (reference is not None and abs(value - reference) > likelihood_tolerance(record)):
        raise RuntimeError(f"Truth likelihood differs from production: {value}, reference {reference}")
    settings = record["sampler"]
    nlive = args.nlive or int(settings["nlive"])
    walks = args.walks or int(settings["walks"])
    dlogz = args.dlogz if args.dlogz is not None else float(settings["dlogz"])
    outdir = args.output.resolve() / record["id"]
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "posterior.h5").exists() and not args.resume:
        raise FileExistsError(f"Use a fresh output directory or --resume: {outdir}")
    priors = bilby.prior.PriorDict({name: bilby.prior.Uniform(*para_range[name][:2], name=name)
                                  for name in record["fit_names"]})

    class PaperLikelihood(bilby.Likelihood):
        def __init__(self):
            super().__init__(parameters={name: para_range[name][2] for name in record["fit_names"]})

        def log_likelihood(self):
            theta = theta0.copy()
            for name, index in zip(record["fit_names"], positions):
                theta[index] = self.parameters[name]
            return float(loglike(jnp.asarray(theta)))

    np.random.seed(args.seed)
    bilby.core.utils.random.seed(args.seed)
    started = time.monotonic()
    result = bilby.run_sampler(likelihood=PaperLikelihood(), priors=priors, sampler="dynesty",
                              nlive=nlive, walks=walks, dlogz=dlogz, npool=1, update_interval=100,
                              seed=args.seed,
                              outdir=str(outdir), label=record["id"], resume=args.resume,
                              clean=not args.resume)
    trace = np.column_stack([result.posterior[name].to_numpy(dtype=float) for name in record["fit_names"]])
    with h5py.File(outdir / "posterior.h5", "w") as dst:
        dst.create_dataset("parameters_true", data=arrays["parameters_true"])
        for key, value in attrs.items():
            dst.attrs[key] = value
        source_link = os.path.relpath(args.bundle.resolve() if args.bundle else ROOT / manifest()["file"], outdir)
        source_group = "" if args.bundle else record["group"]
        dst["tables"] = h5py.ExternalLink(source_link, source_group + "/tables")
        dst["frequency"] = h5py.ExternalLink(source_link, source_group + "/frequency")
        pe = dst.create_group("pe")
        for key, data in {"trace": trace, "posterior_mean": trace.mean(axis=0),
                          "posterior_std": trace.std(axis=0), "truth_fit": theta0[positions],
                          "fit_parameter_names": np.asarray(record["fit_names"], dtype="S32"),
                          "evidence": result.log_evidence, "evidence_err": result.log_evidence_err,
                          "truth_loglike": value}.items():
            pe.create_dataset(key, data=data)
        pe.attrs.update(nlive=nlive, dlogz=dlogz, walks=walks, seed=args.seed, seconds=time.monotonic() - started)
        dst.attrs["source_fit"] = record["id"]
        dst.attrs["source_posterior_reference"] = record["posterior"]
    print(f"Stored fresh inference: {outdir / 'posterior.h5'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("list", "verify-data", "check", "run"))
    p.add_argument("--case", help="fit_NNN, or exact posterior path from infer.py list")
    p.add_argument("--all", action="store_true", help="Run the complete production campaign sequentially; potentially very expensive")
    p.add_argument("--bundle", type=Path, help="Use one regenerated production bundle instead of the optional large inference archive; requires --case")
    p.add_argument("--output", type=Path, default=ROOT / "inference_outputs")
    p.add_argument("--nlive", type=int)
    p.add_argument("--walks", type=int)
    p.add_argument("--dlogz", type=float)
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.bundle and (not args.case or args.all or args.command not in ("check", "run")):
        p.error("--bundle requires --case and the check or run command")
    data = manifest()
    if args.command == "list":
        for r in data["fits"]:
            print(r["id"], r["dimensions"], r["sampler"]["nlive"], r["posterior"])
    elif args.command == "verify-data":
        path = ROOT / data["file"]
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        if path.stat().st_size != data["bytes"] or sha.hexdigest() != data["sha256"]:
            raise RuntimeError("Inference input checksum mismatch")
        print(f"PASS: {len(data['fits'])} fits, {path.stat().st_size} bytes")
    elif args.command == "check":
        records = [record_for(args.case)] if args.case else data["fits"]
        return check_inputs(records, args.output / "likelihood_verification.json", args.bundle)
    else:
        if bool(args.case) == bool(args.all):
            p.error("run requires exactly one of --case or --all")
        records = data["fits"] if args.all else [record_for(args.case)]
        for record in records:
            completed = args.output.resolve() / record["id"] / "posterior.h5"
            if args.resume and completed.is_file():
                with h5py.File(completed, "r") as src:
                    if "pe/trace" not in src or not len(src["pe/trace"]):
                        raise RuntimeError(f"Incomplete fresh result: {completed}")
                print(f"Completed result retained: {completed}", flush=True)
                continue
            run_fit(record, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
