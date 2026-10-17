"""Read the release's non-executable numerical HDF5 data."""

from pathlib import Path
import json

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT_OVERRIDES = {}


def use_fresh_results(directory):
    """Require a complete fresh campaign; never silently mix it with archives."""
    records = json.loads((ROOT / "reference/inference_manifest.json").read_text())["fits"]
    missing = []
    for record in records:
        path = Path(directory).resolve() / record["id"] / "posterior.h5"
        if not path.is_file():
            missing.append(record["id"])
        RESULT_OVERRIDES[record["posterior"]] = path
    if missing:
        raise FileNotFoundError(f"Fresh campaign is incomplete ({len(missing)} missing fits): {', '.join(missing[:10])}")


def resolve_result_path(path):
    path = Path(path)
    try:
        relative = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
    except ValueError:
        return path
    return RESULT_OVERRIDES.get(relative, path)


def read_tree(group):
    if isinstance(group, h5py.Dataset):
        if h5py.check_string_dtype(group.dtype) is not None:
            return group.asstr()[()]
        return group[()]
    if group.attrs.get("is_none", False):
        return None
    return {name: read_tree(group[name]) for name in group}


def load_plot_inputs():
    with h5py.File(ROOT / "data/plot_inputs.h5", "r") as h5:
        return read_tree(h5)


def load_cases():
    return json.loads((ROOT / "data/cases.json").read_text())


def read_result(relative):
    with h5py.File(resolve_result_path(ROOT / relative), "r") as h5:
        trace = np.asarray(h5["pe/trace"][:], dtype=float)
        truth_key = "pe/truth_fit" if "pe/truth_fit" in h5 else "parameters_true"
        truth = np.asarray(h5[truth_key][:], dtype=float)
        if truth.size != trace.shape[1]:
            raise ValueError(f"Parameter count differs between truth and trace in {relative}")
        result = {
            "truth": truth,
            "mean": np.asarray(h5["pe/posterior_mean"][:], dtype=float),
            "std": np.asarray(h5["pe/posterior_std"][:], dtype=float),
            "evidence": float(h5["pe/evidence"][()]),
            "trace": trace,
        }
    return result


def read_model(relative):
    with h5py.File(ROOT / relative, "r") as h5:
        return (
            np.asarray(h5["frequency"][:], dtype=float),
            np.asarray(h5["tables/response"][:], dtype=np.complex128),
            np.asarray(h5["tables/acc_transfer"][:], dtype=np.complex128),
            np.asarray(h5["tables/oms_transfer"][:], dtype=np.complex128),
            np.asarray(h5["parameters_true"][:], dtype=float),
            int(h5.attrs["segments"]),
        )
