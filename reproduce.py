#!/usr/bin/env python3
"""Regenerate every current manuscript figure and numerical table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_data():
    manifest_path = ROOT / "reference/data_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Missing release data. Unpack the paper-data archive at the package root.")
    manifest = json.loads(manifest_path.read_text())
    bad = []
    for name, expected in manifest.items():
        p = ROOT / name
        if not p.is_file() or p.stat().st_size != expected["bytes"] or sha256(p) != expected["sha256"]:
            bad.append(name)
    if bad:
        raise RuntimeError("Missing or changed input files: " + ", ".join(bad[:12]))
    print(f"Input hashes verified: {len(manifest)} files", flush=True)
    return len(manifest)


def write_csv(path, names, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(rows)


def make_tables(output):
    import numpy as np
    from diagnostics import calculate
    from paper_io import load_cases, read_model, read_result
    from py_code.parameters import para_range

    cases = load_cases()
    refs = json.loads((ROOT / "reference/paper_tables.json").read_text())
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    f, response, acc, oms, theta, segments = read_model(cases["astro_benchmark"]["fd_equal"]["posterior"])
    metrics = {}
    diagnostics_rows = []
    for amp in sorted(cases["astro_scan"]["fd_equal"], key=float):
        truth = read_result(cases["astro_scan"]["fd_equal"][amp]["posterior"])["truth"]
        result = calculate(f, response, acc, oms, truth, segments)
        metrics[amp] = result
        diagnostics_rows.append([float(amp), result["relative_uncertainty_percent"][6], result["snr_abs"], result["snr_rel"]])
    write_csv(tables / "astro_diagnostics.csv",
              ["log10_Omega_ast", "fisher_logomega_relative_uncertainty_percent", "snr_abs", "snr_rel"], diagnostics_rows)
    sw_metric = calculate(*read_model(cases["sw_benchmark"]["fd_equal"]["posterior"]))
    (tables / "fisher_snr_details.json").write_text(json.dumps({"astro": metrics, "sw": sw_metric}, indent=2))
    computed = {}

    names = ["N_acc_amp", "delta_x_amp", "log_A1", "alpha1", "log_A2", "alpha2", "log_Omega_ast", "alpha_ast", "log_Omega_0", "log_fp"]
    truth10 = read_result(cases["sw_benchmark"]["fd_equal"]["posterior"])["truth"]
    computed["tab:pepriors"] = [[float(t), float(para_range[name][0]), float(para_range[name][1])] for name, t in zip(names, truth10)]
    four = read_result(cases["four_parameter"][1])
    computed["tab:xyzpreview"] = [
        [t, m, 100.0 * s / abs(t)] for t, m, s in zip(four["truth"], four["mean"], four["std"])
    ]

    def recovery(group, fisher):
        products = [read_result(cases[group][cfg]["posterior"]) for cfg in ("fd_equal", "td_equal", "td_unequal")]
        rows = []
        for i, truth in enumerate(products[0]["truth"]):
            row = [truth, fisher["relative_uncertainty_percent"][i]]
            for product in products:
                row += [product["mean"][i], 100.0 * product["std"][i] / abs(product["truth"][i])]
            rows.append(row)
        return rows

    computed["tab:astro8corner"] = recovery("astro_benchmark", metrics["-11.5"])
    scan = []
    for amp in sorted(cases["astro_scan"]["fd_equal"], key=float, reverse=True):
        row = [float(amp), metrics[amp]["snr_abs"], metrics[amp]["snr_rel"]]
        for cfg in ("fd_equal", "td_equal", "td_unequal"):
            spec = cases["astro_scan"][cfg][amp]
            row.append(read_result(spec["posterior"])["evidence"] - read_result(spec["null"])["evidence"])
        scan.append(row)
    computed["tab:astro8summary"] = scan
    computed["tab:sw10corner"] = recovery("sw_benchmark", sw_metric)

    headers = {
        "tab:pepriors": ["benchmark", "prior_lower", "prior_upper"],
        "tab:xyzpreview": ["injected", "posterior_mean", "relative_uncertainty_percent"],
        "tab:astro8corner": ["injected", "fisher_unc_percent", "fd_mean", "fd_unc_percent", "td_equal_mean", "td_equal_unc_percent", "td_unequal_mean", "td_unequal_unc_percent"],
        "tab:astro8summary": ["log10_Omega_ast", "SNR_abs", "SNR_rel", "fd_lnBF", "td_equal_lnBF", "td_unequal_lnBF"],
        "tab:sw10corner": ["injected", "fisher_unc_percent", "fd_mean", "fd_unc_percent", "td_equal_mean", "td_equal_unc_percent", "td_unequal_mean", "td_unequal_unc_percent"],
    }
    checks = []
    corrections = json.loads((ROOT / "reference/rounding_corrections.json").read_text())
    for number, reference in enumerate(refs, 1):
        label = reference["label"]
        values = computed[label]
        if label == "tab:astro8summary":
            write_csv(tables / f"table_{number}.csv", headers[label], values)
        else:
            write_csv(tables / f"table_{number}.csv", ["parameter"] + headers[label],
                      [[row["parameter_tex"]] + list(map(float, value)) for row, value in zip(reference["rows"], values)])
        if len(values) != len(reference["rows"]):
            raise RuntimeError(f"Table row count differs: {label}")
        for i, (expected, actual) in enumerate(zip(reference["rows"], values)):
            if len(actual) != len(expected["values"]):
                raise RuntimeError(f"Table column count differs: {label} row {i}")
            for j, (want, have, decimals) in enumerate(zip(expected["values"], actual, expected["decimals"])):
                tolerance = 0.5 * 10.0 ** -decimals + 1.0e-11
                correction = next((c for c in corrections if (c["table"], c["row"], c["column"]) == (number, i + 1, j + 1)), None)
                documented = bool(correction and want == correction["published"] and
                                  abs(have - correction["direct_value"]) < 1.0e-7)
                checks.append({"table": number, "label": label, "row": i + 1, "column": j + 1,
                               "expected": want, "computed": float(have), "tolerance": tolerance,
                               "documented_rounding_difference": documented,
                               "matches_corrected_rounding": bool(correction and abs(have - correction["corrected_rounded"]) <= tolerance),
                               "pass": bool(np.isfinite(have) and abs(have - want) <= tolerance)})
    summary_checks = []
    for path in cases["all_result_paths"]:
        result = read_result(path)
        trace = result["trace"]
        mean_error = float(np.max(np.abs(trace.mean(axis=0) - result["mean"])))
        std_error0 = float(np.max(np.abs(trace.std(axis=0, ddof=0) - result["std"])))
        std_error1 = float(np.max(np.abs(trace.std(axis=0, ddof=1) - result["std"])))
        summary_checks.append({"path": path, "samples": int(len(trace)), "dimensions": int(trace.shape[1]),
                               "mean_max_abs_error": mean_error, "std_max_abs_error": min(std_error0, std_error1),
                               "std_ddof": 0 if std_error0 <= std_error1 else 1,
                               "pass": bool(np.isfinite(trace).all() and mean_error < 1.0e-9 and min(std_error0, std_error1) < 1.0e-9)})
    return checks, summary_checks


def make_figures(output, latex):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = output / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)
    namespace = {"__name__": "__paper_reproduction__", "REPO_ROOT": ROOT,
                 "FIGURE_DIR": figure_dir, "OUTPUT_DIR": output, "USE_TEX": latex,
                 "display": lambda *_args, **_kwargs: None}
    from paper_io import resolve_result_path
    namespace["resolve_result_path"] = resolve_result_path
    cells = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 27, 29, 33, 35, 37, 39, 41, 43]
    records = []
    for index in cells:
        path = ROOT / "paper_cells" / f"cell_{index:02d}.py"
        np.random.seed(20260520)
        started = time.monotonic()
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        plt.close("all")
        records.append({"cell": index, "seconds": time.monotonic() - started})
        print(f"Rendered cell {index:02d} ({records[-1]['seconds']:.1f} s)", flush=True)
    expected = json.loads((ROOT / "reference/manuscript_artifacts.json").read_text())["figures"]
    figures = []
    from PIL import Image
    for item in expected:
        path = output / item["files"][0]
        if not path.is_file():
            raise FileNotFoundError(f"Missing regenerated manuscript figure: {path}")
        with Image.open(path) as image:
            image.verify()
        figures.append({"figure": item["number"], "file": item["files"][0], "bytes": path.stat().st_size, "sha256": sha256(path)})
    return figures, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["all", "tables", "figures", "verify-data"], nargs="?", default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--latex", action="store_true", help="Use the paper's LaTeX fonts; requires latex and dvipng")
    parser.add_argument("--results", type=Path, help="Plot a complete fresh 84-fit infer.py campaign instead of the archived posteriors")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    started = time.monotonic()
    count = verify_data()
    if args.results:
        from paper_io import use_fresh_results
        use_fresh_results(args.results.resolve())
    if args.command == "verify-data":
        return 0
    checks, summaries = make_tables(output)
    figures, cells = ([], [])
    if args.command in {"all", "figures"}:
        figures, cells = make_figures(output, args.latex)
    differences = [x for x in checks + summaries if not x["pass"]]
    failures = [x for x in differences if not x.get("documented_rounding_difference", False)]
    status = "NUMERICAL_DIFFERENCES" if failures else ("PASS_WITH_DOCUMENTED_DIFFERENCES" if differences else "PASS")
    report = {
        "status": status,
        "scope": "Archived posterior/evidence replay plus independent retained-bin full-XYZ Fisher/SNR calculation",
        "input_files_verified": count,
        "figures_regenerated": len(figures), "tables_generated": 5,
        "table_numbers_checked": len(checks), "posterior_summaries_checked": len(summaries),
        "original_table_values_matched": sum(x["pass"] for x in checks),
        "documented_rounding_differences": sum(x.get("documented_rounding_difference", False) and not x["pass"] for x in checks),
        "unexpected_differences": len(failures),
        "production_inference_rerun": False, "seconds": time.monotonic() - started,
        "posterior_source": "fresh campaign" if args.results else "archived production chains",
        "table_checks": checks, "posterior_checks": summaries,
        "figures": figures, "cell_timings": cells, "differences": differences,
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2))
    print(f"{report['status']}: {len(figures)} figures, 5 tables, {len(differences)} documented/total numerical differences, {len(failures)} unexpected", flush=True)
    for failure in differences[:12]:
        print(json.dumps(failure), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
