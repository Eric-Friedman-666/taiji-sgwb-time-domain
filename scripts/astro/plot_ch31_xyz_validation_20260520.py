#!/usr/bin/env python3
"""Paper-facing Chapter 3.1 XYZ validation plots.

This script consumes the first-day 1000-realization XYZ CSD diagnostic and the
10-day/10-segment four-parameter PE trace.  It writes the three figures needed
for Sec. 3.1:

  1. compressed full-XYZ CSD overlay, simulated mean versus calculated model;
  2. direct ratio, P_sim / P_model;
  3. GetDist corner for the 10-day, ten one-day-segment PE run.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import argparse
import io
from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.lines as mlines


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "py_code").exists() and (parent / "data").exists():
            return parent
    fallback = Path(__file__).resolve().parents[2]
    if (fallback / "py_code").exists():
        return fallback
    raise RuntimeError("Could not locate lisa_td_paper_codex")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ASTRO_SCRIPT_DIR = ROOT / "scripts" / "astro"
if str(ASTRO_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(ASTRO_SCRIPT_DIR))

CHANNELS = ("X", "Y", "Z")
DEFAULT_NOTCH_HALF_WIDTH = 0.010
DEFAULT_NOTCH_DAY_MIDPOINT = 0.5
TRUTH = np.array([3.0, 8.0, -11.5, 2.0 / 3.0], dtype=float)
PARAM_NAMES = ["N_acc_amp", "delta_x_amp", "log_Omega_ast", "alpha_ast"]
PARAM_LABELS = [
    r"N_{\rm acc}/10^{-15}",
    r"\delta x/10^{-12}",
    r"\log_{10}\Omega_{\rm ast}",
    r"\varepsilon",
]


def phase(values: np.ndarray) -> np.ndarray:
    return np.angle(values)


def finite_percentile(values: np.ndarray, q: float) -> float:
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.nanpercentile(vals, q))


def load_firstday(input_h5: Path, *, f_min: float, f_max: float):
    with h5py.File(input_h5, "r") as h5:
        freq_all = h5["frequency"][()]
        mask = h5["mask"][()] if "mask" in h5 else np.ones_like(freq_all, dtype=bool)
        mask = mask & (freq_all >= f_min) & (freq_all <= f_max)
        p_sim_all = h5["td_unequal_triangle_2ndtdi/P_sim"][()]
        attrs = dict(h5.attrs)
    return freq_all[mask], p_sim_all[mask], attrs


def write_model_h5(model_h5: Path, freq: np.ndarray, p_model: np.ndarray, *, n_source: int, seed: int) -> None:
    model_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(model_h5, "w") as h5:
        h5.attrs["description"] = "Calculated first-day unequal-arm second-generation XYZ model PSD for Ch. 3.1 plots."
        h5.attrs["n_source"] = int(n_source)
        h5.attrs["seed"] = int(seed)
        h5.attrs["orbit"] = "unequal-arm, MicroSateOrbitEclipticTCB, pn_order=2"
        h5.attrs["tdi"] = "second-generation Michelson X2/Y2/Z2"
        h5.attrs["covariance_convention"] = "left-conjugated P_star_ij = <d_i^* d_j>"
        h5.create_dataset("frequency", data=freq)
        h5.create_dataset("P_model", data=p_model, compression="gzip", compression_opts=4)


def read_model_h5(model_h5: Path, freq: np.ndarray) -> np.ndarray:
    with h5py.File(model_h5, "r") as h5:
        model_freq = h5["frequency"][()]
        p_model = h5["P_model"][()]
    if model_freq.shape != freq.shape or not np.allclose(model_freq, freq, rtol=0.0, atol=1.0e-15):
        raise RuntimeError(f"Model frequency grid in {model_h5} does not match first-day data.")
    return p_model


def calculate_model(freq: np.ndarray, *, n_source: int, seed: int) -> np.ndarray:
    from firstday_xyz_simulated_csd_20260512 import model_psd_matrix  # noqa: E402

    return model_psd_matrix(freq, "unequal", n_source=n_source, seed=seed)


def notch_intervals(
    *,
    f_min: float,
    f_max: float,
    half_width: float = DEFAULT_NOTCH_HALF_WIDTH,
    day_midpoint: float = DEFAULT_NOTCH_DAY_MIDPOINT,
) -> list[tuple[float, float]]:
    from firstday_xyz_simulated_csd_20260512 import build_orbit_sensitivity  # noqa: E402
    from py_code.triangle_code import MOSA_labels  # noqa: E402

    sens = build_orbit_sensitivity("unequal", day_midpoint)
    intervals: list[tuple[float, float]] = []
    for link in MOSA_labels:
        tau = float(sens.dij[link])
        max_n = int(np.ceil((f_max + half_width) * 4.0 * tau)) + 1
        for n in range(1, max_n + 1):
            center = n / (4.0 * tau)
            if f_min - half_width <= center <= f_max + half_width:
                intervals.append((max(f_min, center - half_width), min(f_max, center + half_width)))

    intervals.sort()
    merged: list[tuple[float, float]] = []
    for start, stop in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, stop))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
    return merged


def shade_notches(ax: plt.Axes, intervals: list[tuple[float, float]]) -> None:
    for start, stop in intervals:
        ax.axvspan(start, stop, color="0.92", alpha=0.32, lw=0, zorder=0)


def plot_overlay(freq: np.ndarray, p_sim: np.ndarray, p_model: np.ndarray, output: Path, attrs: dict) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13.6, 10.0), dpi=210, sharex=True)
    sim_color = "#0072B2"
    model_color = "#111111"
    phase_sim_color = "#D55E00"
    phase_model_color = "#009E73"
    intervals = notch_intervals(f_min=float(freq.min()), f_max=float(freq.max()))

    diag_entries = [(0, 0), (1, 1), (2, 2)]
    offdiag_entries = [(0, 1), (0, 2), (1, 2)]

    for col, (i, j) in enumerate(diag_entries):
        ax = axes[0, col]
        label = f"{CHANNELS[i]}{CHANNELS[j]}"
        shade_notches(ax, intervals)
        ax.loglog(freq, np.abs(p_model[:, i, j]), color=model_color, lw=1.4)
        ax.loglog(freq, np.abs(p_sim[:, i, j]), ".", ms=1.8, alpha=0.55, color=sim_color)
        ax.set_title(label, fontsize=11)
        ax.grid(True, which="both", alpha=0.20)
        ax.set_xlim(float(freq.min()), float(freq.max()))
        if col == 0:
            ax.set_ylabel(r"$|P^\star_{II}(f)|$")

    for col, (i, j) in enumerate(offdiag_entries):
        ax = axes[1, col]
        label = f"{CHANNELS[i]}{CHANNELS[j]}"
        shade_notches(ax, intervals)
        ax.loglog(freq, np.abs(p_model[:, i, j]), color=model_color, lw=1.4)
        ax.loglog(freq, np.abs(p_sim[:, i, j]), ".", ms=1.8, alpha=0.55, color=sim_color)
        ax.set_title(rf"$|{label}|$", fontsize=11)
        ax.grid(True, which="both", alpha=0.20)
        ax.set_xlim(float(freq.min()), float(freq.max()))
        if col == 0:
            ax.set_ylabel(r"$|P^\star_{IJ}(f)|$")

    for col, (i, j) in enumerate(offdiag_entries):
        ax = axes[2, col]
        label = f"{CHANNELS[i]}{CHANNELS[j]}"
        shade_notches(ax, intervals)
        ax.semilogx(freq, phase(p_model[:, i, j]), color=phase_model_color, lw=1.0, alpha=0.95)
        ax.semilogx(freq, phase(p_sim[:, i, j]), ".", ms=1.2, color=phase_sim_color, alpha=0.35)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_yticks([-np.pi, 0.0, np.pi])
        ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
        ax.set_title(rf"$\arg {label}$", fontsize=11)
        ax.grid(True, which="both", alpha=0.20)
        ax.set_xlim(float(freq.min()), float(freq.max()))
        ax.set_xlabel(r"$f\,[{\rm Hz}]$")
        if col == 0:
            ax.set_ylabel(r"$\arg P^\star_{IJ}$")

    handles = [
        mlines.Line2D([], [], color=sim_color, marker=".", linestyle="None", markersize=6, label="TD simulation mean"),
        mlines.Line2D([], [], color=model_color, lw=1.8, label="calculated covariance"),
        mlines.Line2D([], [], color=phase_sim_color, marker=".", linestyle="None", markersize=5, label="simulated phase"),
        mlines.Line2D([], [], color=phase_model_color, lw=1.4, label="model phase"),
        mlines.Line2D([], [], color="0.82", lw=8, alpha=0.32, label=r"$\pm0.010$ Hz notch"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.992), ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0.03, 0.03, 0.985, 0.955])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_ratio(freq: np.ndarray, p_sim: np.ndarray, p_model: np.ndarray, output: Path, attrs: dict) -> None:
    ratio = np.full_like(p_sim, np.nan + 0j, dtype=np.complex128)
    good = np.isfinite(p_sim) & np.isfinite(p_model) & (np.abs(p_model) > 0)
    ratio[good] = p_sim[good] / p_model[good]

    fig, axes = plt.subplots(3, 3, figsize=(13.6, 9.6), dpi=210, sharex=True)
    ratio_color = "#D55E00"
    ref_color = "#009E73"
    intervals = notch_intervals(f_min=float(freq.min()), f_max=float(freq.max()))
    diag_entries = [(0, 0), (1, 1), (2, 2)]
    offdiag_entries = [(0, 1), (0, 2), (1, 2)]

    for col, (i, j) in enumerate(diag_entries):
        ax = axes[0, col]
        label = f"{CHANNELS[i]}{CHANNELS[j]}"
        shade_notches(ax, intervals)
        ax.semilogx(
            freq,
            np.abs(ratio[:, i, j]),
            marker=".",
            linestyle="None",
            ms=2.2,
            alpha=0.70,
            color=ratio_color,
            zorder=2,
        )
        ax.axhline(1.0, color=ref_color, ls="--", lw=1.0, zorder=1)
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0.0, 2.2)
        ax.grid(True, which="both", alpha=0.22)
        ax.set_xlim(float(freq.min()), float(freq.max()))
        if col == 0:
            ax.set_ylabel(r"$|Q^\star_{II,\rm sim}/P^\star_{II,\rm model}|$")

    for col, (i, j) in enumerate(offdiag_entries):
        ax = axes[1, col]
        label = f"{CHANNELS[i]}{CHANNELS[j]}"
        shade_notches(ax, intervals)
        ax.semilogx(
            freq,
            np.abs(ratio[:, i, j]),
            marker=".",
            linestyle="None",
            ms=2.2,
            alpha=0.70,
            color=ratio_color,
            zorder=2,
        )
        ax.axhline(1.0, color=ref_color, ls="--", lw=1.0, zorder=1)
        ax.set_title(rf"$|{label}|$", fontsize=11)
        ax.set_ylim(0.0, 2.2)
        ax.grid(True, which="both", alpha=0.22)
        ax.set_xlim(float(freq.min()), float(freq.max()))
        if col == 0:
            ax.set_ylabel(r"$|Q^\star_{IJ,\rm sim}/P^\star_{IJ,\rm model}|$")

    for col, (i, j) in enumerate(offdiag_entries):
        ax = axes[2, col]
        label = f"{CHANNELS[i]}{CHANNELS[j]}"
        shade_notches(ax, intervals)
        ax.semilogx(
            freq,
            np.angle(ratio[:, i, j]),
            marker=".",
            linestyle="None",
            ms=2.2,
            alpha=0.70,
            color=ratio_color,
            zorder=2,
        )
        ax.axhline(0.0, color=ref_color, ls="--", lw=1.0, zorder=1)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_yticks([-np.pi, 0.0, np.pi])
        ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
        ax.set_title(rf"$\arg({label})$", fontsize=11)
        ax.grid(True, which="both", alpha=0.22)
        ax.set_xlim(float(freq.min()), float(freq.max()))
        ax.set_xlabel(r"$f\,[{\rm Hz}]$")
        if col == 0:
            ax.set_ylabel(r"$\arg(Q^\star_{IJ,\rm sim}/P^\star_{IJ,\rm model})$")

    n_real = int(attrs.get("n_real", 1000))
    handles = [
        mlines.Line2D(
            [],
            [],
            color=ratio_color,
            marker=".",
            linestyle="None",
            markersize=7,
            label=rf"simulation/model ratio, $N_{{\rm real}}={n_real}$",
        ),
        mlines.Line2D([], [], color=ref_color, lw=1.2, ls="--", label="ideal reference"),
        mlines.Line2D([], [], color="0.82", lw=8, alpha=0.32, label=r"$\pm0.010$ Hz notch"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.992), ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0.03, 0.03, 0.985, 0.955])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_corner(pe_h5: Path, output: Path) -> dict:
    from getdist import MCSamples, plots

    with h5py.File(pe_h5, "r") as h5:
        group = h5["fullcov_xyz_pe"]
        trace = np.asarray(group["trace"][:, :4], dtype=float)
        truth = np.asarray(group["parameters_true"][:4], dtype=float)
        frequency_shape = tuple(group["frequency"].shape)
        evidence = float(group["evidence"][()])
    if not np.allclose(truth, TRUTH, rtol=0, atol=1.0e-9):
        raise RuntimeError(f"Unexpected truth vector in {pe_h5}: {truth}")

    with redirect_stdout(io.StringIO()):
        samples = MCSamples(samples=trace, names=PARAM_NAMES, labels=PARAM_LABELS)
    g = plots.get_subplot_plotter(width_inch=9.0)
    g.settings.axes_labelsize = 18
    g.settings.axes_fontsize = 13
    g.settings.legend_fontsize = 15
    g.settings.axis_marker_color = "gray"
    g.triangle_plot(
        [samples],
        filled=True,
        markers=TRUTH,
        colors=["#0072B2"],
        contour_colors=["#0072B2"],
        legend_labels=None,
        legend_loc=None,
    )
    handle = mlines.Line2D([], [], color="#0072B2", lw=6, label=r"10-day, 10-segment full-$XYZ$")
    g.fig.legend(handles=[handle], loc=(0.54, 0.82), fontsize=14, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(g.fig)

    mean = trace.mean(axis=0)
    std = trace.std(axis=0)
    return {
        "pe_h5": str(pe_h5),
        "trace_shape": list(trace.shape),
        "frequency_shape": list(frequency_shape),
        "evidence": evidence,
        "truth": truth.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "relative_uncertainty_percent": (std / np.abs(truth) * 100.0).tolist(),
    }


def write_summary(summary: dict, output: Path) -> None:
    lines = [
        "# Chapter 3.1 validation figure summary",
        f"firstday_h5: {summary['firstday_h5']}",
        f"firstday_attrs: {summary['firstday_attrs']}",
        f"overlay_png: {summary['overlay_png']}",
        f"ratio_png: {summary['ratio_png']}",
        f"corner_png: {summary['corner_png']}",
    ]
    if summary["pe"].get("skipped"):
        lines.append("pe: skipped")
    else:
        lines.extend(
            [
                f"pe_h5: {summary['pe']['pe_h5']}",
                f"trace_shape: {summary['pe']['trace_shape']}",
                f"frequency_shape: {summary['pe']['frequency_shape']}",
                f"truth: {summary['pe']['truth']}",
                f"posterior_mean: {summary['pe']['mean']}",
                f"posterior_std: {summary['pe']['std']}",
                f"relative_uncertainty_percent: {summary['pe']['relative_uncertainty_percent']}",
                f"evidence: {summary['pe']['evidence']}",
            ]
        )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--firstday-h5",
        type=Path,
        default=ROOT / "data/firstday_xyz_csd_20260512/firstday_xyz_csd_two_case_nreal1000_20260512.h5",
    )
    parser.add_argument(
        "--pe-h5",
        type=Path,
        default=ROOT
        / "data/xyz4_10seg_1day_pe_20260506/xyz4_10seg_1day_fmin3e5_fmax015_w001_nlive1000_20260506_r01.h5",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figure/ch31_validation_20260520")
    parser.add_argument(
        "--model-h5",
        type=Path,
        default=ROOT / "data/ch31_validation_20260520/ch31_firstday_xyz_model_psd_20260520.h5",
    )
    parser.add_argument("--f-min", type=float, default=3.0e-5)
    parser.add_argument("--f-max", type=float, default=0.15)
    parser.add_argument("--n-source", type=int, default=256)
    parser.add_argument("--model-seed", type=int, default=20260529)
    parser.add_argument("--write-model-only", action="store_true")
    parser.add_argument("--overlay-only", action="store_true")
    parser.add_argument("--skip-corner", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    freq, p_sim, attrs = load_firstday(args.firstday_h5, f_min=args.f_min, f_max=args.f_max)
    if args.model_h5.exists():
        p_model = read_model_h5(args.model_h5, freq)
    else:
        p_model = calculate_model(freq, n_source=args.n_source, seed=args.model_seed)
        write_model_h5(args.model_h5, freq, p_model, n_source=args.n_source, seed=args.model_seed)
    if args.write_model_only:
        print(f"model_h5={args.model_h5}")
        return

    overlay = args.output_dir / "ch31_fig1_firstday_xyz_csd_overlay_nreal1000_20260520.png"
    ratio = args.output_dir / "ch31_fig2_firstday_xyz_ratio_nreal1000_20260520.png"
    corner = args.output_dir / "ch31_fig3_10day_10segment_xyz_pe_getdist_20260520.png"
    summary_path = args.output_dir / "ch31_validation_summary_20260520.md"

    plot_overlay(freq, p_sim, p_model, overlay, attrs)
    if args.overlay_only:
        print(f"overlay_png={overlay}")
        return

    plot_ratio(freq, p_sim, p_model, ratio, attrs)
    pe_summary = {"skipped": True}
    if not args.skip_corner:
        pe_summary = plot_corner(args.pe_h5, corner)
    summary = {
        "firstday_h5": str(args.firstday_h5),
        "firstday_attrs": attrs,
        "overlay_png": str(overlay),
        "ratio_png": str(ratio),
        "corner_png": str(corner),
        "pe": pe_summary,
    }
    write_summary(summary, summary_path)
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
