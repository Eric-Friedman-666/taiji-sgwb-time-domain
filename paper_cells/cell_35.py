from pathlib import Path

import corner
import h5py
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
# display is supplied by the noninteractive reproduction runner
from scipy.stats import gaussian_kde

# Manuscript Section 3.2 mismatch stress-test corner plot:
# TD unequal-arm simulated data analyzed with the equal-arm PE model.
REPO_ROOT = Path.cwd() if "REPO_ROOT" not in globals() else Path(REPO_ROOT)
FIGURE_DIR = REPO_ROOT / "figure" if "FIGURE_DIR" not in globals() else Path(FIGURE_DIR)
fig_mismatch_output = FIGURE_DIR / "8para_corner_mismatch.png"

plt.rcParams.update({
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
})

mismatch_formal_h5 = REPO_ROOT / "data/ch32_astro_formal_20260520/fd_equal/results/mismatch_tdunequaldata_equalarmmodel_r00_seed38992598_seg360_fmin1em04_fmax0p15_w0p01_omastm11p5_bundle_formal_nlive1000_full8.h5"
mismatch_null_h5 = REPO_ROOT / "data/ch32_astro_formal_20260520/fd_equal/results/mismatch_tdunequaldata_equalarmmodel_r00_seed38992598_seg360_fmin1em04_fmax0p15_w0p01_omastm11p5_bundle_bf_nlive1000_dwd_only.h5"

mismatch_param_names = [
    "N_acc", "delta_x", "log_A1", "alpha1", "log_A2", "alpha2", "log_Omega_ast", "epsilon"
]
mismatch_param_labels = [
    r"$N_{\rm acc}$",
    r"$\delta x$",
    r"$\log_{10}A_1$",
    r"$\alpha_1$",
    r"$\log_{10}A_2$",
    r"$\alpha_2$",
    r"$\log_{10}\Omega_{\rm ast}$",
    r"$\varepsilon$",
]
mismatch_title_labels = [
    r"N_{\rm acc}",
    r"\delta x",
    r"\log_{10}A_1",
    r"\alpha_1",
    r"\log_{10}A_2",
    r"\alpha_2",
    r"\log_{10}\Omega_{\rm ast}",
    r"\varepsilon",
]


def mismatch_read_evidence(path):
    with h5py.File(resolve_result_path(path), "r") as h5:
        for key in ("pe/evidence", "evidence", "ln_evidence"):
            if key in h5:
                return float(np.asarray(h5[key]).reshape(()))
    raise KeyError(f"No evidence field found in {path}")


for path in (mismatch_formal_h5, mismatch_null_h5):
    if not path.exists():
        raise FileNotFoundError(path)

with h5py.File(resolve_result_path(mismatch_formal_h5), "r") as h5:
    mismatch_trace = np.asarray(h5["pe/trace"][:, :8], dtype=float)
    mismatch_truth = np.asarray(h5["pe/truth_fit"][:8], dtype=float)
    mismatch_mean = np.asarray(h5["pe/posterior_mean"][:8], dtype=float)
    mismatch_std = np.asarray(h5["pe/posterior_std"][:8], dtype=float)
    mismatch_evidence = mismatch_read_evidence(mismatch_formal_h5)

mismatch_ln_bf = mismatch_evidence - mismatch_read_evidence(mismatch_null_h5)

mismatch_case = {
    "label": "Mismatch PE",
    "trace": mismatch_trace,
    "truth": mismatch_truth,
    "mean": mismatch_mean,
    "std": mismatch_std,
    "evidence": mismatch_evidence,
    "ln_bf": mismatch_ln_bf,
    "color": "#2E6DB4",
}

# Direct style controls matched to Step 15, without the upper-right spectrum inset.
corner_width_inch = 15.8
corner_smooth = 1.0
corner_smooth1d = 1.2
corner_hist_bin_factor = 2
corner_label_fontsize = 15
corner_tick_fontsize = 11
corner_linewidth = 1.25
corner_labelpad = 0.2
corner_bottom_label_offset = 0.043
corner_title_fontsize = 12
corner_title_pad = 7.0
legend_fontsize = 17
output_dpi = 320

mismatch_ranges = []
for dim in range(mismatch_trace.shape[1]):
    values = mismatch_trace[:, dim]
    values = values[np.isfinite(values)]
    lo, hi = np.percentile(values, [0.2, 99.8])
    pad = 0.08 * max(hi - lo, 1.0e-12)
    mismatch_ranges.append((float(lo - pad), float(hi + pad)))

fig_mismatch = corner.corner(
    mismatch_case["trace"],
    labels=mismatch_param_labels,
    color=mismatch_case["color"],
    truths=mismatch_case["truth"],
    truth_color="black",
    levels=(0.393, 0.865),
    smooth=corner_smooth,
    smooth1d=corner_smooth1d,
    hist_bin_factor=corner_hist_bin_factor,
    use_math_text=True,
    plot_datapoints=False,
    plot_density=False,
    fill_contours=False,
    show_titles=False,
    max_n_ticks=3,
    range=mismatch_ranges,
    label_kwargs={"fontsize": corner_label_fontsize},
    hist_kwargs={"linewidth": corner_linewidth},
    contour_kwargs={"linewidths": corner_linewidth},
    labelpad=corner_labelpad,
)


def redraw_mismatch_diagonal_kde(fig, case, truth_values, labels, n_dim):
    axes = np.asarray(fig.axes).reshape((n_dim, n_dim))
    for dim in range(n_dim):
        ax = axes[dim, dim]
        lo, hi = map(float, ax.get_xlim())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        ax.cla()
        grid = np.linspace(lo, hi, 400)
        values = np.asarray(case["trace"][:, dim], dtype=float)
        values = values[np.isfinite(values)]
        ymax = 0.0
        if values.size >= 8 and np.nanstd(values) > 0.0:
            density = gaussian_kde(values)(grid)
            ymax = float(np.nanmax(density))
            ax.plot(grid, density, color=case["color"], linewidth=corner_linewidth)
        ax.axvline(truth_values[dim], color="black", linewidth=1.25)
        ax.set_xlim(lo, hi)
        if ymax > 0.0:
            ax.set_ylim(0.0, ymax * 1.08)
        ax.set_yticks([])
        if dim == n_dim - 1:
            ax.set_xlabel(labels[dim], fontsize=corner_label_fontsize)
        else:
            ax.set_xticklabels([])


def restore_mismatch_axis_labels(fig, labels, n_dim, label_fontsize, bottom_label_offset=0.043, labelpad=0.2):
    axes = np.asarray(fig.axes[: n_dim * n_dim]).reshape((n_dim, n_dim))
    bottom = n_dim - 1
    bottom_y0 = min(axes[bottom, col].get_position().y0 for col in range(n_dim))
    bottom_label_y = max(0.010, bottom_y0 - bottom_label_offset)
    for col, label in enumerate(labels):
        ax = axes[bottom, col]
        ax.set_xlabel("")
        bbox = ax.get_position()
        fig.text(
            0.5 * (bbox.x0 + bbox.x1),
            bottom_label_y,
            label,
            ha="center",
            va="top",
            fontsize=label_fontsize,
        )
    for row in range(1, n_dim):
        ax = axes[row, 0]
        ax.set_ylabel(labels[row], fontsize=label_fontsize, labelpad=labelpad)
        ax.yaxis.label.set_visible(True)


def format_recovered_value(value):
    abs_value = abs(float(value))
    if abs_value >= 100.0:
        return f"{value:.0f}"
    if abs_value >= 10.0:
        return f"{value:.2f}"
    if abs_value >= 1.0:
        return f"{value:.3f}"
    if abs_value >= 1.0e-2:
        return f"{value:.3f}"
    return f"{value:.2e}"


def annotate_recovered_values(fig, title_labels, recovered_values, n_dim, title_fontsize, title_pad):
    axes = np.asarray(fig.axes[: n_dim * n_dim]).reshape((n_dim, n_dim))
    for dim, (label, value) in enumerate(zip(title_labels, recovered_values)):
        axes[dim, dim].set_title(
            rf"${label}={format_recovered_value(value)}$",
            fontsize=title_fontsize,
            pad=title_pad,
        )


redraw_mismatch_diagonal_kde(
    fig_mismatch,
    mismatch_case,
    mismatch_case["truth"],
    mismatch_param_labels,
    len(mismatch_param_labels),
)

fig_mismatch.set_size_inches(corner_width_inch, corner_width_inch * 1.05)
fig_mismatch.subplots_adjust(left=0.055, right=0.985, bottom=0.070, top=0.985, wspace=0.0, hspace=0.0)

for ax in fig_mismatch.axes:
    ax.tick_params(axis="both", which="major", labelsize=corner_tick_fontsize, length=3.0, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2.0, width=0.6)
    ax.xaxis.label.set_fontsize(corner_label_fontsize)
    ax.yaxis.label.set_fontsize(corner_label_fontsize)

restore_mismatch_axis_labels(
    fig_mismatch,
    mismatch_param_labels,
    len(mismatch_param_labels),
    corner_label_fontsize,
    bottom_label_offset=corner_bottom_label_offset,
    labelpad=corner_labelpad,
)
annotate_recovered_values(
    fig_mismatch,
    mismatch_title_labels,
    mismatch_case["mean"],
    len(mismatch_param_labels),
    corner_title_fontsize,
    corner_title_pad,
)


def format_mismatch_corner_ticks(fig, n_dim):
    axes = np.asarray(fig.axes[: n_dim * n_dim]).reshape((n_dim, n_dim))

    def formatter_for(lo, hi):
        span = abs(float(hi) - float(lo))
        if span < 2.0e-3:
            decimals = 4
        elif span < 2.0e-2:
            decimals = 3
        elif span < 2.0e-1:
            decimals = 2
        elif span < 2.0:
            decimals = 1
        else:
            decimals = 0
        return FuncFormatter(lambda value, pos, decimals=decimals: f"{value:.{decimals}f}")

    for row in range(n_dim):
        for col in range(n_dim):
            ax = axes[row, col]
            if row < col:
                ax.set_axis_off()
                continue
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.xaxis.set_major_formatter(formatter_for(*ax.get_xlim()))
            ax.yaxis.set_major_formatter(formatter_for(*ax.get_ylim()))
            ax.xaxis.offsetText.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
            if row == col:
                ax.set_yticks([])
                ax.tick_params(axis="y", left=False, labelleft=False)
            if row < n_dim - 1:
                ax.tick_params(axis="x", labelbottom=False)
            if col > 0:
                ax.tick_params(axis="y", labelleft=False)


format_mismatch_corner_ticks(fig_mismatch, len(mismatch_param_labels))

legend_handles = [
    mlines.Line2D(
        [], [], color=mismatch_case["color"], linewidth=3.0,
        label=r"TD unequal-arm data, equal-arm PE",
    ),
]
fig_mismatch.legend(
    handles=legend_handles,
    loc=(0.42, 0.855),
    frameon=False,
    fontsize=legend_fontsize,
    handlelength=2.8,
)

fig_mismatch_output.parent.mkdir(parents=True, exist_ok=True)
fig_mismatch.savefig(fig_mismatch_output, dpi=output_dpi, bbox_inches="tight", pad_inches=0.05)
display(fig_mismatch)
plt.close(fig_mismatch)

mismatch_summary_df = pd.DataFrame(
    {
        "parameter": mismatch_param_labels,
        "injected": mismatch_case["truth"],
        "posterior mean": mismatch_case["mean"],
        "posterior std": mismatch_case["std"],
    }
)
display(mismatch_summary_df)
print(f"Wrote Section 3.2 mismatch corner figure: {fig_mismatch_output}")
