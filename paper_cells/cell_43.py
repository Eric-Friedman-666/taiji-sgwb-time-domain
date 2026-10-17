
import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
# display is supplied by the noninteractive reproduction runner
from scipy.stats import gaussian_kde

# Manuscript Fig. 7: ten-parameter SW benchmark corner plot.
# The style follows the Section 3.2 corner plot cell above: direct corner contours,
# black truth crosshairs, and a compact legend carrying the BF values.
fig7_output = SW_FIGURE_DIR / "10para_corner.png"

fig7_sample_defs = [
    {"label": "FD PE", "color": "red", "samples": sw10_products["FD PE"]["samples"], "ln_bf": sw10_products["FD PE"]["ln_bf"]},
    {"label": "TD equal PE", "color": "green", "samples": sw10_products["TD equal PE"]["samples"], "ln_bf": sw10_products["TD equal PE"]["ln_bf"]},
    {"label": "TD unequal PE", "color": "purple", "samples": sw10_products["TD unequal PE"]["samples"], "ln_bf": sw10_products["TD unequal PE"]["ln_bf"]},
]

fig7_param_labels = [
    r"$N_{\rm acc}/10^{-15}$",
    r"$\delta x/10^{-12}$",
    r"$\log_{10}A_1$",
    r"$\alpha_1$",
    r"$\log_{10}A_2$",
    r"$\alpha_2$",
    r"$\log_{10}\Omega_{\rm ast}$",
    r"$\varepsilon$",
    r"$\log_{10}\Omega_0$",
    r"$\log_{10}(f_p/{\rm Hz})$",
]

corner_labelpad = 0.2

# Direct style controls matched to Step 15.
corner_width_inch = 15.8
corner_smooth = 1.0
corner_smooth1d = 1.2
corner_hist_bin_factor = 2
corner_label_fontsize = 15
corner_tick_fontsize = 11
corner_linewidth = 1.25
legend_fontsize = 17
spectrum_label_fontsize = 20
spectrum_tick_fontsize = 15
spectrum_legend_fontsize = 15
output_dpi = 320

all_fig7_samples = np.vstack([case["samples"] for case in fig7_sample_defs])

fig7_ranges = []
for i in range(all_fig7_samples.shape[1]):
    lo, hi = np.percentile(all_fig7_samples[:, i], [0.2, 99.8])
    pad = 0.08 * (hi - lo)
    fig7_ranges.append((lo - pad, hi + pad))

fig7_ranges[8] = (-12.1, -11.1)   # log10 Omega_0
fig7_ranges[9] = (-2.35, -2.12) 

base_case = fig7_sample_defs[0]
fig7_fig = corner.corner(
    base_case["samples"],
    labels=fig7_param_labels,
    color=base_case["color"],
    truths=sw10_truth,
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
    label_kwargs={"fontsize": corner_label_fontsize},
    hist_kwargs={"linewidth": corner_linewidth},
    contour_kwargs={"linewidths": corner_linewidth},
    labelpad=corner_labelpad,
)

for case in fig7_sample_defs[1:]:
    corner.corner(
        case["samples"],
        labels=fig7_param_labels,
        color=case["color"],
        truths=sw10_truth,
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
        label_kwargs={"fontsize": corner_label_fontsize},
        hist_kwargs={"linewidth": corner_linewidth},
        contour_kwargs={"linewidths": corner_linewidth},
        fig=fig7_fig,
        range=fig7_ranges,
        labelpad=corner_labelpad,
    )


def redraw_diagonal_kde(fig, cases, sample_key, truth_values, labels, n_dim):
    axes = np.asarray(fig.axes).reshape((n_dim, n_dim))
    for dim in range(n_dim):
        ax = axes[dim, dim]
        xlim = ax.get_xlim()
        lo, hi = map(float, xlim)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        ax.cla()
        grid = np.linspace(lo, hi, 400)
        ymax = 0.0
        for case in cases:
            values = np.asarray(case[sample_key][:, dim], dtype=float)
            values = values[np.isfinite(values)]
            if values.size < 8 or np.nanstd(values) <= 0.0:
                continue
            density = gaussian_kde(values)(grid)
            ymax = max(ymax, float(np.nanmax(density)))
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


def restore_corner_axis_labels(fig, labels, n_dim, label_fontsize, labelpad=0.18):
    axes = np.asarray(fig.axes[: n_dim * n_dim]).reshape((n_dim, n_dim))
    bottom = n_dim - 1
    for col, label in enumerate(labels):
        ax = axes[bottom, col]
        if col == bottom:
            ax.set_xlabel("")
        else:
            ax.set_xlabel(label, fontsize=label_fontsize, labelpad=labelpad)
            ax.xaxis.label.set_visible(True)
    for row in range(1, n_dim):
        ax = axes[row, 0]
        ax.set_ylabel(labels[row], fontsize=label_fontsize, labelpad=labelpad)
        ax.yaxis.label.set_visible(True)
    final_ax = axes[bottom, bottom]
    final_bbox = final_ax.get_position()
    fig.text(
        0.5 * (final_bbox.x0 + final_bbox.x1),
        max(0.014, final_bbox.y0 - 0.042),
        labels[-1],
        ha="center",
        va="top",
        fontsize=label_fontsize,
    )


redraw_diagonal_kde(fig7_fig, fig7_sample_defs, "samples", sw10_truth, fig7_param_labels, len(fig7_param_labels))

fig7_fig.set_size_inches(corner_width_inch, corner_width_inch * 1.05)
fig7_fig.subplots_adjust(left=0.055, right=0.985, bottom=0.070, top=0.985, wspace=0.0, hspace=0.0)

for ax in fig7_fig.axes:
    ax.tick_params(axis="both", which="major", labelsize=corner_tick_fontsize, length=3.0, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2.0, width=0.6)
    ax.xaxis.label.set_fontsize(corner_label_fontsize)
    ax.yaxis.label.set_fontsize(corner_label_fontsize)

restore_corner_axis_labels(fig7_fig, fig7_param_labels, len(fig7_param_labels), corner_label_fontsize, labelpad=corner_labelpad)

legend_handles = [
    mlines.Line2D([], [], color=case["color"], linewidth=3.0, label=rf"{case['label']}: $\ln\mathrm{{BF}} = {case['ln_bf']:.2f}$")
    for case in fig7_sample_defs
]
fig7_fig.legend(
    handles=legend_handles,
    loc=(0.30, 0.83),
    frameon=False,
    fontsize=legend_fontsize,
    handlelength=2.8,
)

# Spectrum inset in the same open upper-right area used by Step 15.
fig7_inset = fig7_fig.add_axes([0.65, 0.64, 0.32, 0.30])
fig7_inset.loglog(sw10_spectrum_context["frequency"], sw10_spectrum_context["dwd"], color="gray", linewidth=2.1, label=r"$\Omega_{\rm DWD}$")
fig7_inset.loglog(sw10_spectrum_context["frequency"], sw10_spectrum_context["astro"], color="green", linewidth=2.1, label=r"$\Omega_{\rm ast}$")
fig7_inset.loglog(sw10_spectrum_context["frequency"], sw10_spectrum_context["sw"], color="darkorange", linewidth=2.1, label=r"$\Omega_{\rm SW}$")
fig7_inset.loglog(sw10_spectrum_context["frequency"], sw10_spectrum_context["noise"], color="purple", linewidth=2.3, label="Taiji")
fig7_inset.set_xlim(float(np.min(sw10_spectrum_context["frequency"])), float(np.max(sw10_spectrum_context["frequency"])))
positive_values = []
for key in ["dwd", "astro", "sw", "noise"]:
    values = np.asarray(sw10_spectrum_context[key], dtype=float)
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size:
        positive_values.append(valid)
if positive_values:
    positive_all = np.concatenate(positive_values)
    fig7_inset.set_ylim(max(float(np.min(positive_all)) * 0.7, 1.0e-20), float(np.max(positive_all)) * 1.35)
fig7_inset.set_xlabel(r"$f\,{\rm [Hz]}$", fontsize=spectrum_label_fontsize)
fig7_inset.set_ylabel(r"$\Omega(f)$", fontsize=spectrum_label_fontsize, rotation=0, labelpad=20)
fig7_inset.tick_params(axis="both", which="major", labelsize=spectrum_tick_fontsize)
fig7_inset.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.75)
fig7_inset.legend(loc="upper left", frameon=False, fontsize=spectrum_legend_fontsize, ncol=2)

fig7_output.parent.mkdir(parents=True, exist_ok=True)
fig7_fig.savefig(fig7_output, dpi=output_dpi, bbox_inches="tight", pad_inches=0.05)
display(fig7_fig)
plt.close(fig7_fig)

