
from contextlib import redirect_stdout
import io
from pathlib import Path

import corner
import h5py
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from getdist import MCSamples
# display is supplied by the noninteractive reproduction runner
from scipy.stats import gaussian_kde

from py_code.model import PL, calculate_response_function, calculate_transfer_function, dwd
from py_code.parameters import H0

# Manuscript Section 3.2 corner plot:
# equal-arm FD, equal-arm TD, and unequal-arm TD posterior samples for the
# foreground plus stochastic astrophysical background benchmark.
fig_output = FIGURE_DIR / "8para_corner.png"

fd_equal_h5 = REPO_ROOT / "data/ch32_astro_formal_20260520/fd_equal/results/fd_equal_fullxyz_r00_seed2026052000_seg360_fmin1em04_fmax0p15_w0p01_ch32_fd_equal_fullxyz_20260520_bundle_formal_nlive1000_full8.h5"
td_equal_h5 = REPO_ROOT / "data/ch32_astro_formal_20260520/td_equal/results/td_equal_fullxyz_r00_seed3210481897_seg360_fmin1em04_fmax0p15_w0p01_ch32_td_equal_fullxyz_20260520_omastm11p5_bundle_formal_nlive1000_full8.h5"
td_unequal_h5 = REPO_ROOT / "data/ch32_astro_formal_20260520/td_unequal/results/td_unequal_fullxyz_r00_seed38992598_seg360_fmin1em04_fmax0p15_w0p01_ch32_td_unequal_fullxyz_20260520_omastm11p5_bundle_formal_nlive1000_full8.h5"

fd_equal_null_h5 = REPO_ROOT / "data/ch32_astro_formal_20260520/fd_equal/results/fd_equal_fullxyz_r00_seed2026052000_seg360_fmin1em04_fmax0p15_w0p01_ch32_fd_equal_fullxyz_20260520_bundle_bf_nlive1000_dwd_only.h5"
td_equal_null_h5 = REPO_ROOT / "data/ch32_astro_formal_20260520/td_equal/results/td_equal_fullxyz_r00_seed3210481897_seg360_fmin1em04_fmax0p15_w0p01_ch32_td_equal_fullxyz_20260520_omastm11p5_bundle_bf_nlive1000_dwd_only.h5"
td_unequal_null_h5 = REPO_ROOT / "data/ch32_astro_formal_20260520/td_unequal/results/td_unequal_fullxyz_r00_seed38992598_seg360_fmin1em04_fmax0p15_w0p01_ch32_td_unequal_fullxyz_20260520_omastm11p5_bundle_bf_nlive1000_dwd_only.h5"

case_specs = [
    ("FD equal", fd_equal_h5, fd_equal_null_h5, "#D62728"),
    ("TD equal", td_equal_h5, td_equal_null_h5, "#1A9850"),
    ("TD unequal", td_unequal_h5, td_unequal_null_h5, "#7B2CBF"),
]

param_names = [
    "N_acc", "delta_x", "log_A1", "alpha1", "log_A2", "alpha2", "log_Omega_ast", "epsilon"
]
param_labels = [
    r"$N_{\rm acc}$",
    r"$\delta x$",
    r"$\log A_1$",
    r"$\alpha_1$",
    r"$\log A_2$",
    r"$\alpha_2$",
    r"$\log \Omega_{\rm ast}$",
    r"$\varepsilon$",
]



def read_evidence(path):
    with h5py.File(resolve_result_path(path), "r") as h5:
        for key in ("pe/evidence", "evidence", "ln_evidence"):
            if key in h5:
                return float(np.asarray(h5[key]).reshape(()))
    raise KeyError(f"No evidence field found in {path}")


def load_case(label, posterior_path, null_path, color):
    with h5py.File(resolve_result_path(posterior_path), "r") as h5:
        trace = np.asarray(h5["pe/trace"][:, :8], dtype=float)
        truth = np.asarray(h5["pe/truth_fit"][:8], dtype=float)
        mean = np.asarray(h5["pe/posterior_mean"][:8], dtype=float)
        std = np.asarray(h5["pe/posterior_std"][:8], dtype=float)
        evidence = read_evidence(posterior_path)
    ln_bf = evidence - read_evidence(null_path)
    return {
        "label": label,
        "path": posterior_path,
        "trace": trace,
        "truth": truth,
        "mean": mean,
        "std": std,
        "evidence": evidence,
        "ln_bf": ln_bf,
        "color": color,
    }


case_data = [load_case(*spec) for spec in case_specs]
truth = case_data[0]["truth"]

# Direct style controls. Adjust these values in-place when tuning the figure.
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

base_case = case_data[0]
fig = corner.corner(
    base_case["trace"],
    labels=param_labels,
    color=base_case["color"],
    truths=truth,
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
)

for case in case_data[1:]:
    corner.corner(
        case["trace"],
        labels=param_labels,
        color=case["color"],
        truths=truth,
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
        fig=fig,
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
        max(0.014, final_bbox.y0 - 0.035),
        labels[-1],
        ha="center",
        va="top",
        fontsize=label_fontsize,
    )


redraw_diagonal_kde(fig, case_data, "trace", truth, param_labels, len(param_labels))

fig.set_size_inches(corner_width_inch, corner_width_inch * 1.05)
fig.subplots_adjust(left=0.055, right=0.985, bottom=0.070, top=0.985, wspace=0.0, hspace=0.0)

for ax in fig.axes:
    ax.tick_params(axis="both", which="major", labelsize=corner_tick_fontsize, length=3.0, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2.0, width=0.6)
    ax.xaxis.label.set_fontsize(corner_label_fontsize)
    ax.yaxis.label.set_fontsize(corner_label_fontsize)

restore_corner_axis_labels(fig, param_labels, len(param_labels), corner_label_fontsize, labelpad=0.5)

legend_handles = [
    mlines.Line2D([], [], color=case["color"], linewidth=3.0, label=rf"{case['label']}: $\ln\mathrm{{BF}} = {case['ln_bf']:.2f}$")
    for case in case_data
]
fig.legend(
    handles=legend_handles,
    # title="Bayes factors",
    loc=(0.30, 0.83),
    frameon=False,
    fontsize=legend_fontsize,
    title_fontsize=legend_fontsize,
    handlelength=2.8,
)

# Spectrum inset in the open upper-right corner of the triangle plot.
def omega_noise_curve(f_grid, params, seg_idx=1, orbit_type="equal"):
    response_A, _, _ = calculate_response_function(f_grid, seg_idx, orbit_type=orbit_type)
    acc_A, oms_A, _, _, _, _ = calculate_transfer_function(f_grid, seg_idx, orbit_type=orbit_type)
    n_acc = params[0] * 1.0e-15
    delta_x = params[1] * 1.0e-12
    noise_psd = (n_acc**2) * acc_A + (delta_x**2) * oms_A
    conversion = 3.0 * H0**2 * response_A / (4.0 * np.pi**2 * np.clip(f_grid**3, 1.0e-300, None))
    return noise_psd / np.clip(conversion, 1.0e-300, None)


frequency = np.logspace(-4, np.log10(0.15), 900)
seg_index = 1
noise_omega = omega_noise_curve(frequency, truth, seg_idx=seg_index, orbit_type="equal")

spectrum_ax = fig.add_axes([0.65, 0.64, 0.32, 0.30])
spectrum_ax.loglog(frequency, dwd(frequency, truth[2], truth[3], truth[4], truth[5]), color="gray", linewidth=2.1, label=r"$\Omega_{\rm DWD}$")
spectrum_ax.loglog(frequency, PL(frequency, truth[-2], truth[-1]), color="green", linewidth=2.1, label=r"$\Omega_{\rm ast}$")
spectrum_ax.loglog(frequency, noise_omega, color="purple", linewidth=2.3, label=r"Taiji")
spectrum_ax.set_xlim(1.0e-4, 0.15)
spectrum_ax.set_ylim(2.0e-13, 3.0e-7)
spectrum_ax.set_xlabel(r"$f\,{\rm [Hz]}$", fontsize=spectrum_label_fontsize)
spectrum_ax.set_ylabel(r"$\Omega(f)$", fontsize=spectrum_label_fontsize, rotation=0, labelpad=20)
spectrum_ax.tick_params(axis="both", which="major", labelsize=spectrum_tick_fontsize)
spectrum_ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.75)
spectrum_ax.legend(loc="upper left", frameon=False, fontsize=spectrum_legend_fontsize)

fig_output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_output, dpi=output_dpi, bbox_inches="tight", pad_inches=0.05)
display(fig)
plt.close(fig)

# Keep these legacy variables available for the older exploratory cells below,
# while the manuscript figure above is generated directly in this cell.
parameters_true = truth.copy()
mcmc_samples_fd = case_data[0]["trace"]
lnBF_fd = case_data[0]["ln_bf"]
with redirect_stdout(io.StringIO()):
    mcmc_sample_fd = MCSamples(samples=mcmc_samples_fd, names=param_names, labels=[label.strip("$") for label in param_labels])

summary_rows = []
for case in case_data:
    row = {"case": case["label"], "ln BF": case["ln_bf"]}
    for name, value, sigma in zip(param_names, case["mean"], case["std"]):
        row[f"{name} mean"] = value
        row[f"{name} std"] = sigma
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
display(summary_df)
print(f"Wrote manuscript Section 3.2 corner figure: {fig_output}")
