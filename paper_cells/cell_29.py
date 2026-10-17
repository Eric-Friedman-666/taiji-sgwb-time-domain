from pathlib import Path

import h5py
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# display is supplied by the noninteractive reproduction runner

# Manuscript Fig. 4: ten-realization recovery summary for the validation benchmark.
# This cell is intentionally explicit so markers, labels, legends, and fonts can be tuned in place.
pe_dir = REPO_ROOT / "data/ast4_pe_performance_highnlive_20260511/results"
run_tag = "fmin1e4_w010_nlive1000"
fig4_output = FIGURE_DIR / "4para_unbias_check.png"
TABLE_DIR = OUTPUT_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
fig4_csv_output = TABLE_DIR / "ch31_fig4_formal_mask_10run_xyz_recovery_bias_20260520.csv"

truth = np.array([3.0, 8.0, -11.5, 2.0 / 3.0], dtype=float)
param_names = ["N_acc_amp", "delta_x_amp", "log_Omega_ast", "alpha_ast"]
panel_labels = [
    r"$N_{\rm acc}/10^{-15}$",
    r"$\delta x/10^{-12}$",
    r"$\log_{10}\Omega_{\rm ast}$",
    r"$\varepsilon$",
]

rows = []
for file_index in range(10):
    realization_index = file_index + 1
    pe_h5 = pe_dir / f"td_full_xyz_tenseg_1day_full4_r{file_index:02d}_nlive1000_{run_tag}.h5"
    with h5py.File(resolve_result_path(pe_h5), "r") as h5:
        parameters_true = np.asarray(h5["parameters_true"][:4], dtype=float)
        posterior_mean = np.asarray(h5["pe/posterior_mean"][:4], dtype=float)
        posterior_std = np.asarray(h5["pe/posterior_std"][:4], dtype=float)
        evidence = float(h5["pe/evidence"][()])
    if not np.allclose(parameters_true, truth, rtol=0, atol=1.0e-9):
        raise RuntimeError(f"Unexpected truth vector in {pe_h5}: {parameters_true}")
    rows.append(
        {
            "realization": realization_index,
            "mean": posterior_mean,
            "std": posterior_std,
            "evidence": evidence,
            "file": pe_h5.name,
        }
    )

means = np.vstack([row["mean"] for row in rows])
stds = np.vstack([row["std"] for row in rows])
realizations = np.array([row["realization"] for row in rows], dtype=int)
mean_of_means = means.mean(axis=0)
std_of_means = means.std(axis=0, ddof=1)
mean_posterior_std = stds.mean(axis=0)

# Direct style controls for the recovery summary figure.
fig_width = 9.0
fig_height = 8.2
point_color = "#0072B2"
error_color = "#7DB8DA"
injected_color = "#D55E00"
mean_color = "#009E73"
label_size = 16
tick_size = 13
legend_size = 12
output_dpi = 320

fig, axes = plt.subplots(4, 1, figsize=(fig_width, fig_height), sharex=True)
for ax, i, label in zip(axes, range(4), panel_labels):
    ax.errorbar(
        realizations,
        means[:, i],
        yerr=stds[:, i],
        fmt="o",
        color=point_color,
        ecolor=error_color,
        elinewidth=1.8,
        capsize=0,
        markersize=5.5,
        zorder=3,
    )
    ax.axhline(truth[i], color=injected_color, linestyle="-", linewidth=2.0, zorder=2)
    ax.axhline(mean_of_means[i], color=mean_color, linestyle="--", linewidth=2.0, zorder=2)
    ax.set_ylabel(label, rotation=0, labelpad=50, fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.grid(True, alpha=0.28, linewidth=0.7)
    ax.set_xlim(0.5, 10.5)

axes[-1].set_xlabel("Realization", fontsize=label_size)
axes[-1].set_xticks(realizations)
legend_handles = [
    mlines.Line2D([], [], color=point_color, marker="o", linestyle="None", markersize=5.5, label="Posterior mean"),
    mlines.Line2D([], [], color=injected_color, linestyle="-", linewidth=2.0, label="Injected value"),
    mlines.Line2D([], [], color=mean_color, linestyle="--", linewidth=2.0, label="Mean over realizations"),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.57, 0.995),
    ncol=3,
    frameon=False,
    fontsize=legend_size,
    columnspacing=1.4,
    handlelength=2.1,
    borderaxespad=0.0,
)
fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94], h_pad=0.28)

fig4_output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig4_output, dpi=output_dpi, bbox_inches="tight")
display(fig)

# fig4_recovery_df = pd.DataFrame(
#     [
#         {
#             "realization": row["realization"],
#             "parameter": panel_labels[i],
#             "injected": truth[i],
#             "posterior mean": row["mean"][i],
#             "posterior std": row["std"][i],
#             "bias": row["mean"][i] - truth[i],
#         }
#         for row in rows
#         for i in range(4)
#     ]
# )
# display(fig4_recovery_df)

fig4_summary_df = pd.DataFrame(
    {
        "parameter": panel_labels,
        "injected": truth,
        "mean over realizations": mean_of_means,
        "bias of mean": mean_of_means - truth,
        "scatter of posterior means": std_of_means,
        "mean posterior std": mean_posterior_std,
        "relative bias (%)": (mean_of_means - truth) / np.abs(truth) * 100.0,
    }
)
display(fig4_summary_df)
