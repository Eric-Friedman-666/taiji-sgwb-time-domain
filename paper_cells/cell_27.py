from contextlib import redirect_stdout
import io
from pathlib import Path

import h5py
import matplotlib.lines as mlines
from matplotlib.contour import QuadContourSet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# display is supplied by the noninteractive reproduction runner
from getdist import MCSamples, plots

# Manuscript Fig. 3: minimal four-parameter full-covariance validation corner plot.
# Edit the plotting parameters in this cell directly when adjusting the figure.
pe_dir = REPO_ROOT / "data/ast4_pe_performance_highnlive_20260511/results"
run_index = 1
run_tag = "fmin1e4_w010_nlive1000"
pe_h5 = pe_dir / f"td_full_xyz_tenseg_1day_full4_r{run_index:02d}_nlive1000_{run_tag}.h5"
fig3_output = FIGURE_DIR / "4para_corner.png"

truth = np.array([3.0, 8.0, -11.5, 2.0 / 3.0], dtype=float)
param_names = ["N_acc_amp", "delta_x_amp", "log_Omega_ast", "alpha_ast"]
param_labels = [
    r"N_{\rm acc}/10^{-15}",
    r"\delta x/10^{-12}",
    r"\log_{10}\Omega_{\rm ast}",
    r"\varepsilon",
]

with h5py.File(resolve_result_path(pe_h5), "r") as h5:
    trace = np.asarray(h5["pe/trace"][:, :4], dtype=float)
    parameters_true = np.asarray(h5["parameters_true"][:4], dtype=float)
    posterior_mean = np.asarray(h5["pe/posterior_mean"][:4], dtype=float)
    posterior_std = np.asarray(h5["pe/posterior_std"][:4], dtype=float)
    evidence = float(h5["pe/evidence"][()])

if not np.allclose(parameters_true, truth, rtol=0, atol=1.0e-9):
    raise RuntimeError(f"Unexpected truth vector in {pe_h5}: {parameters_true}")

# Direct style controls for the corner plot.
corner_width_inch = 9.0
axes_label_size = 18
axes_tick_size = 13
legend_size = 14
contour_linewidth = 1.9
contour_color = "#0072B2"
truth_marker_color = "gray"
output_dpi = 320

with redirect_stdout(io.StringIO()):
    samples = MCSamples(samples=trace, names=param_names, labels=param_labels)

plotter = plots.get_subplot_plotter(width_inch=corner_width_inch)
plotter.settings.axes_labelsize = axes_label_size
plotter.settings.axes_fontsize = axes_tick_size
plotter.settings.legend_fontsize = legend_size
plotter.settings.axis_marker_color = truth_marker_color
plotter.settings.num_plot_contours = 2
plotter.settings.linewidth_contour = contour_linewidth
plotter.triangle_plot(
    [samples],
    filled=False,
    markers=truth,
    colors=[contour_color],
    contour_colors=[contour_color],
    legend_labels=None,
    legend_loc=None,
    contour_lws=[contour_linewidth],
)

# Matplotlib orders the two unfilled contour levels from outer to inner.
for ax in plotter.fig.axes:
    for contour_set in ax.collections:
        if isinstance(contour_set, QuadContourSet):
            contour_set.set_linestyles(["--", "-"])  # 2 sigma outer, 1 sigma inner

legend_handles = [
    mlines.Line2D([], [], color=contour_color, linewidth=contour_linewidth, label=r"$1\sigma$ credible contour"),
    mlines.Line2D([], [], color=contour_color, linewidth=contour_linewidth, linestyle="--", label=r"$2\sigma$ credible contour"),
]
plotter.fig.legend(handles=legend_handles, loc=(0.55, 0.81), fontsize=legend_size, frameon=False)

fig3_output.parent.mkdir(parents=True, exist_ok=True)
plotter.fig.savefig(fig3_output, dpi=output_dpi, bbox_inches="tight")
display(plotter.fig)

fig3_summary_df = pd.DataFrame(
    {
        "parameter": param_labels,
        "injected": truth,
        "posterior mean": posterior_mean,
        "posterior std": posterior_std,
        "relative uncertainty (%)": posterior_std / np.abs(truth) * 100.0,
    }
)

display(fig3_summary_df)

