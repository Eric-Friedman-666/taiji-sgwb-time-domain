# Direct plotting cell for Appendix A ratio grid.
matrix_name = "response"
output = FIGURE_DIR / "R_ratio.png"

spec = ch22.MATRIX_SPECS[matrix_name]
f_min = 1.0e-4
f_max = 0.15
dpi = 220

validation = CH22_VALIDATIONS[matrix_name]
freq = np.asarray(validation["frequency"], dtype=float)
mask = np.isfinite(freq) & (freq >= f_min) & (freq <= f_max)
calc = np.asarray(validation["calc"], dtype=np.complex128)
sim = np.asarray(validation["sim"], dtype=np.complex128)
n_real = int(validation["n_real"])
centers = np.asarray(validation["centers"], dtype=float)
notch_half_width = float(validation["notch_half_width"])
point_mask = ch22.thin_plot_mask(mask, target_points=3600)

colors = {"ratio": "#E69F00", "reference": "#00897B"}
sim_legend_label = "Simulation/model, $N_{\\rm real}=%d$" % n_real

fig, axes = plt.subplots(3, 3, figsize=(13.8, 11.2), sharex=True)
fig.subplots_adjust(left=0.145, right=0.992, bottom=0.085, top=0.895, wspace=0.25, hspace=0.34)

for panel_idx, (ax, (entry, i, j, mode)) in enumerate(zip(axes.ravel(), ch22.PANEL_SPECS)):
    col_idx = panel_idx % axes.shape[1]
    is_phase = mode == "phase"
    ch22.add_notch_bands(ax, centers, notch_half_width)

    if is_phase:
        phase_resid = ch22.wrap_phase_residual(sim[:, i, j] / calc[:, i, j])
        ax.axhline(0.0, color=colors["reference"], ls="--", lw=1.0, label="Ideal")
        ax.scatter(
            freq[point_mask],
            phase_resid[point_mask],
            s=5.0,
            color=colors["ratio"],
            alpha=0.56,
            linewidths=0,
            label=sim_legend_label,
        )
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
        ax.set_yticklabels([r"$-0.5$", r"$-0.25$", r"$0$", r"$0.25$", r"$0.5$"])
        if col_idx == 0:
            ch22.set_readable_ylabel(ax, r"$\arg\!\left(\frac{\widehat G_{IJ}}{P_{IJ}}\right)\,[{\rm rad}]$", labelpad=25.0)
        ch22.style_axis_text(ax)
        panel_key = entry
    else:
        calc_y = ch22.finite_positive(np.abs(calc[:, i, j]))
        sim_y = ch22.finite_positive(np.abs(sim[:, i, j]))
        ratio = sim_y / calc_y
        ax.axhline(1.0, color=colors["reference"], ls="--", lw=1.0, label="Ideal")
        ax.scatter(
            freq[point_mask],
            ratio[point_mask],
            s=5.0,
            color=colors["ratio"],
            alpha=0.56,
            linewidths=0,
            label=sim_legend_label,
        )
        ax.set_ylim(0.9, 1.1)
        if col_idx == 0:
            ch22.set_readable_ylabel(ax, r"$\left|\frac{\widehat{G}_{IJ}(f_k)}{P_{IJ}(f_k)}\right|$", labelpad=35.0)
        panel_key = entry

    ax.set_title(panel_key, fontsize=ch22.PANEL_TITLE_FONTSIZE)
    ch22.setup_axis(ax, freq, mask, is_phase, phase_range="none")

for ax in axes[-1, :]:
    ax.set_xlabel(r"$f\,[{\rm Hz}]$", fontsize=ch22.AXIS_LABEL_FONTSIZE)

handles, labels = axes[0, 0].get_legend_handles_labels()
unique = {
    label: ch22.darker_sim_marker(label, colors["ratio"]) if label.startswith("Simulation/model") else handle
    for handle, label in zip(handles, labels)
}
if centers.size:
    unique["Notch"] = ch22.notch_legend_handle()
legend_order = [sim_legend_label, "Ideal", "Notch"]
fig.legend(
    [unique[label] for label in legend_order if label in unique],
    [label for label in legend_order if label in unique],
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 0.975),
    frameon=False,
    fontsize=ch22.LEGEND_FONTSIZE,
    handlelength=2.0,
    columnspacing=1.6,
)

output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
display(fig)
