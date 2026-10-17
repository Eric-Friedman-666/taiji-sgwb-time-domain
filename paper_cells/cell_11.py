# Direct plotting cell for Appendix A overlay grid.
matrix_name = "response"
output = FIGURE_DIR / "R_validation.png"

spec = ch22.MATRIX_SPECS[matrix_name]
f_min = 1.0e-4
f_max = 0.15
notch_half_width = 0.010
dpi = 220

freq = CH22_FD["frequency"]
mask = np.isfinite(freq) & (freq >= f_min) & (freq <= f_max)
fd_matrix = CH22_FD[matrix_name]
td_eq_matrix = CH22_TD_EQUAL[matrix_name]
td_uneq_matrix = CH22_TD_UNEQUAL[matrix_name]
validation = CH22_VALIDATIONS[matrix_name]
sim_freq = np.asarray(validation["frequency"], dtype=float)
sim_matrix = np.asarray(validation["sim"], dtype=np.complex128)
sim_mask = np.isfinite(sim_freq) & (sim_freq >= f_min) & (sim_freq <= f_max)
sim_point_mask = ch22.thin_plot_mask(sim_mask, target_points=2200)
n_real = int(validation["n_real"])
sim_label = f"Average of {n_real} realizations"

colors = {
    "fd": "#202020",
    "td_equal": "#1f77b4",
    "td_unequal": "#7b2cbf",
    "sim": "#E69F00",
}

fig, axes = plt.subplots(3, 3, figsize=(13.8, 11.2), sharex=True)
fig.subplots_adjust(left=0.145, right=0.992, bottom=0.090, top=0.875, wspace=0.25, hspace=0.34)

for panel_idx, (ax, (entry, i, j, mode)) in enumerate(zip(axes.ravel(), ch22.PANEL_SPECS)):
    col_idx = panel_idx % axes.shape[1]
    is_phase = mode == "phase"
    ch22.add_notch_bands(ax, CH22_FD.get("notch_centers", np.asarray([])), notch_half_width)

    if is_phase:
        fd_y = np.angle(fd_matrix[:, i, j])
        td_eq_y = td_eq_matrix[f"{i}{j}"]["phase_mean"]
        td_uneq_y = td_uneq_matrix[f"{i}{j}"]["phase_mean"]
        sim_y = np.angle(sim_matrix[:, i, j])

        ax.semilogx(freq[mask], fd_y[mask], color=colors["fd"], lw=1.55, label="Static equal")
        ax.semilogx(freq[mask], td_eq_y[mask], color=colors["td_equal"], lw=1.35, label="TD equal")
        ax.semilogx(freq[mask], td_uneq_y[mask], color=colors["td_unequal"], lw=1.35, label="TD unequal")
        ax.scatter(
            sim_freq[sim_point_mask],
            sim_y[sim_point_mask],
            s=4.0,
            color=colors["sim"],
            alpha=0.18,
            linewidths=0,
            label=sim_label,
            zorder=2,
        )
        ax.set_ylim(-ch22.PHASE_YLIM_PAD * np.pi, ch22.PHASE_YLIM_PAD * np.pi)
        ax.set_yticks([-np.pi, 0.0, np.pi])
        ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
        if col_idx == 0:
            ch22.set_readable_ylabel(ax, spec["phase_ylabel"], labelpad=30.0)
        panel_key = entry
    else:
        fd_y = ch22.finite_positive(np.abs(fd_matrix[:, i, j]))
        td_eq_mean, td_eq_lo, td_eq_hi = tuple(td_eq_matrix[f"{i}{j}"][key] for key in ("amplitude_mean", "amplitude_p05", "amplitude_p95"))
        td_uneq_mean, td_uneq_lo, td_uneq_hi = tuple(td_uneq_matrix[f"{i}{j}"][key] for key in ("amplitude_mean", "amplitude_p05", "amplitude_p95"))
        sim_y = ch22.finite_positive(np.abs(sim_matrix[:, i, j]))

        ax.loglog(freq[mask], fd_y[mask], color=colors["fd"], lw=1.55, label="Static equal")
        ax.loglog(freq[mask], ch22.finite_positive(td_eq_mean)[mask], color=colors["td_equal"], lw=1.35, label="TD equal")
        ax.fill_between(
            freq[mask],
            ch22.finite_positive(td_eq_lo)[mask],
            ch22.finite_positive(td_eq_hi)[mask],
            color=colors["td_equal"],
            alpha=0.10,
            lw=0,
        )
        ax.loglog(freq[mask], ch22.finite_positive(td_uneq_mean)[mask], color=colors["td_unequal"], lw=1.35, label="TD unequal")
        ax.fill_between(
            freq[mask],
            ch22.finite_positive(td_uneq_lo)[mask],
            ch22.finite_positive(td_uneq_hi)[mask],
            color=colors["td_unequal"],
            alpha=0.12,
            lw=0,
        )
        ax.scatter(
            sim_freq[sim_point_mask],
            sim_y[sim_point_mask],
            s=4.0,
            color=colors["sim"],
            alpha=0.18,
            linewidths=0,
            label=sim_label,
            zorder=2,
        )
        ax.set_ylim(10**-10, 10**2)
        if col_idx == 0:
            ch22.set_readable_ylabel(ax, spec["ylabel"], labelpad=30.0)
        panel_key = entry

    ax.set_title(panel_key, fontsize=ch22.PANEL_TITLE_FONTSIZE)
    ch22.setup_axis(ax, freq, mask, is_phase, phase_range="none" if is_phase else "zero_to_twopi")

for ax in axes[-1, :]:
    ax.set_xlabel(r"$f\,[{\rm Hz}]$", fontsize=ch22.AXIS_LABEL_FONTSIZE, labelpad=5.0)

handles, labels = axes[0, 0].get_legend_handles_labels()
handles = [
    ch22.darker_sim_marker(sim_label, colors["sim"]) if label == sim_label else handle
    for handle, label in zip(handles, labels)
]
handles, labels = ch22.append_notch_legend(handles, labels, CH22_FD.get("notch_centers", np.asarray([])))
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    bbox_to_anchor=(0.5, 0.975),
    frameon=False,
    fontsize=ch22.LEGEND_FONTSIZE,
    handlelength=2.1,
    columnspacing=1.5,
)

output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
display(fig)
