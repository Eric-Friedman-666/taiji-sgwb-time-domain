# Direct plotting cell for the Sec. 2.2 off-diagonal residual diagnostic.
output = FIGURE_DIR / "cross_ru.png"
panel_specs = ch22.OFFDIAG_SPECS
title = r"Off diagonal cross spectrum scaled squared normalized residual $N_{\rm real}[r_{IJ}^{\kappa}]^2$"
ylabel = r"$N_{\rm real}\left(r_{IJ}^{\kappa}\right)^2$"
f_min = 1.0e-4
f_max = 0.15
dpi = 250

colors = {
    "td_unequal": "#D18A00",
    "fd_static": "#0072B2",
    "td_theory": "#009E73",
    "fd_theory": "#D62728",
}

fig, axes = plt.subplots(3, 3, figsize=(13.8, 10.2), sharex=True, sharey=True)
fig.subplots_adjust(left=0.130, right=0.992, bottom=0.095, top=0.900, wspace=0.16, hspace=0.32)
residuals_for_ylim = []
has_notch = False

for row, matrix_name in enumerate(ch22.MATRIX_SPECS):
    spec = ch22.MATRIX_SPECS[matrix_name]
    validation = CH22_VALIDATIONS[matrix_name]
    freq = np.asarray(validation["frequency"], dtype=float)
    calc = np.asarray(validation["calc"], dtype=np.complex128)
    sim = np.asarray(validation["sim"], dtype=np.complex128)
    fd_freq = np.asarray(CH22_FD["frequency"], dtype=float)
    fd_raw = np.asarray(CH22_FD[matrix_name], dtype=np.complex128)

    if fd_freq.shape == freq.shape and np.allclose(fd_freq, freq):
        fd_calc = fd_raw
    else:
        fd_calc = np.empty_like(calc)
        for a in range(calc.shape[1]):
            for b in range(calc.shape[2]):
                fd_calc[:, a, b] = ch22.interpolate_complex(freq, fd_freq, fd_raw[:, a, b])

    centers = np.asarray(validation["centers"], dtype=float)
    notch_half_width = float(validation["notch_half_width"])
    has_notch = has_notch or bool(centers.size)
    mask = np.isfinite(freq) & (freq >= f_min) & (freq <= f_max)
    point_mask = ch22.thin_plot_mask(mask, target_points=2600)

    for col, (entry, i, j) in enumerate(panel_specs):
        ax = axes[row, col]
        ch22.add_notch_bands(ax, centers, notch_half_width)

        residual_by_reference = {}
        for reference_name, reference_matrix in (("td_unequal", calc), ("fd_static", fd_calc)):
            residual_by_reference[reference_name] = np.abs(sim[:, i, j] - reference_matrix[:, i, j]) / ch22.finite_positive(
                np.abs(reference_matrix[:, i, j])
            )
        squared_residual_by_reference = {
            reference_name: residual**2 for reference_name, residual in residual_by_reference.items()
        }

        n_real = int(validation["n_real"])
        plot_scale = float(n_real)
        td_scatter_numerator = np.sqrt(
            ch22.finite_positive(np.abs(calc[:, i, i])) * ch22.finite_positive(np.abs(calc[:, j, j]))
        ) / np.sqrt(float(n_real))
        rms_by_reference = {
            "td_unequal": td_scatter_numerator / ch22.finite_positive(np.abs(calc[:, i, j])),
            "fd_static": td_scatter_numerator / ch22.finite_positive(np.abs(fd_calc[:, i, j])),
        }
        squared_expectation_by_reference = {
            reference_name: rms**2 for reference_name, rms in rms_by_reference.items()
        }

        ax.scatter(
            freq[point_mask],
            plot_scale * squared_residual_by_reference["fd_static"][point_mask],
            s=13.0,
            color=colors["fd_static"],
            alpha=0.58,
            marker="^",
            linewidths=0,
            label="FD residuals",
            zorder=1,
        )
        ax.scatter(
            freq[point_mask],
            plot_scale * squared_residual_by_reference["td_unequal"][point_mask],
            s=8.0,
            color=colors["td_unequal"],
            alpha=0.72,
            marker="x",
            linewidths=0.55,
            label="TD residuals",
            zorder=4,
        )
        ax.plot(
            freq[mask],
            plot_scale * squared_expectation_by_reference["td_unequal"][mask],
            color=colors["td_theory"],
            lw=2.15,
            alpha=1.0,
            label="TD expected level",
            zorder=30,
        )
        ax.plot(
            freq[mask],
            plot_scale * squared_expectation_by_reference["fd_static"][mask],
            color=colors["fd_theory"],
            lw=1.55,
            ls=(0, (5.0, 2.4)),
            alpha=1.0,
            dash_capstyle="round",
            label="FD expected level",
            zorder=80,
        )

        ax.set_title(entry, fontsize=ch22.PANEL_TITLE_FONTSIZE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(float(np.nanmin(freq[mask])), float(np.nanmax(freq[mask])))
        ax.grid(True, which="both", alpha=0.24, lw=0.45)
        ch22.style_axis_text(ax)
        if col == 0:
            ch22.set_readable_ylabel(ax, ylabel, fontsize=ch22.SCATTER_AXIS_LABEL_FONTSIZE, labelpad=34.0)
        if row == 2:
            ax.set_xlabel(r"$f\,[{\rm Hz}]$", fontsize=ch22.SCATTER_AXIS_LABEL_FONTSIZE)

        for values in (
            plot_scale * squared_residual_by_reference["td_unequal"],
            plot_scale * squared_residual_by_reference["fd_static"],
            plot_scale * squared_expectation_by_reference["td_unequal"],
            plot_scale * squared_expectation_by_reference["fd_static"],
        ):
            finite_values = values[mask]
            finite_values = finite_values[np.isfinite(finite_values) & (finite_values > 0.0)]
            if finite_values.size:
                residuals_for_ylim.append(finite_values)

if residuals_for_ylim:
    merged_residuals = np.concatenate(residuals_for_ylim)
    y_lo = float(np.nanpercentile(merged_residuals, 1.0) * 0.75)
    y_hi = float(np.nanpercentile(merged_residuals, 99.0) * 1.20)
else:
    y_lo, y_hi = 2.0e-3, 3.0

y_lo = min(max(y_lo, 1.0e-5), 2.0e-2)
y_hi = min(max(y_hi, 0.3), 50.0)
for ax in axes.flat:
    ax.set_ylim(10**-1, 10**4)

handles, labels = axes[0, 0].get_legend_handles_labels()
unique = dict(zip(labels, handles))
if has_notch:
    unique["Notch"] = ch22.notch_legend_handle()
legend_order = ["TD residuals", "TD expected level", "FD residuals", "FD expected level", "Notch"]
fig.legend(
    [unique[label] for label in legend_order if label in unique],
    [label for label in legend_order if label in unique],
    loc="upper center",
    ncol=5,
    bbox_to_anchor=(0.5, 0.975),
    frameon=False,
    fontsize=ch22.SCATTER_LEGEND_FONTSIZE,
    handlelength=2.2,
    columnspacing=1.6,
    markerscale=2.8,
)

output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
display(fig)
