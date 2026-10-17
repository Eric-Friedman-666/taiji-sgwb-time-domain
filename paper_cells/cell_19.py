# Direct plotting cell for the Appendix A full XYZ CSD ratio figure.
output = FIGURE_DIR / "PSD_ratio.png"

freq = CH31_FREQ
p_sim = CH31_P_SIM
p_model = CH31_P_MODEL
attrs = CH31_ATTRS
ratio = np.full_like(p_sim, np.nan + 0j, dtype=np.complex128)
good = np.isfinite(p_sim) & np.isfinite(p_model) & (np.abs(p_model) > 0)
ratio[good] = p_sim[good] / p_model[good]

def taiji_reference_notches(f_min, f_max, half_width=0.010):
    c_si = 299_792_458.0
    l_taiji = 3.0e9
    intervals = []
    m = 1
    while True:
        center = m * c_si / (4.0 * l_taiji)
        if center > f_max + half_width:
            break
        if center >= f_min - half_width:
            intervals.append((max(f_min, center - half_width), min(f_max, center + half_width)))
        m += 1
    return intervals

def shade_ch31_notches(ax, intervals):
    for start, stop in intervals:
        ax.axvspan(start, stop, color=ch22.NOTCH_COLOR, alpha=ch22.NOTCH_ALPHA, lw=0, zorder=0)

channels = ("X", "Y", "Z")
diag_entries = [(0, 0), (1, 1), (2, 2)]
offdiag_entries = [(0, 1), (0, 2), (1, 2)]
intervals = taiji_reference_notches(float(freq.min()), float(freq.max()), half_width=0.010)
ratio_color = "#E69F00"
ref_color = "#00897B"
n_real = int(attrs.get("n_real", 1000))
ratio_label = "Simulation/model, $N_{\\rm real}=%d$" % n_real

fig, axes = plt.subplots(3, 3, figsize=(13.8, 11.2), dpi=210, sharex=True)
fig.subplots_adjust(left=0.145, right=0.992, bottom=0.085, top=0.895, wspace=0.25, hspace=0.34)

for col, (i, j) in enumerate(diag_entries):
    ax = axes[0, col]
    label = f"{channels[i]}{channels[j]}"
    shade_ch31_notches(ax, intervals)
    ax.semilogx(
        freq,
        np.abs(ratio[:, i, j]),
        marker=".",
        linestyle="None",
        ms=2.2,
        alpha=0.70,
        color=ratio_color,
        zorder=2,
        label=ratio_label,
    )
    ax.axhline(1.0, color=ref_color, ls="--", lw=1.0, zorder=1, label="Ideal")
    ax.set_title(label, fontsize=ch22.PANEL_TITLE_FONTSIZE)
    ax.set_ylim(0.0, 2.2)
    ax.grid(True, which="both", alpha=0.22, lw=0.45)
    ax.set_xscale("log")
    ax.set_xlim(float(freq.min()), float(freq.max()))
    ch22.style_axis_text(ax)
    if col == 0:
        ch22.set_readable_ylabel(
            ax,
            r"$\left|\frac{\widehat{G}_{IJ}(f_k)}{P_{IJ}(f_k)}\right|$",
            fontsize=ch22.AXIS_LABEL_FONTSIZE,
            labelpad=42.0,
        )

for col, (i, j) in enumerate(offdiag_entries):
    ax = axes[1, col]
    label = f"{channels[i]}{channels[j]}"
    shade_ch31_notches(ax, intervals)
    ax.semilogx(
        freq,
        np.abs(ratio[:, i, j]),
        marker=".",
        linestyle="None",
        ms=2.2,
        alpha=0.70,
        color=ratio_color,
        zorder=2,
        label=ratio_label,
    )
    ax.axhline(1.0, color=ref_color, ls="--", lw=1.0, zorder=1, label="Ideal")
    ax.set_title(label, fontsize=ch22.PANEL_TITLE_FONTSIZE)
    ax.set_ylim(0.0, 2.2)
    ax.grid(True, which="both", alpha=0.22, lw=0.45)
    ax.set_xscale("log")
    ax.set_xlim(float(freq.min()), float(freq.max()))
    ch22.style_axis_text(ax)
    if col == 0:
        ch22.set_readable_ylabel(
            ax,
            r"$\left|\frac{\widehat{G}_{IJ}(f_k)}{P_{IJ}(f_k)}\right|$",
            fontsize=ch22.AXIS_LABEL_FONTSIZE,
            labelpad=42.0,
        )

for col, (i, j) in enumerate(offdiag_entries):
    ax = axes[2, col]
    label = f"{channels[i]}{channels[j]}"
    shade_ch31_notches(ax, intervals)
    ax.semilogx(
        freq,
        np.angle(ratio[:, i, j]),
        marker=".",
        linestyle="None",
        ms=2.2,
        alpha=0.70,
        color=ratio_color,
        zorder=2,
        label=ratio_label,
    )
    ax.axhline(0.0, color=ref_color, ls="--", lw=1.0, zorder=1, label="Ideal")
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, 0.0, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    ax.set_title(label, fontsize=ch22.PANEL_TITLE_FONTSIZE)
    ax.grid(True, which="both", alpha=0.22, lw=0.45)
    ax.set_xscale("log")
    ax.set_xlim(float(freq.min()), float(freq.max()))
    ax.set_xlabel(r"$f\,[{\rm Hz}]$", fontsize=ch22.AXIS_LABEL_FONTSIZE)
    ch22.style_axis_text(ax)
    if col == 0:
        ch22.set_readable_ylabel(
            ax,
            r"$\arg\!\left[\frac{\widehat{G}_{IJ}(f_k)}{P_{IJ}(f_k)}\right]$",
            fontsize=ch22.AXIS_LABEL_FONTSIZE,
            labelpad=42.0,
        )

handles = [
    ch22.darker_sim_marker(ratio_label, ratio_color),
    mlines.Line2D([], [], color=ref_color, lw=1.2, ls="--", label="Ideal"),
    mlines.Line2D([], [], color=ch22.NOTCH_COLOR, lw=8, alpha=ch22.NOTCH_ALPHA, label="Notch"),
]
fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.975),
    ncol=3,
    frameon=False,
    fontsize=ch22.LEGEND_FONTSIZE,
    handlelength=2.0,
    columnspacing=1.6,
)

output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=210, bbox_inches="tight", pad_inches=0.05)
display(fig)
