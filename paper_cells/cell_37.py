from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from paper_io import load_cases, read_result

cases = load_cases()
diagnostics = pd.read_csv(OUTPUT_DIR / "tables/astro_diagnostics.csv").set_index("log10_Omega_ast")
rows = []
for amp in sorted(cases["astro_scan"]["fd_equal"], key=float):
    value = float(amp)
    metric = diagnostics.loc[value]
    row = {"log10_Omega_ast_true": value,
           "fisher_rel_unc_pct": metric["fisher_logomega_relative_uncertainty_percent"],
           "fd_snr_rel": metric["snr_rel"], "snr_abs": metric["snr_abs"]}
    for cfg, prefix in (("fd_equal", "fd"), ("td_equal", "td_equal"), ("td_unequal", "td_unequal")):
        spec = cases["astro_scan"][cfg][amp]
        full, null = read_result(spec["posterior"]), read_result(spec["null"])
        row[prefix + "_rel_unc_pct"] = 100.0 * full["std"][6] / abs(full["truth"][6])
        row[prefix + "_lnBF"] = full["evidence"] - null["evidence"]
    rows.append(row)
astro_8_summary_df = pd.DataFrame(rows)
fig6_output = FIGURE_DIR / "ast_scan.png"
with plt.rc_context({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 20,
}):
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    x = astro_8_summary_df["log10_Omega_ast_true"]
    ax.plot(
        x,
        astro_8_summary_df["fisher_rel_unc_pct"],
        color="#1F5BFF",
        linewidth=2.6,
        label="FIM forecast",
        zorder=2,
    )
    ax.plot(
        x,
        astro_8_summary_df["fd_rel_unc_pct"],
        color="#E52521",
        marker="o",
        linewidth=2.2,
        markersize=8.5,
        label="FD PE",
        zorder=3,
    )
    ax.plot(
        x,
        astro_8_summary_df["td_equal_rel_unc_pct"],
        color="#19A64A",
        marker="^",
        linewidth=2.2,
        markersize=9.0,
        label="TD equal PE",
        zorder=3,
    )
    ax.plot(
        x,
        astro_8_summary_df["td_unequal_rel_unc_pct"],
        color="#7A24D8",
        marker="s",
        linewidth=2.2,
        markersize=8.2,
        label="TD unequal PE",
        zorder=3,
    )
    ax.set_xlabel(r"$\log_{10}\Omega_{\rm ast}$", fontsize=24)
    ax.set_ylabel(
        r"$\frac{\Delta\log_{10}\Omega_{\rm ast}}{|\log_{10}\Omega_{\rm ast}|}\;(\%)$",
        fontsize=24,
        rotation=0,
        labelpad=70,
    )
    ax.set_xlim(-12.53, -11.47)
    ax.set_ylim(0.08, 1.10)
    ax.set_xticks([-12.5, -12.3, -12.1, -11.9, -11.7, -11.5])
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=14, length=6, width=1.1)
    ax.tick_params(axis="both", which="minor", length=3, width=0.9)
    ax.grid(True, which="major", color="#BFC7D5", alpha=0.55, linewidth=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.legend(frameon=True, fontsize=15, loc="upper right")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.16)
    fig6_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig6_output, dpi=320, bbox_inches="tight")
    display(fig)
    plt.close(fig)

