# 8-parameter relative uncertainty / SNR / BF table (-11.5 to -12.5)
if "astro_8_summary_df" not in globals():
    raise RuntimeError("Run the previous 8-parameter relative-uncertainty block first.")

astro_8_table = pd.DataFrame(
    {
        "log10_Omega_ast_true": astro_8_summary_df["log10_Omega_ast_true"],
        "SNR_abs": astro_8_summary_df["snr_abs"],
        "SNR_rel": astro_8_summary_df["fd_snr_rel"],
        "Fisher rel. unc. (%)": astro_8_summary_df["fisher_rel_unc_pct"],
        "FD equal rel. unc. (%)": astro_8_summary_df["fd_rel_unc_pct"],
        "TD equal rel. unc. (%)": astro_8_summary_df["td_equal_rel_unc_pct"],
        "TD unequal rel. unc. (%)": astro_8_summary_df["td_unequal_rel_unc_pct"],
        "FD equal ln BF": astro_8_summary_df["fd_lnBF"],
        "TD equal ln BF": astro_8_summary_df["td_equal_lnBF"],
        "TD unequal ln BF": astro_8_summary_df["td_unequal_lnBF"],
    }
)

display(astro_8_table.round(4))
TABLE_DIR = OUTPUT_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
astro_8_summary_df.to_csv(TABLE_DIR / "astro_uncertainty_snr_bf_table_115_125.csv", index=False)
