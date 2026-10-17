
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
# display is supplied by the noninteractive reproduction runner

from py_code.model import H0, PL, dwd, sw

SW_TARGET_LABEL = 115
SW_BURNIN = 0
SW_FIGURE_DIR = FIGURE_DIR
SW_TABLE_DIR = OUTPUT_DIR / "tables"
SW_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SW_TABLE_DIR.mkdir(parents=True, exist_ok=True)

# These are the formal nlive=1000 products that correspond to manuscript Fig. 7.
SW10_FORMAL_FILES = {
    "FD PE": {
        "posterior": Path("data/ch33_sw10_fullxyz_20260531/fd_equal/results/fd_equal_fullxyz_r00_seed2026060255_seg360_fmin1em04_fmax0p15_w0p01_ch33_sw10_fd_equal_fullxyz_20260531_om0m11p5_bundle_formal_nlive1000_full10.h5"),
        "null": Path("data/ch33_sw10_fullxyz_20260531/fd_equal/results/fd_equal_fullxyz_r00_seed2026060255_seg360_fmin1em04_fmax0p15_w0p01_ch33_sw10_fd_equal_fullxyz_20260531_om0m11p5_bundle_bf_nlive1000_no_sw.h5"),
        "color": "red",
    },
    "TD equal PE": {
        "posterior": Path("data/ch33_sw10_fullxyz_20260531/td_equal/results/td_equal_fullxyz_r00_seed2026069815_seg360_fmin1em04_fmax0p15_w0p01_ch33_sw10_td_equal_fullxyz_20260531_om0m11p5_bundle_formal_nlive1000_full10.h5"),
        "null": Path("data/ch33_sw10_fullxyz_20260531/td_equal/results/td_equal_fullxyz_r00_seed2026069815_seg360_fmin1em04_fmax0p15_w0p01_ch33_sw10_td_equal_fullxyz_20260531_om0m11p5_bundle_bf_nlive1000_no_sw.h5"),
        "color": "green",
    },
    "TD unequal PE": {
        "posterior": Path("data/ch33_sw10_fullxyz_20260531/td_unequal/results/td_unequal_fullxyz_r00_seed2026069415_seg360_fmin1em04_fmax0p15_w0p01_ch33_sw10_td_unequal_fullxyz_20260531_om0m11p5_bundle_formal_nlive1000_full10.h5"),
        "null": Path("data/ch33_sw10_fullxyz_20260531/td_unequal/results/td_unequal_fullxyz_r00_seed2026069415_seg360_fmin1em04_fmax0p15_w0p01_ch33_sw10_td_unequal_fullxyz_20260531_om0m11p5_bundle_bf_nlive1000_no_sw.h5"),
        "color": "purple",
    },
}

sw10_param_names = [
    "N_acc", "delta_x", "log10_A1", "alpha1", "log10_A2", "alpha2",
    "log10_Omega_ast", "epsilon", "log10_Omega0", "log10_fp",
]
sw10_param_labels = [
    r"N_{\rm acc}/10^{-15}",
    r"\delta x/10^{-12}",
    r"\log_{10}A_1",
    r"\alpha_1",
    r"\log_{10}A_2",
    r"\alpha_2",
    r"\log_{10}\Omega_{\rm ast}",
    r"\varepsilon",
    r"\log_{10}\Omega_0",
    r"\log_{10}(f_p/{\rm Hz})",
]


def read_formal_sw_product(posterior_path, null_path, burnin=0):
    posterior_path = Path(posterior_path)
    null_path = Path(null_path)
    if not posterior_path.exists():
        raise FileNotFoundError(posterior_path)
    if not null_path.exists():
        raise FileNotFoundError(null_path)

    with h5py.File(resolve_result_path(posterior_path), "r") as h5:
        samples = np.asarray(h5["pe/trace"][int(burnin):, :10], dtype=float)
        truth = np.asarray(h5["parameters_true"][:10], dtype=float)
        posterior_mean = np.asarray(h5["pe/posterior_mean"][:10], dtype=float)
        posterior_std = np.asarray(h5["pe/posterior_std"][:10], dtype=float)
        evidence = float(h5["pe/evidence"][()])

    with h5py.File(resolve_result_path(null_path), "r") as h5:
        null_evidence = float(h5["pe/evidence"][()])

    return {
        "path": posterior_path,
        "null_path": null_path,
        "samples": samples,
        "truth": truth,
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "evidence": evidence,
        "null_evidence": null_evidence,
        "ln_bf": evidence - null_evidence,
    }


sw10_products = {
    label: read_formal_sw_product(info["posterior"], info["null"], burnin=SW_BURNIN)
    for label, info in SW10_FORMAL_FILES.items()
}

sw10_truth = sw10_products["FD PE"]["truth"]
for label, product in sw10_products.items():
    if not np.allclose(product["truth"], sw10_truth, rtol=0.0, atol=1.0e-10):
        raise RuntimeError(f"Truth vector mismatch for {label}")

# Variables retained for the later particle-physics appendix cells.
sw_unequal_trace_index = {str(SW_TARGET_LABEL): 5}
sw_equal_trace_index = {str(SW_TARGET_LABEL): 10}

source_rows = []
for label, info in SW10_FORMAL_FILES.items():
    product = sw10_products[label]
    source_rows.append({
        "configuration": label,
        "posterior_file": product["path"].name,
        "null_file": product["null_path"].name,
        "samples": product["samples"].shape[0],
        "ln_BF": product["ln_bf"],
    })

display(pd.DataFrame(source_rows))

recovery_rows = []
for label, product in sw10_products.items():
    for name, mean, std, truth in zip(sw10_param_names, product["posterior_mean"], product["posterior_std"], product["truth"]):
        recovery_rows.append({
            "configuration": label,
            "parameter": name,
            "truth": truth,
            "posterior_mean": mean,
            "posterior_std": std,
            "relative_uncertainty_percent": 100.0 * std / abs(truth) if truth != 0 else np.nan,
        })

sw10_recovery_df = pd.DataFrame(recovery_rows)
display(sw10_recovery_df)

# Spectrum curves for the upper-right inset of Fig. 7, built from the same formal product.
with h5py.File(resolve_result_path(sw10_products["TD equal PE"]["path"]), "r") as h5:
    sw10_frequency = np.asarray(h5["frequency"][:], dtype=float)
    sw10_response_xx = np.real(np.asarray(h5["tables/response"][:, 0, 0], dtype=complex))
    sw10_acc_transfer_xx = np.real(np.asarray(h5["tables/acc_transfer"][:, 0, 0], dtype=complex))
    sw10_oms_transfer_xx = np.real(np.asarray(h5["tables/oms_transfer"][:, 0, 0], dtype=complex))

n_acc = sw10_truth[0] * 1.0e-15
delta_x = sw10_truth[1] * 1.0e-12
noise_psd_xx = n_acc**2 * sw10_acc_transfer_xx + delta_x**2 * sw10_oms_transfer_xx
strain_to_omega_xx = (3.0 * H0**2 / (4.0 * np.pi**2)) * sw10_response_xx / np.clip(sw10_frequency**3, 1.0e-300, None)
sw10_noise_omega = noise_psd_xx / np.clip(strain_to_omega_xx, 1.0e-300, None)

sw10_spectrum_context = {
    "frequency": sw10_frequency,
    "dwd": dwd(sw10_frequency, sw10_truth[2], sw10_truth[3], sw10_truth[4], sw10_truth[5]),
    "astro": PL(sw10_frequency, sw10_truth[6], sw10_truth[7]),
    "sw": sw(sw10_frequency, sw10_truth[8], sw10_truth[9]),
    "noise": sw10_noise_omega,
}
