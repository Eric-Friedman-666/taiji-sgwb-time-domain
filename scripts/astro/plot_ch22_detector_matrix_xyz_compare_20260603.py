#!/usr/bin/env python3
"""Plot full-XYZ response and transfer matrices for manuscript Sec. 2.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


AXIS_LABEL_FONTSIZE = 17
TICK_LABEL_FONTSIZE = 14
PANEL_TITLE_FONTSIZE = 13
LEGEND_FONTSIZE = 14
SCATTER_AXIS_LABEL_FONTSIZE = 21
SCATTER_LEGEND_FONTSIZE = 17
NOTCH_COLOR = "0.82"
NOTCH_ALPHA = 0.45
PHASE_YLIM_PAD = 1.05


MATRIX_SPECS = {
    "acc_transfer": {
        "short": "acc",
        "title": r"ACC noise transfer matrix $\mathcal{N}^{\rm acc}_{IJ}$",
        "ylabel": r"$|\mathcal{N}^{\rm acc}_{IJ}|$",
        "phase_ylabel": r"$\arg(\mathcal{N}^{\rm acc}_{IJ})$ [rad]",
        "overlay_name": "ch22_fd_td_acc_transfer_overlay_sim_grid_20260605.png",
        "ratio_name": "ch22_td_unequal_acc_transfer_sim_ratio_grid_20260605.png",
        "validation_exp": "acc",
        "validation_dataset": "acc_transfer",
    },
    "oms_transfer": {
        "short": "oms",
        "title": r"OMS noise transfer matrix $\mathcal{N}^{\rm OMS}_{IJ}$",
        "ylabel": r"$|\mathcal{N}^{\rm OMS}_{IJ}|$",
        "phase_ylabel": r"$\arg(\mathcal{N}^{\rm OMS}_{IJ})$ [rad]",
        "overlay_name": "ch22_fd_td_oms_transfer_overlay_sim_grid_20260605.png",
        "ratio_name": "ch22_td_unequal_oms_transfer_sim_ratio_grid_20260605.png",
        "validation_exp": "oms",
        "validation_dataset": "oms_transfer",
    },
    "response": {
        "short": "response",
        "title": r"Response matrix $\mathcal{R}_{IJ}$",
        "ylabel": r"$|\mathcal{R}_{IJ}|$",
        "phase_ylabel": r"$\arg(\mathcal{R}_{IJ})$ [rad]",
        "overlay_name": "ch22_fd_td_response_overlay_sim_grid_20260605.png",
        "ratio_name": "ch22_td_unequal_response_sim_ratio_grid_20260605.png",
        "validation_exp": "response",
        "validation_dataset": "response",
    },
}

PANEL_SPECS = (
    ("XX", 0, 0, "amplitude"),
    ("YY", 1, 1, "amplitude"),
    ("ZZ", 2, 2, "amplitude"),
    ("XY", 0, 1, "amplitude"),
    ("XZ", 0, 2, "amplitude"),
    ("YZ", 1, 2, "amplitude"),
    ("XY", 0, 1, "phase"),
    ("XZ", 0, 2, "phase"),
    ("YZ", 1, 2, "phase"),
)

OFFDIAG_SPECS = (
    ("XY", 0, 1),
    ("XZ", 0, 2),
    ("YZ", 1, 2),
)

DIAG_SPECS = (
    ("XX", 0, 0),
    ("YY", 1, 1),
    ("ZZ", 2, 2),
)


def load_table(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        out = {
            "frequency": h5["frequency"][()],
            "response": h5["response"][()],
            "acc_transfer": h5["acc_transfer"][()],
            "oms_transfer": h5["oms_transfer"][()],
        }
        if "notch_centers" in h5:
            centers = h5["notch_centers"][()]
            out["notch_centers"] = np.asarray([row["center_hz"] for row in centers], dtype=float)
        else:
            out["notch_centers"] = np.asarray([], dtype=float)
    return out


def load_fd_static_fullfreq(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        return {
            "frequency": h5["frequency"][()],
            "response": h5["response_single"][()],
            "acc_transfer": h5["acc_transfer_single"][()],
            "oms_transfer": h5["oms_transfer_single"][()],
            "notch_centers": np.asarray([], dtype=float),
        }


def load_td_fullfreq(path: Path, group_name: str) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        group = h5[group_name]
        return {
            "frequency": group["frequency"][()],
            "response": group["response"][()],
            "acc_transfer": group["acc_transfer"][()],
            "oms_transfer": group["oms_transfer"][()],
            "notch_centers": np.asarray([], dtype=float),
        }


def load_notch_centers(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if "notch_centers" not in h5:
            return np.asarray([], dtype=float)
        rows = h5["notch_centers"][()]
    return np.asarray([float(row["center_hz"]) for row in rows], dtype=float)


def finite_positive(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    return np.where(np.isfinite(out) & (out > 0.0), out, np.nan)


def wrap_phase_residual(values: np.ndarray) -> np.ndarray:
    return (np.angle(values) + np.pi) % (2.0 * np.pi) - np.pi


def circular_mean_phase(values: np.ndarray, axis: int = 0) -> np.ndarray:
    phase = np.angle(values)
    return np.angle(np.nanmean(np.exp(1j * phase), axis=axis))


def interpolate_complex(x_new: np.ndarray, x_old: np.ndarray, y_old: np.ndarray) -> np.ndarray:
    """Linear complex interpolation for plotting against a matching reference grid."""
    if x_new.shape == x_old.shape and np.allclose(x_new, x_old):
        return y_old
    real = np.interp(x_new, x_old, np.real(y_old), left=np.nan, right=np.nan)
    imag = np.interp(x_new, x_old, np.imag(y_old), left=np.nan, right=np.nan)
    return real + 1j * imag


def td_amplitude_stats(matrix: np.ndarray, i: int, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amp = np.abs(matrix[:, :, i, j])
    return (
        np.nanmean(amp, axis=0),
        np.nanpercentile(amp, 5.0, axis=0),
        np.nanpercentile(amp, 95.0, axis=0),
    )


def add_notch_bands(ax: plt.Axes, centers: np.ndarray, half_width: float) -> None:
    for center in centers:
        ax.axvspan(center - half_width, center + half_width, color=NOTCH_COLOR, alpha=NOTCH_ALPHA, lw=0, zorder=0)


def notch_legend_handle() -> Patch:
    return Patch(facecolor=NOTCH_COLOR, alpha=NOTCH_ALPHA, edgecolor="none", label="Notch")


def append_notch_legend(handles: list[object], labels: list[str], centers: np.ndarray) -> tuple[list[object], list[str]]:
    if np.asarray(centers).size and "Notch" not in labels:
        handles = [*handles, notch_legend_handle()]
        labels = [*labels, "Notch"]
    return handles, labels


def darker_sim_marker(label: str, color: str) -> Line2D:
    return Line2D(
        [],
        [],
        color=color,
        marker="o",
        linestyle="None",
        markersize=5.0,
        markeredgewidth=0.0,
        alpha=0.90,
        label=label,
    )


def set_readable_ylabel(
    ax: plt.Axes,
    label: str,
    *,
    fontsize: float = AXIS_LABEL_FONTSIZE,
    labelpad: float = 38.0,
) -> None:
    ax.set_ylabel(
        label,
        fontsize=fontsize,
        rotation=0,
        labelpad=labelpad,
        ha="right",
        va="center",
    )


def style_axis_text(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="minor", labelsize=max(TICK_LABEL_FONTSIZE - 1, 8))


def setup_axis(
    ax: plt.Axes,
    freq: np.ndarray,
    mask: np.ndarray,
    is_phase: bool,
    *,
    phase_range: str = "zero_to_twopi",
) -> None:
    ax.set_xscale("log")
    ax.set_xlim(float(np.nanmin(freq[mask])), float(np.nanmax(freq[mask])))
    ax.grid(True, which="both", alpha=0.24, lw=0.45)
    style_axis_text(ax)
    if is_phase and phase_range == "zero_to_twopi":
        ax.set_ylim(0.0, 2.0 * np.pi)
        ax.set_yticks([0.0, np.pi, 2.0 * np.pi])
        ax.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
        style_axis_text(ax)


def set_residual_axis(ax: plt.Axes, values: list[np.ndarray]) -> None:
    finite_values = []
    for value in values:
        arr = np.asarray(value, dtype=float)
        finite_values.append(arr[np.isfinite(arr)])
    if finite_values:
        merged = np.concatenate([arr for arr in finite_values if arr.size])
    else:
        merged = np.asarray([], dtype=float)
    if merged.size:
        lim = float(np.nanpercentile(np.abs(merged), 99.0) * 1.15)
    else:
        lim = 0.2
    lim = min(max(lim, 0.12), np.pi)
    ax.set_ylim(-lim, lim)
    if lim > 1.0:
        ax.set_yticks([-np.pi, 0.0, np.pi])
        ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    else:
        tick = 0.2 if lim <= 0.35 else 0.5
        ax.set_yticks([-tick, 0.0, tick])


def thin_plot_mask(mask: np.ndarray, *, target_points: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size <= target_points:
        return mask
    step = int(np.ceil(idx.size / target_points))
    out = np.zeros(mask.shape, dtype=bool)
    out[idx[::step]] = True
    return out


def plot_fd_td_grid(
    matrix_name: str,
    fd: dict[str, np.ndarray],
    td_equal: dict[str, np.ndarray],
    td_unequal: dict[str, np.ndarray],
    validation: dict[str, object] | None,
    output: Path,
    *,
    f_min: float,
    f_max: float,
    notch_half_width: float,
    dpi: int,
) -> dict[str, object]:
    spec = MATRIX_SPECS[matrix_name]
    freq = fd["frequency"]
    mask = np.isfinite(freq) & (freq >= f_min) & (freq <= f_max)
    fd_matrix = fd[matrix_name]
    td_eq_matrix = td_equal[matrix_name]
    td_uneq_matrix = td_unequal[matrix_name]
    sim_freq = None if validation is None else np.asarray(validation["frequency"], dtype=float)
    sim_matrix = None if validation is None else np.asarray(validation["sim"], dtype=np.complex128)
    sim_mask = None
    if sim_freq is not None:
        sim_mask = np.isfinite(sim_freq) & (sim_freq >= f_min) & (sim_freq <= f_max)
        sim_point_mask = thin_plot_mask(sim_mask, target_points=2200)
    else:
        sim_point_mask = None

    fig, axes = plt.subplots(3, 3, figsize=(13.8, 11.2), sharex=True)
    fig.subplots_adjust(left=0.145, right=0.992, bottom=0.090, top=0.875, wspace=0.25, hspace=0.34)

    colors = {
        "fd": "#202020",
        "td_equal": "#1f77b4",
        "td_unequal": "#7b2cbf",
        "sim": "#E69F00",
    }
    n_real = None if validation is None else int(validation["n_real"])
    sim_label = None if n_real is None else f"Average of {n_real} realizations"
    summary: dict[str, object] = {"output": str(output), "matrix": matrix_name, "n_real_overlay": n_real, "panels": {}}
    summary["notch_half_width_hz"] = float(notch_half_width)
    summary["notch_center_count"] = int(np.asarray(fd.get("notch_centers", np.asarray([]))).size)
    summary["notch_center_source"] = "fd_equal_reference"

    for panel_idx, (ax, (entry, i, j, mode)) in enumerate(zip(axes.ravel(), PANEL_SPECS)):
        col_idx = panel_idx % axes.shape[1]
        is_phase = mode == "phase"
        add_notch_bands(ax, fd.get("notch_centers", np.asarray([])), notch_half_width)

        if is_phase:
            fd_y = np.angle(fd_matrix[:, i, j])
            td_eq_y = circular_mean_phase(td_eq_matrix[:, :, i, j], axis=0)
            td_uneq_y = circular_mean_phase(td_uneq_matrix[:, :, i, j], axis=0)
            ax.semilogx(freq[mask], fd_y[mask], color=colors["fd"], lw=1.55, label="Static equal")
            ax.semilogx(freq[mask], td_eq_y[mask], color=colors["td_equal"], lw=1.35, label="TD equal")
            ax.semilogx(freq[mask], td_uneq_y[mask], color=colors["td_unequal"], lw=1.35, label="TD unequal")
            if sim_matrix is not None and sim_freq is not None and sim_mask is not None:
                sim_y = np.angle(sim_matrix[:, i, j])
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
            if col_idx == 0:
                set_readable_ylabel(ax, spec["phase_ylabel"], labelpad=30.0)
            panel_key = f"arg {entry}"
            ax.set_ylim(-PHASE_YLIM_PAD * np.pi, PHASE_YLIM_PAD * np.pi)
            ax.set_yticks([-np.pi, 0.0, np.pi])
            ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
            style_axis_text(ax)
        else:
            fd_y = finite_positive(np.abs(fd_matrix[:, i, j]))
            td_eq_mean, td_eq_lo, td_eq_hi = td_amplitude_stats(td_eq_matrix, i, j)
            td_uneq_mean, td_uneq_lo, td_uneq_hi = td_amplitude_stats(td_uneq_matrix, i, j)
            ax.loglog(freq[mask], fd_y[mask], color=colors["fd"], lw=1.55, label="Static equal")
            ax.loglog(freq[mask], finite_positive(td_eq_mean)[mask], color=colors["td_equal"], lw=1.35, label="TD equal")
            ax.fill_between(
                freq[mask],
                finite_positive(td_eq_lo)[mask],
                finite_positive(td_eq_hi)[mask],
                color=colors["td_equal"],
                alpha=0.10,
                lw=0,
            )
            ax.loglog(freq[mask], finite_positive(td_uneq_mean)[mask], color=colors["td_unequal"], lw=1.35, label="TD unequal")
            ax.fill_between(
                freq[mask],
                finite_positive(td_uneq_lo)[mask],
                finite_positive(td_uneq_hi)[mask],
                color=colors["td_unequal"],
                alpha=0.12,
                lw=0,
            )
            if sim_matrix is not None and sim_freq is not None and sim_mask is not None:
                sim_y = finite_positive(np.abs(sim_matrix[:, i, j]))
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
            if col_idx == 0:
                set_readable_ylabel(ax, spec["ylabel"], labelpad=30.0)
            panel_key = f"|{entry}|"
            summary["panels"][panel_key] = {
                "td_equal_over_fd_median": float(np.nanmedian((td_eq_mean / np.abs(fd_matrix[:, i, j]))[mask])),
                "td_unequal_over_fd_median": float(np.nanmedian((td_uneq_mean / np.abs(fd_matrix[:, i, j]))[mask])),
            }

        ax.set_title(panel_key, fontsize=PANEL_TITLE_FONTSIZE)
        setup_axis(ax, freq, mask, is_phase, phase_range="none" if is_phase else "zero_to_twopi")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$f\,[{\rm Hz}]$", fontsize=AXIS_LABEL_FONTSIZE, labelpad=5.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if sim_label is not None:
        handles = [darker_sim_marker(sim_label, colors["sim"]) if label == sim_label else handle for handle, label in zip(handles, labels)]
    handles, labels = append_notch_legend(handles, labels, fd.get("notch_centers", np.asarray([])))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.975),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.1,
        columnspacing=1.5,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return summary


def load_validation_data(
    matrix_name: str,
    validation_h5: Path,
    *,
    day_label: str,
    notch_half_width: float | None,
) -> dict[str, object]:
    spec = MATRIX_SPECS[matrix_name]
    exp = spec["validation_exp"]
    dataset = spec["validation_dataset"]
    with h5py.File(validation_h5, "r") as h5:
        day_idx = validation_day_index(h5, day_label)
        sim = h5[f"{exp}/{day_label}/transfer_sim"][()]
        source_notch_half_width = None
        if "notch_centers" in h5 and len(h5["notch_centers"]):
            source_notch_half_width = float(h5["notch_centers"][0]["half_width_hz"])
        if notch_half_width is None:
            notch_half_width = 0.0 if source_notch_half_width is None else source_notch_half_width
        return {
            "frequency": h5["frequency"][()],
            "calc": h5[f"calc/{dataset}"][day_idx],
            "sim": sim,
            "n_real": int(h5.attrs.get("N_real", sim.shape[0] if sim.ndim == 5 else 1000)),
            "centers": validation_notch_centers(h5, day_label),
            "notch_half_width": float(notch_half_width),
            "source_notch_half_width": source_notch_half_width,
            "validation_h5": str(validation_h5),
            "day_label": day_label,
        }


def validation_day_index(h5: h5py.File, label: str) -> int:
    labels = [x.decode("ascii") for x in h5["calc/day_label"][()]]
    return labels.index(label)


def validation_plot_mask(h5: h5py.File, f_min: float, f_max: float) -> np.ndarray:
    freq = h5["frequency"][()]
    return np.isfinite(freq) & (freq >= f_min) & (freq <= f_max)


def validation_notch_centers(h5: h5py.File, day_label: str) -> np.ndarray:
    if "notch_centers" not in h5:
        return np.asarray([], dtype=float)
    rows = h5["notch_centers"][()]
    centers = []
    for row in rows:
        day = row["day"].decode("ascii") if hasattr(row["day"], "decode") else str(row["day"])
        if day == day_label:
            centers.append(float(row["center_hz"]))
    return np.asarray(centers, dtype=float)


def plot_validation_ratio_grid(
    matrix_name: str,
    validation: dict[str, object],
    output: Path,
    *,
    f_min: float,
    f_max: float,
    dpi: int,
) -> dict[str, object]:
    spec = MATRIX_SPECS[matrix_name]
    freq = np.asarray(validation["frequency"], dtype=float)
    mask = np.isfinite(freq) & (freq >= f_min) & (freq <= f_max)
    calc = np.asarray(validation["calc"], dtype=np.complex128)
    sim = np.asarray(validation["sim"], dtype=np.complex128)
    n_real = int(validation["n_real"])
    centers = np.asarray(validation["centers"], dtype=float)
    notch_half_width = float(validation["notch_half_width"])
    point_mask = thin_plot_mask(mask, target_points=3600)

    fig, axes = plt.subplots(3, 3, figsize=(13.8, 11.2), sharex=True)
    fig.subplots_adjust(left=0.145, right=0.992, bottom=0.085, top=0.895, wspace=0.25, hspace=0.34)
    colors = {"ratio": "#E69F00", "reference": "#00897B"}
    summary: dict[str, object] = {
        "output": str(output),
        "matrix": matrix_name,
        "validation_h5": validation["validation_h5"],
        "day_label": validation["day_label"],
        "n_real": n_real,
        "notch_half_width_hz": notch_half_width,
        "notch_source_half_width_hz": validation.get("source_notch_half_width"),
        "notch_center_count": int(centers.size),
        "panels": {},
    }

    for panel_idx, (ax, (entry, i, j, mode)) in enumerate(zip(axes.ravel(), PANEL_SPECS)):
        col_idx = panel_idx % axes.shape[1]
        is_phase = mode == "phase"
        add_notch_bands(ax, centers, notch_half_width)

        if is_phase:
            phase_resid = wrap_phase_residual(sim[:, i, j] / calc[:, i, j])
            ax.axhline(0.0, color=colors["reference"], ls="--", lw=1.0, label="Ideal")
            ax.scatter(freq[point_mask], phase_resid[point_mask], s=5.0, color=colors["ratio"], alpha=0.56, linewidths=0, label=rf"Simulation/model, $N_{{\rm real}}={n_real}$")
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
            ax.set_yticklabels([r"$-0.5$", r"$-0.25$", r"$0$", r"$0.25$", r"$0.5$"])
            if col_idx == 0:
                set_readable_ylabel(ax, r"$\arg\!\left(\frac{\widehat G}{P}\right)\,[{\rm rad}]$", labelpad=42.0)
            style_axis_text(ax)
            panel_key = f"arg {entry}"
            summary["panels"][panel_key] = {
                "phase_residual_rad_p5_p50_p95": [
                    float(x) for x in np.nanpercentile(phase_resid[mask], [5.0, 50.0, 95.0])
                ],
            }
        else:
            calc_y = finite_positive(np.abs(calc[:, i, j]))
            sim_y = finite_positive(np.abs(sim[:, i, j]))
            ratio = sim_y / calc_y
            ax.axhline(1.0, color=colors["reference"], ls="--", lw=1.0, label="Ideal")
            ax.scatter(freq[point_mask], ratio[point_mask], s=5.0, color=colors["ratio"], alpha=0.56, linewidths=0, label=rf"Simulation/model, $N_{{\rm real}}={n_real}$")
            ax.set_ylim(0.9, 1.1)
            if col_idx == 0:
                set_readable_ylabel(ax, r"$|\frac{\widehat G}{P}|$", labelpad=42.0)
            panel_key = f"|{entry}|"
            summary["panels"][panel_key] = {
                "sim_over_calc_p5_p50_p95": [
                    float(x) for x in np.nanpercentile(ratio[mask], [5.0, 50.0, 95.0])
                ],
            }

        ax.set_title(panel_key, fontsize=PANEL_TITLE_FONTSIZE)
        setup_axis(ax, freq, mask, is_phase, phase_range="none")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$f\,[{\rm Hz}]$", fontsize=AXIS_LABEL_FONTSIZE)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = {
        label: darker_sim_marker(label, colors["ratio"]) if label.startswith("Simulation/model") else handle
        for handle, label in zip(handles, labels)
    }
    if centers.size:
        unique["Notch"] = notch_legend_handle()
    sim_legend_label = rf"Simulation/model, $N_{{\rm real}}={n_real}$"
    legend_order = [sim_legend_label, "Ideal", "Notch"]
    fig.legend(
        [unique[label] for label in legend_order if label in unique],
        [label for label in legend_order if label in unique],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.975),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.0,
        columnspacing=1.6,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return summary


def plot_relative_uncertainty_diagnostic(
    validations: dict[str, dict[str, object]],
    fd_static: dict[str, np.ndarray],
    output: Path,
    *,
    panel_specs: tuple[tuple[str, int, int], ...],
    title: str,
    ylabel: str,
    f_min: float,
    f_max: float,
    dpi: int,
) -> dict[str, object]:
    fig, axes = plt.subplots(3, 3, figsize=(13.8, 10.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.130, right=0.992, bottom=0.095, top=0.900, wspace=0.16, hspace=0.32)
    colors = {
        "td_unequal": "#D18A00",
        "fd_static": "#0072B2",
        "td_theory": "#009E73",
        "fd_theory": "#D62728",
    }
    summary: dict[str, object] = {"output": str(output), "panels": {}}
    residuals_for_ylim: list[np.ndarray] = []
    has_notch = False

    for row, matrix_name in enumerate(MATRIX_SPECS):
        spec = MATRIX_SPECS[matrix_name]
        validation = validations[matrix_name]
        freq = np.asarray(validation["frequency"], dtype=float)
        calc = np.asarray(validation["calc"], dtype=np.complex128)
        sim = np.asarray(validation["sim"], dtype=np.complex128)
        fd_freq = np.asarray(fd_static["frequency"], dtype=float)
        fd_raw = np.asarray(fd_static[matrix_name], dtype=np.complex128)
        if fd_freq.shape == freq.shape and np.allclose(fd_freq, freq):
            fd_calc = fd_raw
        else:
            fd_calc = np.empty_like(calc)
            for a in range(calc.shape[1]):
                for b in range(calc.shape[2]):
                    fd_calc[:, a, b] = interpolate_complex(freq, fd_freq, fd_raw[:, a, b])
        centers = np.asarray(validation["centers"], dtype=float)
        notch_half_width = float(validation["notch_half_width"])
        has_notch = has_notch or bool(centers.size)
        mask = np.isfinite(freq) & (freq >= f_min) & (freq <= f_max)
        point_mask = thin_plot_mask(mask, target_points=2600)

        for col, (entry, i, j) in enumerate(panel_specs):
            ax = axes[row, col]
            add_notch_bands(ax, centers, notch_half_width)
            residual_by_reference: dict[str, np.ndarray] = {}
            for reference_name, reference_matrix in (("td_unequal", calc), ("fd_static", fd_calc)):
                residual_by_reference[reference_name] = np.abs(sim[:, i, j] - reference_matrix[:, i, j]) / finite_positive(
                    np.abs(reference_matrix[:, i, j])
                )
            squared_residual_by_reference = {
                reference_name: residual**2 for reference_name, residual in residual_by_reference.items()
            }
            n_real = int(validation["n_real"])
            td_scatter_numerator = np.sqrt(
                finite_positive(np.abs(calc[:, i, i])) * finite_positive(np.abs(calc[:, j, j]))
            ) / np.sqrt(float(n_real))
            rms_by_reference = {
                "td_unequal": td_scatter_numerator / finite_positive(np.abs(calc[:, i, j])),
                "fd_static": td_scatter_numerator / finite_positive(np.abs(fd_calc[:, i, j])),
            }
            squared_expectation_by_reference = {
                reference_name: rms**2 for reference_name, rms in rms_by_reference.items()
            }

            ax.scatter(
                freq[point_mask],
                squared_residual_by_reference["fd_static"][point_mask],
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
                squared_residual_by_reference["td_unequal"][point_mask],
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
                squared_expectation_by_reference["td_unequal"][mask],
                color=colors["td_theory"],
                lw=2.15,
                alpha=1.0,
                label="TD expected level",
                zorder=30,
            )
            ax.plot(
                freq[mask],
                squared_expectation_by_reference["fd_static"][mask],
                color=colors["fd_theory"],
                lw=1.55,
                ls=(0, (5.0, 2.4)),
                alpha=1.0,
                dash_capstyle="round",
                label="FD expected level",
                zorder=80,
            )
            ax.set_title(f"{spec['short'].upper()} {entry}", fontsize=PANEL_TITLE_FONTSIZE)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(float(np.nanmin(freq[mask])), float(np.nanmax(freq[mask])))
            ax.grid(True, which="both", alpha=0.24, lw=0.45)
            style_axis_text(ax)
            if col == 0:
                set_readable_ylabel(ax, ylabel, fontsize=SCATTER_AXIS_LABEL_FONTSIZE, labelpad=34.0)
            if row == 2:
                ax.set_xlabel(r"$f\,[{\rm Hz}]$", fontsize=SCATTER_AXIS_LABEL_FONTSIZE)

            finite_td_residual = squared_residual_by_reference["td_unequal"][mask]
            finite_td_residual = finite_td_residual[np.isfinite(finite_td_residual) & (finite_td_residual > 0.0)]
            finite_fd_residual = squared_residual_by_reference["fd_static"][mask]
            finite_fd_residual = finite_fd_residual[np.isfinite(finite_fd_residual) & (finite_fd_residual > 0.0)]
            if finite_td_residual.size:
                residuals_for_ylim.append(finite_td_residual)
            if finite_fd_residual.size:
                residuals_for_ylim.append(finite_fd_residual)
            finite_td_rms = squared_expectation_by_reference["td_unequal"][mask]
            finite_td_rms = finite_td_rms[np.isfinite(finite_td_rms) & (finite_td_rms > 0.0)]
            finite_fd_rms = squared_expectation_by_reference["fd_static"][mask]
            finite_fd_rms = finite_fd_rms[np.isfinite(finite_fd_rms) & (finite_fd_rms > 0.0)]
            if finite_td_rms.size:
                residuals_for_ylim.append(finite_td_rms)
            if finite_fd_rms.size:
                residuals_for_ylim.append(finite_fd_rms)
            panel_key = f"{matrix_name} {entry}"
            summary["panels"][panel_key] = {
                "td_unequal_squared_residual_p5_p50_p95": [float(x) for x in np.nanpercentile(finite_td_residual, [5.0, 50.0, 95.0])] if finite_td_residual.size else [],
                "fd_static_squared_residual_p5_p50_p95": [float(x) for x in np.nanpercentile(finite_fd_residual, [5.0, 50.0, 95.0])] if finite_fd_residual.size else [],
                "td_unequal_expected_squared_residual_p5_p50_p95": [float(x) for x in np.nanpercentile(finite_td_rms, [5.0, 50.0, 95.0])] if finite_td_rms.size else [],
                "fd_static_expected_squared_residual_p5_p50_p95": [float(x) for x in np.nanpercentile(finite_fd_rms, [5.0, 50.0, 95.0])] if finite_fd_rms.size else [],
            }

    if residuals_for_ylim:
        merged_residuals = np.concatenate(residuals_for_ylim)
        y_lo = float(np.nanpercentile(merged_residuals, 1.0) * 0.75)
        y_hi = float(np.nanpercentile(merged_residuals, 99.0) * 1.20)
    else:
        y_lo, y_hi = 2.0e-3, 3.0
    y_lo = min(max(y_lo, 1.0e-5), 2.0e-2)
    y_hi = min(max(y_hi, 0.3), 50.0)
    for ax in axes.flat:
        ax.set_ylim(y_lo, y_hi)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if has_notch:
        unique["Notch"] = notch_legend_handle()
    legend_order = ["TD residuals", "TD expected level", "FD residuals", "FD expected level", "Notch"]
    fig.legend(
        [unique[label] for label in legend_order if label in unique],
        [label for label in legend_order if label in unique],
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.975),
        frameon=False,
        fontsize=SCATTER_LEGEND_FONTSIZE,
        handlelength=2.2,
        columnspacing=1.6,
        markerscale=2.8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return summary


def plot(args: argparse.Namespace) -> None:
    fd = load_fd_static_fullfreq(args.fd_static_fullfreq)
    td_equal = load_td_fullfreq(args.td_fullfreq, "equal")
    td_unequal = load_td_fullfreq(args.td_fullfreq, "unequal")
    fd["notch_centers"] = load_notch_centers(args.notch_reference)

    freq = fd["frequency"]
    for name, table in [("TD equal", td_equal), ("TD unequal", td_unequal)]:
        if not np.allclose(freq, table["frequency"]):
            raise ValueError(f"{name} frequency grid does not match FD equal table")

    summary: dict[str, object] = {
        "fd_static_fullfreq": str(args.fd_static_fullfreq),
        "td_fullfreq": str(args.td_fullfreq),
        "notch_reference": str(args.notch_reference),
        "validation_h5": str(args.validation_h5),
        "frequency_bins_fdtd": int(freq.size),
        "fdtd_plot_f_min": args.fdtd_f_min,
        "fdtd_plot_f_max": args.fdtd_f_max,
        "notch_points_are_plotted": True,
        "outputs": {},
    }

    validations: dict[str, dict[str, object]] = {}
    for matrix_name, spec in MATRIX_SPECS.items():
        validation = load_validation_data(
            matrix_name,
            args.validation_h5,
            day_label=args.validation_day,
            notch_half_width=args.validation_notch_half_width,
        )
        validations[matrix_name] = validation
        overlay_output = args.output_dir / spec["overlay_name"]
        ratio_output = args.output_dir / spec["ratio_name"]
        summary["outputs"][matrix_name] = {
            "overlay": plot_fd_td_grid(
                matrix_name,
                fd,
                td_equal,
                td_unequal,
                validation,
                overlay_output,
                f_min=args.fdtd_f_min,
                f_max=args.fdtd_f_max,
                notch_half_width=args.notch_half_width,
                dpi=args.dpi,
            ),
            "ratio": plot_validation_ratio_grid(
                matrix_name,
                validation,
                ratio_output,
                f_min=args.validation_f_min,
                f_max=args.validation_f_max,
                dpi=args.dpi,
            ),
        }

    diagonal_output = args.output_dir / args.diagonal_scatter_name
    summary["outputs"]["diagonal_relative_uncertainty"] = plot_relative_uncertainty_diagnostic(
        validations,
        fd,
        diagonal_output,
        panel_specs=DIAG_SPECS,
        title=r"Diagonal auto spectrum squared normalized residual $[r_{II}^{\kappa}]^2$",
        ylabel=r"$\left(r_{II}^{\kappa}\right)^2$",
        f_min=args.validation_f_min,
        f_max=args.validation_f_max,
        dpi=args.dpi,
    )

    coherence_output = args.output_dir / args.coherence_scatter_name
    summary["outputs"]["cross_coherence_scatter"] = plot_relative_uncertainty_diagnostic(
        validations,
        fd,
        coherence_output,
        panel_specs=OFFDIAG_SPECS,
        title=r"Off diagonal cross spectrum squared normalized residual $[r_{IJ}^{\kappa}]^2$",
        ylabel=r"$\left(r_{IJ}^{\kappa}\right)^2$",
        f_min=args.validation_f_min,
        f_max=args.validation_f_max,
        dpi=args.dpi,
    )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for matrix_info in summary["outputs"].values():
        if "overlay" in matrix_info:
            print(matrix_info["overlay"]["output"], flush=True)
            print(matrix_info["ratio"]["output"], flush=True)
        else:
            print(matrix_info["output"], flush=True)
    print(args.summary, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--fd-static-fullfreq",
        type=Path,
        default=root / "data/ch32_astro_formal_20260520/precomputed_tables/ch32_response_noise_fullfreq_20260521/chunks/equal_static_fullfreq.h5",
    )
    parser.add_argument(
        "--td-fullfreq",
        type=Path,
        default=root / "data/ch32_astro_formal_20260520/precomputed_tables/ch32_response_noise_fullfreq_20260521/ch32_response_noise_fullfreq_20260521_equal_unequal_fullfreq_seg360.h5",
    )
    parser.add_argument(
        "--notch-reference",
        type=Path,
        default=root / "data/ch32_astro_formal_20260520/fd_equal/tables/fd_equal_fullxyz_table_fmin1em04_fmax0p15_w0p01_ch32_fd_equal_fullxyz_20260520.h5",
    )
    parser.add_argument(
        "--validation-h5",
        type=Path,
        default=root / "data/ast4_cov_validation_20260510/ast4_covariance_validation_nreal1000_day01sim_20260511.h5",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "figure/ch22_detector_matrix_grids_20260604")
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "figure/ch22_detector_matrix_grids_20260604/ch22_detector_matrix_grids_20260604_summary.json",
    )
    parser.add_argument("--fdtd-f-min", type=float, default=1.0e-4)
    parser.add_argument("--fdtd-f-max", type=float, default=0.15)
    parser.add_argument("--notch-half-width", type=float, default=0.010)
    parser.add_argument("--validation-day", default="day01")
    parser.add_argument("--validation-f-min", type=float, default=1.0e-4)
    parser.add_argument("--validation-f-max", type=float, default=0.15)
    parser.add_argument("--validation-notch-half-width", type=float, default=0.010)
    parser.add_argument("--diagonal-scatter-name", default="ch22_diagonal_relative_uncertainty_diagnostic_20260605.png")
    parser.add_argument("--coherence-scatter-name", default="ch22_cross_coherence_scatter_diagnostic_20260605.png")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


if __name__ == "__main__":
    plot(parse_args())
