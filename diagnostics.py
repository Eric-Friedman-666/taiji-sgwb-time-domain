"""Full-XYZ Fisher and SNR calculations on the published retained bins."""

from __future__ import annotations

import numpy as np

from py_code.parameters import H0


def components(frequency, theta):
    f = np.asarray(frequency, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    x = f / 0.016
    turnover = 10.0 ** theta[4] * x ** theta[5]
    dwd = 10.0 ** theta[2] * x ** theta[3] / (1.0 + turnover)
    ast = 10.0 ** theta[6] * (f / 1.0e-3) ** theta[7]
    sw = np.zeros_like(f)
    if theta.size == 10:
        ratio = f / 10.0 ** theta[9]
        sw = 10.0 ** theta[8] * ratio**3 * (7.0 / (4.0 + 3.0 * ratio**2)) ** 3.5
    return dwd, ast, sw


def covariance_and_derivatives(frequency, response, acc, oms, theta):
    f = np.asarray(frequency, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    response, acc, oms = [np.asarray(x, dtype=np.complex128) for x in (response, acc, oms)]
    dwd, ast, sw = components(f, theta)
    conversion = 3.0 * H0**2 / (4.0 * np.pi**2 * f**3)

    def signal(omega):
        return conversion[..., None, None] * omega[..., None, None] * response

    noise = (theta[0] * 1.0e-15) ** 2 * acc + (theta[1] * 1.0e-12) ** 2 * oms
    total = noise + signal(dwd + ast + sw)
    turnover = 10.0 ** theta[4] * (f / 0.016) ** theta[5]
    ln10 = np.log(10.0)
    derivatives = [
        2.0 * theta[0] * 1.0e-30 * acc,
        2.0 * theta[1] * 1.0e-24 * oms,
        signal(ln10 * dwd),
        signal(np.log(f / 0.016) * dwd),
        signal(-ln10 * turnover / (1.0 + turnover) * dwd),
        signal(-np.log(f / 0.016) * turnover / (1.0 + turnover) * dwd),
        signal(ln10 * ast),
        signal(np.log(f / 1.0e-3) * ast),
    ]
    if theta.size == 10:
        ratio = f / 10.0 ** theta[9]
        derivatives += [
            signal(ln10 * sw),
            signal(ln10 * sw * (-3.0 + 21.0 * ratio**2 / (4.0 + 3.0 * ratio**2))),
        ]
    return total, derivatives, noise, signal(dwd), signal(ast), signal(sw)


def trace_square(base, target):
    solved = np.linalg.solve(base, target)
    return float(np.sum(np.real(np.einsum("...ij,...ji->...", solved, solved))))


def calculate(frequency, response, acc, oms, theta, segments=360):
    total, derivatives, noise, dwd, ast, sw = covariance_and_derivatives(
        frequency, response, acc, oms, theta
    )
    # A three-dimensional table is static and reused in each segment. A
    # four-dimensional table already includes the segment axis in the sum.
    weight = int(segments) if np.ndim(response) == 3 else 1
    solved = [np.linalg.solve(total, derivative) for derivative in derivatives]
    fisher = np.asarray([
        [weight * np.sum(np.real(np.einsum("...ij,...ji->...", a, b))) for b in solved]
        for a in solved
    ], dtype=np.float64)
    fisher = (fisher + fisher.T) * 0.5
    scale = 1.0 / np.sqrt(np.diag(fisher))
    scaled = fisher * scale[:, None] * scale[None, :]
    covariance = np.linalg.inv(scaled) * scale[:, None] * scale[None, :]
    std = np.sqrt(np.diag(covariance))
    target = sw if len(theta) == 10 else ast
    baseline = noise + dwd + (ast if len(theta) == 10 else 0.0)
    return {
        "fisher_matrix": fisher.tolist(),
        "covariance": covariance.tolist(),
        "std": std.tolist(),
        "relative_uncertainty_percent": (100.0 * std / np.abs(theta)).tolist(),
        "snr_abs": float(np.sqrt(weight * trace_square(noise, target))),
        "snr_rel": float(np.sqrt(weight * trace_square(baseline, target))),
        "frequency_bins": int(len(frequency)),
        "segments": int(segments),
        "snr_definition": "sum Tr(P_base^-1 P_target P_base^-1 P_target), no extra factor of two",
    }
