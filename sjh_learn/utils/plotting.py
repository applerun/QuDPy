"""绘图与结果保存工具。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .optical_bloch import OpticalBlochResult


def save_comparison_plot(result: OpticalBlochResult, output: Path) -> Path:
    """保存实验室系、精确旋转系和 RWA 的对比图。"""
    import matplotlib.pyplot as plt

    times = result.times
    lab_rho11, lab_rho22, lab_rho12, _ = result.components("lab")
    rot_rho11, rot_rho22, rot_rho12, _ = result.components("rotating")
    rwa_rho11, rwa_rho22, rwa_rho12, _ = result.components("rwa")

    detuning = result.parameters.detuning
    dipole = result.parameters.dipole
    field_amplitude = result.parameters.field_amplitude
    hbar = result.parameters.hbar
    rabi_coupling = dipole * field_amplitude
    omega_rabi = np.sqrt(detuning**2 + 4.0 * rabi_coupling**2) / hbar

    fig, axes = plt.subplots(2, 3, figsize=(18, 7), sharex="col")

    axes[0, 0].plot(times, lab_rho11.real, label=r"$\rho_{11}$")
    axes[0, 0].plot(times, lab_rho22.real, label=r"$\rho_{22}$")
    axes[0, 0].set_title("Exact Laboratory Frame")
    axes[0, 0].set_ylabel("Population")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[1, 0].plot(times, lab_rho12.real, label=r"Re($\rho_{12}$)")
    axes[1, 0].plot(times, lab_rho12.imag, label=r"Im($\rho_{12}$)")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Coherence")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[0, 1].plot(times, rot_rho11.real, label=r"$\tilde{\rho}_{11}$")
    axes[0, 1].plot(times, rot_rho22.real, label=r"$\tilde{\rho}_{22}$")
    axes[0, 1].set_title("Exact Rotating Frame")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 1].plot(times, rot_rho12.real, label=r"Re($\tilde{\rho}_{12}$)")
    axes[1, 1].plot(times, rot_rho12.imag, label=r"Im($\tilde{\rho}_{12}$)")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[0, 2].plot(times, rwa_rho11.real, label=r"$\tilde{\rho}_{11}^{\mathrm{RWA}}$")
    axes[0, 2].plot(times, rwa_rho22.real, label=r"$\tilde{\rho}_{22}^{\mathrm{RWA}}$")
    axes[0, 2].set_title("Rotating-Wave Approximation")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()

    axes[1, 2].plot(times, rwa_rho12.real, label=r"Re($\tilde{\rho}_{12}^{\mathrm{RWA}}$)")
    axes[1, 2].plot(times, rwa_rho12.imag, label=r"Im($\tilde{\rho}_{12}^{\mathrm{RWA}}$)")
    axes[1, 2].set_xlabel("Time")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    fig.suptitle(
        "Two-Level Optical Bloch Comparison\n"
        f"E0={field_amplitude:.4f}, mu={dipole:.4f}, g=muE0={rabi_coupling:.4f}, "
        f"Delta={detuning:.4f}, hbar={hbar:.4f}, Omega_RWA={omega_rabi:.4f}"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
