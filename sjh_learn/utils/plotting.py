"""绘图与结果保存工具。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .results import MultiLevelResult, OpticalBlochResult


def save_comparison_plot(result: OpticalBlochResult, output: Path) -> Path:
    """保存实验室系、精确旋转系和 RWA 的对比图。"""
    import matplotlib.pyplot as plt

    times, time_label = result.plot_times_and_label()
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
    axes[1, 0].set_xlabel(time_label)
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
    axes[1, 1].set_xlabel(time_label)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[0, 2].plot(times, rwa_rho11.real, label=r"$\tilde{\rho}_{11}^{\mathrm{RWA}}$")
    axes[0, 2].plot(times, rwa_rho22.real, label=r"$\tilde{\rho}_{22}^{\mathrm{RWA}}$")
    axes[0, 2].set_title("Rotating-Wave Approximation")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()

    axes[1, 2].plot(times, rwa_rho12.real, label=r"Re($\tilde{\rho}_{12}^{\mathrm{RWA}}$)")
    axes[1, 2].plot(times, rwa_rho12.imag, label=r"Im($\tilde{\rho}_{12}^{\mathrm{RWA}}$)")
    axes[1, 2].set_xlabel(time_label)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    if result.physical_params is not None and result.solver_params is not None:
        physical = result.physical_params
        solver = result.solver_params
        title_lines = [
            "Two-Level Optical Bloch Comparison",
            (
                f"Eg={physical.energy_gap_eV:.4f} eV, "
                f"EL={physical.laser_energy_eV:.4f} eV, "
                f"Field={physical.field_MV_per_cm:.4f} MV/cm, "
                f"Dipole={physical.dipole_D:.4f} D"
            ),
            (
                f"T1={physical.T1_fs}, Tphi={physical.Tphi_fs}, "
                f"Delta={solver.detuning_fs_inv:.6g} fs^-1, "
                f"Rabi={solver.rabi_fs_inv:.6g} fs^-1"
            ),
        ]
        fig.suptitle("\n".join(title_lines))
    else:
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


def save_multilevel_plot(
    result: MultiLevelResult,
    output: Path,
    populations: list[int] | tuple[int, ...] | None = None,
    coherences: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
) -> Path:
    """保存多能级实验室系结果图。"""
    import matplotlib.pyplot as plt

    times, time_label = result.plot_times_and_label()
    density = result.density_array()
    n_levels = result.dimension()
    population_indices = list(range(n_levels)) if populations is None else list(populations)

    if coherences is None:
        coherence_pairs: list[tuple[int, int]] = []
    else:
        coherence_pairs = list(coherences)

    n_rows = 1 if not coherence_pairs else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 + 3 * (n_rows - 1)), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for index in population_indices:
        axes[0].plot(times, density[:, index, index].real, label=fr"$\rho_{{{index + 1}{index + 1}}}$")
    axes[0].set_ylabel("Population")
    axes[0].set_title("Multi-Level Lab-Frame Exact Simulation")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if coherence_pairs:
        for i, j in coherence_pairs:
            label = fr"$\rho_{{{i + 1}{j + 1}}}$"
            axes[1].plot(times, density[:, i, j].real, label=f"Re({label})")
            axes[1].plot(times, density[:, i, j].imag, linestyle="--", label=f"Im({label})")
        axes[1].set_ylabel("Coherence")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    axes[-1].set_xlabel(time_label)
    energy_text = ", ".join(f"{energy:.3f}" for energy in result.parameters.energies)
    field_text = ", ".join(
        f"{field.name}: A={field.amplitude:.3f}, w={field.omega:.3f}" for field in result.parameters.fields
    )
    fig.suptitle(f"Multi-Level Lab-Frame Exact Simulation\nEnergies=({energy_text}); Fields=({field_text})")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
