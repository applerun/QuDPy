"""Plotting helpers that draw on matplotlib axes and return handles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .results import DynamicsResult, MultiLevelResult, OpticalBlochResult


def _new_axes(ax=None, *, figsize=(8, 4)):
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.figure, ax
    fig, created_ax = plt.subplots(figsize=figsize)
    return fig, created_ax


def _new_axes_array(axes=None, *, nrows: int = 2, ncols: int = 1, figsize=(8, 6), sharex=True):
    import matplotlib.pyplot as plt

    if axes is not None:
        axes_array = np.asarray(axes, dtype=object)
        return axes_array.flat[0].figure, axes_array
    fig, created_axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex)
    return fig, np.asarray(created_axes, dtype=object)


def _mode_title(result: DynamicsResult) -> str:
    titles = {
        "lab_exact": "Exact Laboratory Frame",
        "rotating_view": "Exact Rotating View",
        "rwa": "Rotating-Wave Approximation",
        "multilevel_lab": "Multi-Level Lab Frame",
    }
    return titles.get(result.mode, result.mode)


def _times_and_label(result: Any) -> tuple[np.ndarray, str]:
    if hasattr(result, "plot_times_and_label"):
        return result.plot_times_and_label()
    if result.times_fs is not None:
        return result.times_fs, "Time (fs)"
    return result.times, "Time"


def plot_populations(result: DynamicsResult | MultiLevelResult, ax=None, populations=None, *, title: str | None = None):
    fig, ax = _new_axes(ax)
    times, time_label = _times_and_label(result)
    density = result.density_array()
    population_indices = list(range(result.dimension())) if populations is None else list(populations)

    for index in population_indices:
        ax.plot(times, density[:, index, index].real, label=fr"$\rho_{{{index + 1}{index + 1}}}$")
    ax.set_xlabel(time_label)
    ax.set_ylabel("Population")
    ax.set_title(title if title is not None else getattr(result, "mode", "populations"))
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_coherences(result: DynamicsResult | MultiLevelResult, ax=None, coherences=None, *, title: str | None = None):
    fig, ax = _new_axes(ax)
    times, time_label = _times_and_label(result)
    if coherences is None:
        coherences = [(0, 1)] if result.dimension() >= 2 else []

    for i, j in coherences:
        values = result.matrix_element(i, j)
        label = fr"$\rho_{{{i + 1}{j + 1}}}$"
        ax.plot(times, values.real, label=f"Re({label})")
        ax.plot(times, values.imag, linestyle="--", label=f"Im({label})")
    ax.set_xlabel(time_label)
    ax.set_ylabel("Coherence")
    ax.set_title(title if title is not None else getattr(result, "mode", "coherences"))
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_density_components(result: DynamicsResult, axes=None, *, title: str | None = None):
    if result.dimension() != 2:
        raise ValueError("plot_density_components expects a two-level DynamicsResult.")
    fig, axes_array = _new_axes_array(axes, nrows=2, ncols=1, figsize=(7, 5), sharex=True)
    axes_flat = axes_array.reshape(-1)
    plot_populations(result, ax=axes_flat[0], title=title if title is not None else _mode_title(result))
    plot_coherences(result, ax=axes_flat[1], coherences=[(0, 1)], title="")
    axes_flat[0].set_xlabel("")
    return fig, axes_array


def plot_multilevel_components(
    result: DynamicsResult | MultiLevelResult,
    axes=None,
    *,
    populations: list[int] | tuple[int, ...] | None = None,
    coherences: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    title: str | None = None,
):
    n_rows = 1 if not coherences else 2
    fig, axes_array = _new_axes_array(axes, nrows=n_rows, ncols=1, figsize=(10, 4 + 3 * (n_rows - 1)), sharex=True)
    axes_flat = axes_array.reshape(-1)
    plot_populations(result, ax=axes_flat[0], populations=populations, title=title if title is not None else "Multi-Level")
    if coherences:
        plot_coherences(result, ax=axes_flat[1], coherences=coherences, title="")
        axes_flat[0].set_xlabel("")
    return fig, axes_array


def save_comparison_plot(result: OpticalBlochResult, output: Path, dpi: int = 160) -> Path:
    """Deprecated compatibility wrapper. New code should compose figures in top-level scripts."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 7), sharex="col")
    for col, frame in enumerate(result.available_frames()):
        rho11, rho22, rho12, _rho21 = result.components(frame)
        times, time_label = result.plot_times_and_label()
        axes[0, col].plot(times, rho11.real, label=r"$\rho_{11}$")
        axes[0, col].plot(times, rho22.real, label=r"$\rho_{22}$")
        axes[0, col].set_title({"lab": "Exact Laboratory Frame", "rotating": "Exact Rotating Frame", "rwa": "RWA"}[frame])
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].legend()
        axes[1, col].plot(times, rho12.real, label=r"Re($\rho_{12}$)")
        axes[1, col].plot(times, rho12.imag, label=r"Im($\rho_{12}$)")
        axes[1, col].set_xlabel(time_label)
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].legend()
    axes[0, 0].set_ylabel("Population")
    axes[1, 0].set_ylabel("Coherence")
    fig.suptitle("Two-Level Optical Bloch Comparison")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi))
    plt.close(fig)
    return output


def save_multilevel_plot(
    result: DynamicsResult | MultiLevelResult,
    output: Path,
    populations: list[int] | tuple[int, ...] | None = None,
    coherences: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    dpi: int = 160,
) -> Path:
    """Deprecated compatibility wrapper. New code should save figures via io.save_figure."""
    import matplotlib.pyplot as plt

    fig, _axes = plot_multilevel_components(result, populations=populations, coherences=coherences)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi))
    plt.close(fig)
    return output


__all__ = [
    "plot_populations",
    "plot_coherences",
    "plot_density_components",
    "plot_multilevel_components",
    "save_comparison_plot",
    "save_multilevel_plot",
]
