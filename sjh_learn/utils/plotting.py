"""Plotting helpers that draw on matplotlib axes and return handles."""

from __future__ import annotations

from typing import Any

import numpy as np

from .results import DynamicsResult


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
    return {
        "lab_exact": "Lab frame",
        "rotating_view": "Rotating view",
        "rwa": "RWA",
        "multilevel_lab": "Multi-level lab",
    }.get(result.mode, result.mode)


def _times_and_label(result: DynamicsResult) -> tuple[np.ndarray, str]:
    if result.times_fs is not None:
        return result.times_fs, "Time (fs)"
    return result.times, "Time"


def plot_field(field, times, ax=None, label: str | None = None, *, ylabel: str = "E(t)"):
    fig, ax = _new_axes(ax)
    values = np.asarray(field(np.asarray(times, dtype=float)), dtype=float)
    ax.plot(times, values, label=label or getattr(field, "name", "field"))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_drive(
    result: DynamicsResult,
    ax=None,
    times: np.ndarray | None = None,
    label: str | None = None,
    *,
    max_points: int = 2000,
):
    fig, ax = _new_axes(ax)
    if result.drive is None:
        ax.text(0.5, 0.5, "derived from lab drive", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("input")
        ax.grid(True, alpha=0.3)
        return fig, ax

    sample_times = result.times if times is None else np.asarray(times, dtype=float)
    if sample_times.size > max_points:
        stride = int(np.ceil(sample_times.size / max_points))
        sample_times = sample_times[::stride]
    values = result.drive_values(sample_times)
    ylabel = "Omega(t)" if result.mode == "rwa" else "E(t)"
    ax.plot(sample_times, values, label=label or result.drive_name or "input")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_populations(result: DynamicsResult, ax=None, populations=None, *, title: str | None = None):
    fig, ax = _new_axes(ax)
    times, time_label = _times_and_label(result)
    density = result.density_array()
    population_indices = list(range(result.dimension())) if populations is None else list(populations)

    for index in population_indices:
        ax.plot(times, density[:, index, index].real, label=fr"$\rho_{{{index + 1}{index + 1}}}$")
    ax.set_xlabel(time_label)
    ax.set_ylabel("Population")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_coherences(result: DynamicsResult, ax=None, coherences=None, *, title: str | None = None):
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
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_density_components(
    result: DynamicsResult,
    axes=None,
    *,
    include_drive: bool = False,
    title: str | None = None,
):
    nrows = 3 if include_drive else 2
    fig, axes_array = _new_axes_array(axes, nrows=nrows, ncols=1, figsize=(7, 2.4 * nrows), sharex=True)
    axes_flat = axes_array.reshape(-1)
    row = 0
    if include_drive:
        plot_drive(result, ax=axes_flat[row])
        axes_flat[row].set_title(title if title is not None else _mode_title(result))
        axes_flat[row].set_xlabel("")
        row += 1
    else:
        axes_flat[row].set_title(title if title is not None else _mode_title(result))
    plot_populations(result, ax=axes_flat[row])
    axes_flat[row].set_xlabel("")
    row += 1
    plot_coherences(result, ax=axes_flat[row], coherences=[(0, 1)])
    return fig, axes_array


def plot_multilevel_components(
    result: DynamicsResult,
    axes=None,
    *,
    populations: list[int] | tuple[int, ...] | None = None,
    coherences: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    title: str | None = None,
):
    n_rows = 1 if not coherences else 2
    fig, axes_array = _new_axes_array(axes, nrows=n_rows, ncols=1, figsize=(10, 4 + 3 * (n_rows - 1)), sharex=True)
    axes_flat = axes_array.reshape(-1)
    plot_populations(result, ax=axes_flat[0], populations=populations, title=title if title is not None else "Multi-level")
    if coherences:
        plot_coherences(result, ax=axes_flat[1], coherences=coherences)
        axes_flat[0].set_xlabel("")
    return fig, axes_array


def build_preview_figure(result: DynamicsResult, *, coherences=None):
    if result.dimension() == 2:
        fig, axes = plot_density_components(result, include_drive=True)
    else:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        plot_drive(result, ax=axes[0])
        axes[0].set_title(_mode_title(result))
        plot_populations(result, ax=axes[1])
        axes[1].set_xlabel("")
        plot_coherences(result, ax=axes[2], coherences=coherences or [(0, 1)])
    fig.tight_layout()
    return fig, axes


__all__ = [
    "plot_field",
    "plot_drive",
    "plot_populations",
    "plot_coherences",
    "plot_density_components",
    "plot_multilevel_components",
    "build_preview_figure",
]
