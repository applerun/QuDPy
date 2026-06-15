from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sjh_learn.utils.core import NLevelPhysicalParams
from three_level_absorption_lab_exact_casewise_meta import make_three_level_params


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _rate_from_T_or_rate(
    *,
    T_fs: float | None,
    rate_fs_inv: float | None,
    channel_name: str,
) -> float:
    if rate_fs_inv is not None:
        rate = float(rate_fs_inv)
    elif T_fs is not None:
        if float(T_fs) <= 0:
            raise ValueError(f"{channel_name}: T_fs must be positive.")
        rate = 1.0 / float(T_fs)
    else:
        raise ValueError(
            f"{channel_name}: either T_fs or rate_fs_inv must be provided."
        )

    if rate < 0:
        raise ValueError(f"{channel_name}: Gamma must be non-negative.")
    return rate


def _relaxation_L_matrix(*, n_levels: int, from_level: int, to_level: int) -> np.ndarray:
    L = np.zeros((n_levels, n_levels), dtype=np.complex128)
    L[int(to_level), int(from_level)] = 1.0
    return L


def _pure_dephasing_L_matrix(*, n_levels: int, level: int) -> np.ndarray:
    L = np.zeros((n_levels, n_levels), dtype=np.complex128)
    L[int(level), int(level)] = 1.0
    return L


def _qutip_C_matrix(L: np.ndarray, gamma_fs_inv: float) -> np.ndarray:
    """QuTiP collapse operator: C_k = sqrt(Gamma_k) L_k."""

    return np.sqrt(float(gamma_fs_inv)) * np.asarray(L, dtype=np.complex128)


def _dissipator_superoperator(L: np.ndarray, gamma_fs_inv: float) -> np.ndarray:
    """Gamma * D[L] in Liouville space.

    Vectorization convention:
        column-major vec(rho)

    For n=3:
        vec(rho) = [
            rho_00, rho_10, rho_20,
            rho_01, rho_11, rho_21,
            rho_02, rho_12, rho_22
        ]^T

    Identity:
        vec(A rho B) = (B^T kron A) vec(rho)
    """

    n = L.shape[0]
    I = np.eye(n, dtype=np.complex128)
    LdagL = L.conj().T @ L

    return float(gamma_fs_inv) * (
        np.kron(L.conj(), L)
        - 0.5 * np.kron(I, LdagL)
        - 0.5 * np.kron(LdagL.T, I)
    )


def _liouville_basis_labels(n_levels: int) -> list[str]:
    """Column-major vec(rho) labels."""

    return [
        rf"\rho_{{{row}{col}}}"
        for col in range(n_levels)
        for row in range(n_levels)
    ]


def _format_scalar_for_latex(z: complex, tol: float = 1e-12) -> str:
    z = complex(z)
    zr = 0.0 if abs(z.real) < tol else z.real
    zi = 0.0 if abs(z.imag) < tol else z.imag

    if zi == 0.0:
        return f"{zr:.4g}"
    if zr == 0.0:
        if abs(zi - 1.0) < tol:
            return "i"
        if abs(zi + 1.0) < tol:
            return "-i"
        return f"{zi:.4g}i"

    sign = "+" if zi >= 0 else "-"
    abs_zi = abs(zi)
    imag_part = "i" if abs(abs_zi - 1.0) < tol else f"{abs_zi:.4g}i"
    return f"{zr:.4g}{sign}{imag_part}"


def _matrix_to_latex_bmatrix(M: np.ndarray) -> str:
    rows = []
    for row in M:
        rows.append(" & ".join(_format_scalar_for_latex(x) for x in row))
    body = r" \\ ".join(rows)
    return r"\begin{bmatrix}" + body + r"\end{bmatrix}"


def _matrix_to_plain_block(M: np.ndarray) -> str:
    lines = []
    for row in M:
        lines.append("[ " + ", ".join(f"{_format_scalar_for_latex(x):>10s}" for x in row) + " ]")
    return "\n".join(lines)


def _save_matrix_text_plot(
    matrix: np.ndarray,
    path: Path,
    *,
    title: str,
    subtitle: str | None = None,
    fontsize: float = 12,
) -> None:
    """Save a readable matrix text plot; .tex is saved separately."""

    text_block = _matrix_to_plain_block(matrix)
    n_rows, n_cols = matrix.shape
    fig_w = max(6.2, 0.9 * n_cols + 2.2)
    fig_h = max(3.2, 0.48 * n_rows + 1.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    y = 0.96
    ax.text(0.02, y, title, ha="left", va="top", fontsize=14, transform=ax.transAxes)
    y -= 0.10

    if subtitle is not None:
        ax.text(0.02, y, subtitle, ha="left", va="top", fontsize=10.5, transform=ax.transAxes)
        y -= 0.10

    ax.text(
        0.02,
        y,
        text_block,
        ha="left",
        va="top",
        family="monospace",
        fontsize=fontsize,
        transform=ax.transAxes,
    )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_liouville_superoperator_plot(
    superop: np.ndarray,
    path: Path,
    *,
    labels: list[str],
    title: str,
    subtitle: str | None = None,
) -> None:
    """Save a Liouville-space superoperator heatmap with rho-index order."""

    matrix = np.asarray(superop, dtype=np.complex128)
    values = np.real(matrix)
    imag_max = float(np.max(np.abs(np.imag(matrix)))) if matrix.size else 0.0

    vmax = float(np.max(np.abs(values))) if values.size else 1.0
    if vmax == 0.0:
        vmax = 1.0

    n = values.shape[0]
    fig_w = max(7.0, 0.62 * n + 3.0)
    fig_h = max(6.2, 0.62 * n + 2.4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, vmin=-vmax, vmax=vmax, aspect="equal")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel(r"input component of $\mathrm{vec}(\rho)$")
    ax.set_ylabel(r"output component of $\mathrm{vec}(\rho)$")
    ax.set_title(title)

    if subtitle is not None:
        ax.text(
            0.0,
            1.06,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )

    for row in range(n):
        for col in range(n):
            value = values[row, col]
            if abs(value) > 0.05 * vmax:
                ax.text(col, row, f"{value:.2g}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Re[$\Gamma_k \mathcal{D}[L_k]$] (fs$^{-1}$)")

    if imag_max > 1e-12:
        ax.text(
            0.0,
            -0.14,
            f"Warning: max |imaginary part| = {imag_max:.3g}; heatmap shows real part only.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_liouville_vector_order_plot(
    *,
    labels: list[str],
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    """Save Liouville vector order as .tex and .png."""

    output_dir.mkdir(parents=True, exist_ok=True)

    latex = (
        r"\begin{bmatrix}"
        + r" \\ ".join(labels)
        + r"\end{bmatrix}"
    )
    entries = [rf"vec(rho)[{idx}] = {label}" for idx, label in enumerate(labels)]

    tex_path = output_dir / f"{prefix}_liouville_vector_order.tex"
    png_path = output_dir / f"{prefix}_liouville_vector_order.png"

    _write_text(
        tex_path,
        "% Column-major vectorization order used by this exporter.\n"
        "% vec(A rho B) = (B^T \\otimes A) vec(rho)\n"
        r"\mathrm{vec}(\rho)=" + latex + "\n",
    )

    fig_h = max(3.2, 0.34 * len(labels) + 1.6)
    fig, ax = plt.subplots(figsize=(6.8, fig_h))
    ax.axis("off")
    ax.text(
        0.02,
        0.97,
        "Liouville vector order: column-major vec(rho)",
        ha="left",
        va="top",
        fontsize=13,
        transform=ax.transAxes,
    )
    ax.text(
        0.02,
        0.88,
        "Convention: vec(A rho B) = (B^T kron A) vec(rho)",
        ha="left",
        va="top",
        fontsize=10,
        transform=ax.transAxes,
    )
    ax.text(
        0.05,
        0.78,
        "\n".join(entries),
        ha="left",
        va="top",
        fontsize=11,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    return {"tex": tex_path, "figure": png_path}


def export_lindblad_gamma_summary(
    physical_params: NLevelPhysicalParams,
    output_dir: str | Path,
    *,
    prefix: str = "lindblad_gamma_summary",
    plot_unit: str = "fs^-1",
    time_scale_fs: float | None = None,
    log_y: bool = False,
) -> dict[str, Path]:
    """Export Lindblad rates, C_k matrices, and Liouville vector order.

    Plot policy:
        1. channel Gamma scalar bar plot;
        2. QuTiP-style C_k matrices for each channel;
        3. Gamma_k D[L_k] Liouville-space superoperator heatmaps for each channel,
           with rho-index vectorization order shown on the axes.

    L_k is not exported as a standalone field. It is only used internally to
    construct C_k and Gamma_k D[L_k].
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_levels = len(physical_params.energies_eV)
    liouville_labels = _liouville_basis_labels(n_levels)
    channels: list[dict[str, Any]] = []

    channel_matrix_dir = output_dir / f"{prefix}_channel_matrices"
    channel_matrix_dir.mkdir(parents=True, exist_ok=True)

    vector_order_paths = _save_liouville_vector_order_plot(
        labels=liouville_labels,
        output_dir=output_dir,
        prefix=prefix,
    )

    def _append_channel(
        *,
        name: str,
        kind: str,
        operator_expression: str,
        gamma_fs_inv: float,
        T_fs: float | None,
        L: np.ndarray,
        extra: dict[str, Any],
    ) -> None:
        C = _qutip_C_matrix(L, gamma_fs_inv)
        superop = _dissipator_superoperator(L, gamma_fs_inv)

        safe_name = (
            name.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

        C_tex = _matrix_to_latex_bmatrix(C)
        GammaD_tex = _matrix_to_latex_bmatrix(superop)

        C_tex_path = channel_matrix_dir / f"{safe_name}_C_qutip_matrix.tex"
        C_png_path = channel_matrix_dir / f"{safe_name}_C_qutip_matrix.png"
        GammaD_tex_path = channel_matrix_dir / f"{safe_name}_GammaD_superoperator.tex"
        GammaD_png_path = channel_matrix_dir / f"{safe_name}_GammaD_superoperator.png"

        _write_text(
            C_tex_path,
            rf"$C_k=\sqrt{{\Gamma_k}}\,L_k={C_tex}$" + "\n",
        )
        _write_text(
            GammaD_tex_path,
            rf"$\Gamma_k\mathcal{{D}}[L_k]={GammaD_tex}$" + "\n",
        )

        _save_matrix_text_plot(
            C,
            C_png_path,
            title=f"{name}: QuTiP collapse operator C_k",
            subtitle=rf"C_k = sqrt(Gamma_k) L_k, Gamma = {gamma_fs_inv:.6g} fs^-1",
            fontsize=12,
        )
        _save_liouville_superoperator_plot(
            superop,
            GammaD_png_path,
            labels=liouville_labels,
            title=f"{name}: Gamma_k D[L_k] in Liouville space",
            subtitle="Axis labels show the exact vec(rho) order used by this exporter.",
        )

        item = {
            "name": name,
            "kind": kind,
            "operator_expression_L": operator_expression,
            "T_fs": None if T_fs is None else float(T_fs),
            "Gamma_scalar_fs_inv": gamma_fs_inv,
            "Gamma_scalar_ps_inv": gamma_fs_inv * 1000.0,
            "C_qutip_matrix": C,
            "Gamma_D_superoperator_fs_inv": superop,
            "liouville_vectorization": {
                "order": "column_major",
                "vec_rho_labels": liouville_labels,
                "identity": "vec(A rho B) = (B^T kron A) vec(rho)",
                "superoperator_shape": list(superop.shape),
            },
            "latex_files": {
                "C_qutip_matrix_tex": C_tex_path,
                "GammaD_superoperator_tex": GammaD_tex_path,
            },
            "plot_files": {
                "C_qutip_matrix_png": C_png_path,
                "GammaD_superoperator_png": GammaD_png_path,
            },
            "qutip_c_op_note": "QuTiP receives c_k = sqrt(Gamma_k) * L_k. This C_qutip_matrix is that c_k.",
            **extra,
        }

        if time_scale_fs is not None:
            item["Gamma_code"] = gamma_fs_inv * float(time_scale_fs)

        channels.append(item)

    for channel in physical_params.relaxation_channels:
        name = str(channel.name)
        gamma_fs_inv = _rate_from_T_or_rate(
            T_fs=channel.T1_fs,
            rate_fs_inv=channel.rate_fs_inv,
            channel_name=name,
        )
        from_level = int(channel.from_level)
        to_level = int(channel.to_level)
        L = _relaxation_L_matrix(
            n_levels=n_levels,
            from_level=from_level,
            to_level=to_level,
        )
        _append_channel(
            name=name,
            kind="relaxation",
            operator_expression=f"|{to_level}><{from_level}|",
            gamma_fs_inv=gamma_fs_inv,
            T_fs=channel.T1_fs,
            L=L,
            extra={"from_level": from_level, "to_level": to_level},
        )

    for channel in physical_params.pure_dephasing_channels:
        name = str(channel.name)
        gamma_fs_inv = _rate_from_T_or_rate(
            T_fs=channel.Tphi_fs,
            rate_fs_inv=channel.rate_fs_inv,
            channel_name=name,
        )
        level = int(channel.level)
        L = _pure_dephasing_L_matrix(n_levels=n_levels, level=level)
        _append_channel(
            name=name,
            kind="pure_dephasing",
            operator_expression=f"|{level}><{level}|",
            gamma_fs_inv=gamma_fs_inv,
            T_fs=channel.Tphi_fs,
            L=L,
            extra={"level": level},
        )

    gamma_vector_fs_inv = np.asarray(
        [item["Gamma_scalar_fs_inv"] for item in channels],
        dtype=float,
    )
    gamma_diagonal_fs_inv = np.diag(gamma_vector_fs_inv)

    if channels:
        total_dissipator_superoperator_fs_inv = np.sum(
            [np.asarray(item["Gamma_D_superoperator_fs_inv"], dtype=np.complex128) for item in channels],
            axis=0,
        )
    else:
        total_dissipator_superoperator_fs_inv = np.zeros(
            (n_levels * n_levels, n_levels * n_levels),
            dtype=np.complex128,
        )

    payload = {
        "equation": "d rho / dt = -i[H(t), rho] + sum_k Gamma_k D[L_k]rho",
        "equivalent_qutip_form": "d rho / dt = -i[H(t), rho] + sum_k D[C_k]rho, C_k = sqrt(Gamma_k) L_k",
        "dissipator_definition": (
            "D[X]rho = X rho X^dagger - 1/2 * (X^dagger X rho + rho X^dagger X)"
        ),
        "n_levels": n_levels,
        "basis": None if physical_params.basis is None else list(physical_params.basis),
        "energies_eV": list(physical_params.energies_eV),
        "liouville_vectorization": {
            "order": "column_major",
            "vec_rho_labels": liouville_labels,
            "identity": "vec(A rho B) = (B^T kron A) vec(rho)",
            "tex": vector_order_paths["tex"],
            "figure": vector_order_paths["figure"],
        },
        "n_channels": len(channels),
        "channels": channels,
        "Gamma_vector_fs_inv": gamma_vector_fs_inv,
        "Gamma_diagonal_fs_inv": gamma_diagonal_fs_inv,
        "total_dissipator_superoperator_fs_inv": total_dissipator_superoperator_fs_inv,
        "notes": {
            "Gamma_vector_fs_inv": "Channel-space Gamma vector, not a Liouville-space matrix.",
            "Gamma_diagonal_fs_inv": "Channel-space diagonal matrix, shape = n_channels x n_channels.",
            "C_qutip_matrix": "QuTiP collapse operator C_k = sqrt(Gamma_k) L_k. This is plotted per channel.",
            "L_operator_matrix": "Not exported as a standalone JSON field; L_k is only used internally.",
            "Gamma_D_superoperator_fs_inv": (
                "Per-channel Liouville-space superoperator, shape = (n_levels^2) x (n_levels^2). "
                "This is also plotted with vec(rho) order on the axes."
            ),
            "total_dissipator_superoperator_fs_inv": (
                "Sum over all channel superoperators, shape = (n_levels^2) x (n_levels^2)."
            ),
        },
    }

    json_path = output_dir / f"{prefix}.json"
    _write_json(json_path, payload)

    fig_path = output_dir / f"{prefix}.png"

    fig, ax = plt.subplots(figsize=(max(6.4, 0.75 * max(1, len(channels))), 4.2))

    if channels:
        labels = [item["name"] for item in channels]

        if plot_unit == "fs^-1":
            values = np.asarray([item["Gamma_scalar_fs_inv"] for item in channels], dtype=float)
            ylabel = r"$\Gamma$ (fs$^{-1}$)"
        elif plot_unit == "ps^-1":
            values = np.asarray([item["Gamma_scalar_ps_inv"] for item in channels], dtype=float)
            ylabel = r"$\Gamma$ (ps$^{-1}$)"
        else:
            raise ValueError('plot_unit must be "fs^-1" or "ps^-1".')

        x = np.arange(len(channels))
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title("Lindblad channel rates")

        if log_y and np.all(values > 0):
            ax.set_yscale("log")

        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No Lindblad channels", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    return {
        "json": json_path,
        "figure": fig_path,
        "channel_matrix_dir": channel_matrix_dir,
        "liouville_vector_order_tex": vector_order_paths["tex"],
        "liouville_vector_order_figure": vector_order_paths["figure"],
    }


if __name__ == "__main__":
    params = make_three_level_params()

    output_dir = (
        Path(__file__).resolve().parent
        / "outputs"
        / "three_level_absorption_lab_exact"
        / "equation_terms"
    )

    paths = export_lindblad_gamma_summary(
        physical_params=params,
        output_dir=output_dir,
        prefix="three_level_lindblad_gamma",
        plot_unit="ps^-1",
        time_scale_fs=None,
        log_y=False,
    )

    print("Exported Lindblad gamma summary:")
    print(f"  JSON                    : {paths['json']}")
    print(f"  Rate figure             : {paths['figure']}")
    print(f"  Channel matrices/plots  : {paths['channel_matrix_dir']}")
    print(f"  Liouville vector order  : {paths['liouville_vector_order_figure']}")
