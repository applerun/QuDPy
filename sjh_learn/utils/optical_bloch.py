"""两能级光学 Bloch 演化相关工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from qutip import Qobj, basis, mesolve
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "运行 sjh_learn 的 optical bloch 工具需要先安装 qutip。"
    ) from exc


@dataclass(frozen=True)
class OpticalBlochParameters:
    """单个两能级驱动问题所需的参数。"""

    t_final: float = 120.0
    dt: float = 0.01
    hbar: float = 1.0
    epsilon_1: float = 0.0
    detuning: float = 0.0
    dipole: float = 0.08
    field_amplitude: float = 1.0
    omega_drive: float = 1.0


@dataclass(frozen=True)
class ParameterSweep:
    """参数扫描设置。"""

    t_final: float = 120.0
    dt: float = 0.01
    hbar: float = 1.0
    epsilon_1: float = 0.0
    dipole: float = 0.08
    omega_drive: float = 1.0
    field_amplitudes: tuple[float, ...] = (0.5, 1.0, 2.0)
    detunings: tuple[float, ...] = (0.0,)


@dataclass
class OpticalBlochResult:
    """保存单个参数点下的全部计算结果。"""

    parameters: OpticalBlochParameters
    epsilon_2: float
    times: np.ndarray
    lab_states: list[Qobj]
    rotating_states: list[Qobj]
    rwa_states: list[Qobj]

    def density_array(self, frame: str) -> np.ndarray:
        """把某个参考系下的密度矩阵列表转换为 `(nt, 2, 2)` 数组。"""
        state_map = {
            "lab": self.lab_states,
            "rotating": self.rotating_states,
            "rwa": self.rwa_states,
        }
        try:
            states = state_map[frame]
        except KeyError as exc:
            raise ValueError(f"未知参考系: {frame}") from exc
        return np.stack([state.full() for state in states], axis=0)

    def components(self, frame: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回 `rho11`、`rho22`、`rho12` 和 `rho21`。"""
        density = self.density_array(frame)
        return density[:, 0, 0], density[:, 1, 1], density[:, 0, 1], density[:, 1, 0]

    def max_trace_error(self, frame: str = "lab") -> float:
        """计算迹偏离 1 的最大误差。"""
        rho11, rho22, _rho12, _rho21 = self.components(frame)
        return float(np.max(np.abs((rho11 + rho22) - 1.0)))

    def max_hermiticity_error(self, frame: str = "lab") -> float:
        """计算厄米性偏离 `rho21 = rho12*` 的最大误差。"""
        _rho11, _rho22, rho12, rho21 = self.components(frame)
        return float(np.max(np.abs(rho21 - np.conjugate(rho12))))

    def summary_dict(self) -> dict[str, str]:
        """生成便于打印或保存的结果摘要。"""
        rho11, rho22, rho12, rho21 = self.components("lab")
        return {
            "epsilon_1": f"{self.parameters.epsilon_1:.6f}",
            "epsilon_2": f"{self.epsilon_2:.6f}",
            "rho11_lab(final)": f"{rho11[-1].real:.6f}",
            "rho22_lab(final)": f"{rho22[-1].real:.6f}",
            "rho12_lab(final)": f"{rho12[-1].real:.6f} + {rho12[-1].imag:.6f}i",
            "rho21_lab(final)": f"{rho21[-1].real:.6f} + {rho21[-1].imag:.6f}i",
            "max trace error": f"{self.max_trace_error('lab'):.3e}",
            "max Hermitian error": f"{self.max_hermiticity_error('lab'):.3e}",
        }


def electric_field(times: np.ndarray, amplitude: float, omega_drive: float) -> np.ndarray:
    """计算驱动场 `E(t) = 2 E0 cos(omega t)`。"""
    return 2.0 * amplitude * np.cos(omega_drive * times)


def compute_detuning(
    epsilon_1: float,
    epsilon_2: float,
    omega_drive: float,
    hbar: float,
) -> float:
    """计算 `Delta = (epsilon_2 - epsilon_1) - hbar * omega_drive`。"""
    return (epsilon_2 - epsilon_1) - hbar * omega_drive


def compute_energy_gap(
    detuning: float,
    omega_drive: float,
    hbar: float,
) -> float:
    """根据失谐量求能级差 `epsilon_2 - epsilon_1`。"""
    return hbar * omega_drive + detuning


def format_value_tag(value: float) -> str:
    """把浮点数格式化成适合文件名的标签。"""
    return f"{value:.4f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _basis_operators() -> tuple[Qobj, Qobj, Qobj]:
    """返回两能级系统常用算符。"""
    ket_1 = basis(2, 0)
    ket_2 = basis(2, 1)
    projector_1 = ket_1 * ket_1.dag()
    projector_2 = ket_2 * ket_2.dag()
    dipole_operator = ket_1 * ket_2.dag() + ket_2 * ket_1.dag()
    return projector_1, projector_2, dipole_operator


def initial_density_matrix() -> Qobj:
    """初态取为 `|1><1|`。"""
    ket_1 = basis(2, 0)
    return ket_1 * ket_1.dag()


def build_lab_hamiltonian(parameters: OpticalBlochParameters) -> list[Qobj | list[object]]:
    """构造实验室参考系中的含时哈密顿量。"""
    projector_1, projector_2, dipole_operator = _basis_operators()
    epsilon_2 = parameters.epsilon_1 + compute_energy_gap(
        detuning=parameters.detuning,
        omega_drive=parameters.omega_drive,
        hbar=parameters.hbar,
    )
    h0 = parameters.epsilon_1 * projector_1 + epsilon_2 * projector_2
    h_drive = -parameters.dipole * dipole_operator
    return [
        h0,
        [h_drive, lambda t, args: 2.0 * args["field_amplitude"] * np.cos(args["omega_drive"] * t)],
    ]


def build_rwa_hamiltonian(parameters: OpticalBlochParameters) -> Qobj:
    """构造旋转参考系下的 RWA 哈密顿量。"""
    g = parameters.dipole * parameters.field_amplitude
    return Qobj(
        np.array(
            [
                [0.0, -g],
                [-g, parameters.detuning],
            ],
            dtype=np.complex128,
        )
    )


def simulate_lab_frame(parameters: OpticalBlochParameters) -> tuple[np.ndarray, list[Qobj]]:
    """用 QuTiP 的 `mesolve` 求实验室参考系中的精确演化。"""
    times = np.arange(0.0, parameters.t_final + 0.5 * parameters.dt, parameters.dt)
    result = mesolve(
        H=build_lab_hamiltonian(parameters),
        rho0=initial_density_matrix(),
        tlist=times,
        c_ops=[],
        e_ops=[],
        args={
            "field_amplitude": parameters.field_amplitude,
            "omega_drive": parameters.omega_drive,
        },
    )
    return times, list(result.states)


def simulate_rwa_frame(parameters: OpticalBlochParameters, times: np.ndarray) -> list[Qobj]:
    """用 QuTiP 的 `mesolve` 求旋波近似下的演化。"""
    result = mesolve(
        H=build_rwa_hamiltonian(parameters),
        rho0=initial_density_matrix(),
        tlist=times,
        c_ops=[],
        e_ops=[],
    )
    return list(result.states)


def rotating_frame_unitary(time: float, omega_drive: float) -> Qobj:
    """构造 `U(t)=diag(1, e^{-i omega t})`。"""
    return Qobj(
        np.array(
            [
                [1.0, 0.0],
                [0.0, np.exp(-1j * omega_drive * time)],
            ],
            dtype=np.complex128,
        )
    )


def rotate_density_trajectory(times: np.ndarray, lab_states: list[Qobj], omega_drive: float) -> list[Qobj]:
    """把实验室参考系中的精确结果变换到旋转参考系。"""
    rotated_states: list[Qobj] = []
    for time, rho_lab in zip(times, lab_states):
        unitary = rotating_frame_unitary(time, omega_drive)
        rotated_states.append(unitary.dag() * rho_lab * unitary)
    return rotated_states


def run_case(parameters: OpticalBlochParameters) -> OpticalBlochResult:
    """运行单个参数点，返回三种参考系下的结果。"""
    epsilon_2 = parameters.epsilon_1 + compute_energy_gap(
        detuning=parameters.detuning,
        omega_drive=parameters.omega_drive,
        hbar=parameters.hbar,
    )
    times, lab_states = simulate_lab_frame(parameters)
    rotating_states = rotate_density_trajectory(times, lab_states, parameters.omega_drive)
    rwa_states = simulate_rwa_frame(parameters, times)
    return OpticalBlochResult(
        parameters=parameters,
        epsilon_2=epsilon_2,
        times=times,
        lab_states=lab_states,
        rotating_states=rotating_states,
        rwa_states=rwa_states,
    )


def run_parameter_sweep(sweep: ParameterSweep) -> list[OpticalBlochResult]:
    """遍历给定的驱动振幅和失谐量。"""
    results: list[OpticalBlochResult] = []
    for detuning in sweep.detunings:
        for field_amplitude in sweep.field_amplitudes:
            parameters = OpticalBlochParameters(
                t_final=sweep.t_final,
                dt=sweep.dt,
                hbar=sweep.hbar,
                epsilon_1=sweep.epsilon_1,
                detuning=detuning,
                dipole=sweep.dipole,
                field_amplitude=field_amplitude,
                omega_drive=sweep.omega_drive,
            )
            results.append(run_case(parameters))
    return results


def default_output_path(output_dir: Path, result: OpticalBlochResult) -> Path:
    """为单个结果生成默认输出图路径。"""
    amplitude_tag = format_value_tag(result.parameters.field_amplitude)
    detuning_tag = format_value_tag(result.parameters.detuning)
    return output_dir / f"comparison_E0_{amplitude_tag}_Delta_{detuning_tag}.png"
