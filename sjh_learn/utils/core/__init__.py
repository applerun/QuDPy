"""core 层公开入口。

core 只负责参数、归一化、Hamiltonian / collapse operator 构造、求解入口和
`DynamicsResult` 容器；不要从这里依赖 analysis observable。
"""

from .model import build_c_ops, build_lab_hamiltonian, compute_detuning, compute_energy_gap
from .normalization import ParaNormalizer
from .parameters import (
    NLevelPhysicalParams,
    NLevelSolverParams,
    ParameterSweep,
    PhysicalParameterSweep,
    PureDephasingChannel,
    RelaxationChannel,
    SolverParams,
)
from .results import DynamicsResult
from .solvers import (
    default_rwa_drive,
    make_rotating_view,
    optical_params_from_solver,
    rotate_density_trajectory,
    rotating_frame_unitary,
    run_case,
    run_lab_case,
    run_parameter_sweep,
    run_physical_case,
    run_physical_parameter_sweep,
    run_rwa_case,
    simulate_lab_frame,
    simulate_rwa_frame,
)

__all__ = [
    "DynamicsResult",
    "NLevelPhysicalParams",
    "NLevelSolverParams",
    "RelaxationChannel",
    "PureDephasingChannel",
    "SolverParams",
    "ParaNormalizer",
    "PhysicalParameterSweep",
    "ParameterSweep",
    "compute_detuning",
    "compute_energy_gap",
    "build_lab_hamiltonian",
    "build_c_ops",
    "simulate_lab_frame",
    "simulate_rwa_frame",
    "rotating_frame_unitary",
    "rotate_density_trajectory",
    "optical_params_from_solver",
    "default_rwa_drive",
    "run_case",
    "run_lab_case",
    "run_physical_case",
    "run_parameter_sweep",
    "run_physical_parameter_sweep",
    "run_rwa_case",
    "make_rotating_view",
]
