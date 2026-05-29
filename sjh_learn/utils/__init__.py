"""`sjh_learn` 里的通用工具模块。"""

from .checks import evaluate_sanity_checks
from .fields import (
    FieldConfig,
    electric_field_array,
    electric_field_value,
    total_electric_field_array,
    total_electric_field_value,
)
from .io import default_output_path, save_parameter_summary
from .model import build_c_ops, build_lab_hamiltonian, build_rwa_hamiltonian, compute_detuning, compute_energy_gap
from .multilevel import (
    CollapseChannel,
    MultiLevelParameters,
    build_multilevel_c_ops,
    build_multilevel_lab_hamiltonian,
    optical_bloch_to_multilevel_parameters,
    run_multilevel_case,
    simulate_multilevel_lab_frame,
    two_level_multilevel_equivalence_check,
)
from .normalization import ParaNormalizer, PhysicalParams, SolverParams
from .parameters import OpticalBlochParameters, ParameterSweep, PhysicalParameterSweep
from .plotting import save_comparison_plot, save_multilevel_plot
from .results import MultiLevelResult, OpticalBlochResult
from .solvers import (
    optical_params_from_solver,
    rotate_density_trajectory,
    rotating_frame_unitary,
    run_case,
    run_parameter_sweep,
    run_physical_case,
    run_physical_parameter_sweep,
    simulate_lab_frame,
    simulate_rwa_frame,
)

__all__ = [
    "OpticalBlochParameters",
    "OpticalBlochResult",
    "MultiLevelResult",
    "PhysicalParams",
    "SolverParams",
    "ParaNormalizer",
    "PhysicalParameterSweep",
    "ParameterSweep",
    "FieldConfig",
    "electric_field_value",
    "electric_field_array",
    "total_electric_field_value",
    "total_electric_field_array",
    "compute_detuning",
    "compute_energy_gap",
    "build_lab_hamiltonian",
    "build_rwa_hamiltonian",
    "build_c_ops",
    "MultiLevelParameters",
    "CollapseChannel",
    "build_multilevel_lab_hamiltonian",
    "build_multilevel_c_ops",
    "optical_bloch_to_multilevel_parameters",
    "simulate_multilevel_lab_frame",
    "run_multilevel_case",
    "two_level_multilevel_equivalence_check",
    "default_output_path",
    "simulate_lab_frame",
    "simulate_rwa_frame",
    "rotating_frame_unitary",
    "rotate_density_trajectory",
    "optical_params_from_solver",
    "run_case",
    "run_physical_case",
    "run_physical_parameter_sweep",
    "run_parameter_sweep",
    "save_parameter_summary",
    "save_comparison_plot",
    "save_multilevel_plot",
    "evaluate_sanity_checks",
]
