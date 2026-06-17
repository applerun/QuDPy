#!/usr/bin/env python3
"""验证 QuDPy compact metadata schema。"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sjh_learn.utils.core import NLevelPhysicalParams, ParaNormalizer, run_case
from sjh_learn.utils.fields import make_default_carrier_field
from sjh_learn.utils.io import save_result_case


OUT_ROOT = Path("/private/tmp/qudpy_validate_metadata_schema")


def run_saved_case(name: str, physical: NLevelPhysicalParams) -> tuple[dict, dict]:
    result = run_case(physical, normalizer=ParaNormalizer(time_scale_fs=1.0, auto_scale=False))
    saved = save_result_case(
        result,
        OUT_ROOT,
        case_name=name,
        output_preview=True,
        append_results_csv=False,
        save_populations_csv=True,
    )
    case_dir = saved["case_dir"]
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    debug = json.loads((case_dir / "debug_meta.json").read_text(encoding="utf-8"))
    return meta, debug


def assert_no_field_user_metadata(payload) -> None:
    if isinstance(payload, dict):
        if "user_metadata" in payload:
            raise AssertionError("field metadata should not contain user_metadata.")
        if "dipoles_D" in payload or "transitions_eV" in payload:
            raise AssertionError("field metadata should not contain matter-system metadata.")
        for value in payload.values():
            assert_no_field_user_metadata(value)
    elif isinstance(payload, list):
        for value in payload:
            assert_no_field_user_metadata(value)


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    user_metadata = {
        "example": "metadata_schema_validation",
        "transitions_eV": {"0_to_1": 1.5, "0_to_2": 1.75},
        "dipoles_D": {"0_to_1": 3.0, "0_to_2": 2.0},
    }

    n2_meta, _n2_debug = run_saved_case(
        "n2_lab_exact",
        NLevelPhysicalParams(
            basis=("g", "e"),
            energies_eV=(0.0, 1.55),
            dipole_matrix_D=((0.0, 3.0), (3.0, 0.0)),
            t_start_fs=0.0,
            t_end_fs=0.2,
            dt_fs=0.1,
            field=make_default_carrier_field(E0_MV_per_cm=0.01, laser_energy_eV=1.55),
            input_description="N=2 metadata validation.",
            input_metadata=user_metadata,
        ),
    )
    if "energy_gap_eV" not in n2_meta["system"]:
        raise AssertionError("N=2 system should keep energy_gap_eV compatibility field.")
    assert_no_field_user_metadata(n2_meta["field"])

    n3_meta, n3_debug = run_saved_case(
        "n3_lab_exact",
        NLevelPhysicalParams(
            basis=("g", "e1", "e2"),
            energies_eV=(0.0, 1.5, 1.75),
            dipole_matrix_D=((0.0, 3.0, 2.0), (3.0, 0.0, 0.5), (2.0, 0.5, 0.0)),
            t_start_fs=0.0,
            t_end_fs=0.2,
            dt_fs=0.1,
            field=make_default_carrier_field(E0_MV_per_cm=0.01, laser_energy_eV=1.625),
            input_description="N=3 metadata validation.",
            input_metadata=user_metadata,
        ),
    )
    system = n3_meta["system"]
    if "energy_gap_eV" in system or "detuning_eV" in system:
        raise AssertionError("N=3 system should not expose singular energy_gap_eV/detuning_eV.")
    if len(system["transition_table"]) != 3:
        raise AssertionError("N=3 transition_table should contain all i<j transitions.")
    if "derived_physical" in n3_meta or "input_field" in n3_meta or "input_field_rebuild" in n3_meta:
        raise AssertionError("meta.json still contains old verbose metadata blocks.")
    if "lab_frame_solver" in n3_meta or "solver_representation" in n3_meta:
        raise AssertionError("meta.json should use compact solver block only.")
    if "input_drive" in n3_meta:
        raise AssertionError("lab_exact meta.json should not contain input_drive.")
    assert_no_field_user_metadata(n3_meta["field"])
    output_files = n3_meta["output_files"]
    if "component_figures_dir" not in output_files or "component_figures" not in output_files:
        raise AssertionError("output_files should record generated component figures.")
    if "rho_01" not in output_files["component_figures"]:
        raise AssertionError("component_figures should include rho_01.")

    drive_dict = n3_debug.get("drive_dict", {})
    if "source_field" in drive_dict:
        assert_no_field_user_metadata(drive_dict["source_field"])
    field = n3_debug.get("parameters_code", {}).get("field")
    if isinstance(field, dict) and "source_field" in field:
        assert_no_field_user_metadata(field["source_field"])

    print("metadata schema compact meta ok")
    print("transition_table multilevel ok")
    print("component output_files ok")
    print("debug field metadata cleanup ok")


if __name__ == "__main__":
    main()
