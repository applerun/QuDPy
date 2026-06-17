#!/usr/bin/env python3
"""快速验证 complex-valued dipole metadata JSON safety。

这个脚本只做 smoke test：确认 real / complex `dipole_matrix_D` 都能保存
metadata JSON，并确认 complex 不被转成字符串或丢掉虚部。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sjh_learn.utils.core import NLevelPhysicalParams, ParaNormalizer, run_case
from sjh_learn.utils.fields import make_default_carrier_field
from sjh_learn.utils.io import make_json_safe, save_result_case


OUT_ROOT = Path("/private/tmp/qudpy_validate_complex_metadata")


@dataclass(frozen=True)
class ComplexPayload:
    value: complex
    array: np.ndarray


def make_params(dipole_matrix_D) -> NLevelPhysicalParams:
    return NLevelPhysicalParams(
        basis=("g", "e"),
        energies_eV=(0.0, 1.55),
        dipole_matrix_D=dipole_matrix_D,
        t_start_fs=0.0,
        t_end_fs=0.2,
        dt_fs=0.1,
        field=make_default_carrier_field(E0_MV_per_cm=0.01, laser_energy_eV=1.55),
    )


def run_and_save(case_name: str, dipole_matrix_D) -> Path:
    normalizer = ParaNormalizer(time_scale_fs=1.0, auto_scale=False)
    result = run_case(make_params(dipole_matrix_D), normalizer=normalizer)
    saved = save_result_case(
        result,
        OUT_ROOT,
        case_name=case_name,
        output_data=True,
        output_preview=False,
        save_npz=True,
        save_csv=True,
        save_json=True,
        append_results_csv=False,
    )
    return saved["case_dir"]


def assert_complex_object(value, *, real: float, imag: float) -> None:
    if not isinstance(value, dict):
        raise AssertionError(f"complex value should be a JSON object, got {type(value).__name__}: {value!r}")
    if set(value) != {"real", "imag"}:
        raise AssertionError(f"complex JSON object keys should be real/imag, got {value!r}")
    if not np.isclose(value["real"], real):
        raise AssertionError(f"real part mismatch: expected {real}, got {value['real']}")
    if not np.isclose(value["imag"], imag):
        raise AssertionError(f"imag part mismatch: expected {imag}, got {value['imag']}")


def check_json_safety_recursion() -> None:
    payload = {
        "python_complex": 1.0 + 2.0j,
        "numpy_complex_scalar": np.complex128(3.0 - 4.0j),
        "list": [5.0 + 6.0j],
        "tuple": (np.complex64(7.0 - 8.0j),),
        "dict": {"nested": -1.0 + 0.5j},
        "dataclass": ComplexPayload(
            value=9.0 + 10.0j,
            array=np.asarray([[11.0 - 12.0j, 13.0 + 0.0j]], dtype=np.complex128),
        ),
        "complex_array_1d": np.asarray([14.0 + 15.0j], dtype=np.complex128),
        "real_array": np.asarray([1.0, 2.0], dtype=float),
    }
    safe = make_json_safe(payload)
    json.dumps(safe)

    assert_complex_object(safe["python_complex"], real=1.0, imag=2.0)
    assert_complex_object(safe["numpy_complex_scalar"], real=3.0, imag=-4.0)
    assert_complex_object(safe["list"][0], real=5.0, imag=6.0)
    assert_complex_object(safe["tuple"][0], real=7.0, imag=-8.0)
    assert_complex_object(safe["dict"]["nested"], real=-1.0, imag=0.5)
    assert_complex_object(safe["dataclass"]["value"], real=9.0, imag=10.0)
    assert_complex_object(safe["dataclass"]["array"][0][0], real=11.0, imag=-12.0)
    assert_complex_object(safe["complex_array_1d"][0], real=14.0, imag=15.0)
    if safe["real_array"] != [1.0, 2.0]:
        raise AssertionError(f"real ndarray output changed unexpectedly: {safe['real_array']!r}")
    print("complex metadata json safety ok")


def check_save_result_case() -> None:
    real_case = run_and_save("real_dipole", ((0.0, 3.0), (3.0, 0.0)))
    real_meta = json.loads((real_case / "meta.json").read_text(encoding="utf-8"))
    real_mu_ge = real_meta["system"]["dipole_matrix_D"][0][1]
    if isinstance(real_mu_ge, dict) or not np.isclose(real_mu_ge, 3.0):
        raise AssertionError(f"real-valued dipole metadata should remain numeric, got {real_mu_ge!r}")

    phase = np.pi / 5.0
    mu_ge = 3.0 * np.exp(1j * phase)
    mu_eg = np.conjugate(mu_ge)
    complex_case = run_and_save("complex_transition_dipole", ((0.0, mu_ge), (mu_eg, 0.0)))
    complex_meta = json.loads((complex_case / "meta.json").read_text(encoding="utf-8"))
    complex_debug = json.loads((complex_case / "debug_meta.json").read_text(encoding="utf-8"))

    meta_mu_ge = complex_meta["system"]["dipole_matrix_D"][0][1]
    debug_mu_ge = complex_debug["physical_params"]["dipole_matrix_D"][0][1]
    assert_complex_object(meta_mu_ge, real=float(mu_ge.real), imag=float(mu_ge.imag))
    assert_complex_object(debug_mu_ge, real=float(mu_ge.real), imag=float(mu_ge.imag))
    print("complex dipole save_result_case ok")


def expect_value_error(dipole_matrix_D, label: str) -> None:
    try:
        ParaNormalizer(time_scale_fs=1.0, auto_scale=False).normalize(make_params(dipole_matrix_D))
    except ValueError:
        return
    raise AssertionError(f"{label} should raise ValueError.")


def check_invalid_dipoles() -> None:
    expect_value_error(((0.0, 1.0 + 1.0j), (1.0 + 1.0j, 0.0)), "non-Hermitian dipole")
    expect_value_error(((1.0j, 3.0), (3.0, 0.0)), "complex diagonal dipole")
    print("invalid dipole checks ok")


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    check_json_safety_recursion()
    check_save_result_case()
    check_invalid_dipoles()


if __name__ == "__main__":
    main()
