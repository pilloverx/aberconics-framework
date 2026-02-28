from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gfe_ctypes import fit_soe_kernel, load_gfe_library


def _lib_path() -> str | None:
    return None


@pytest.mark.smoke
def test_ctypes_fit_call():
    try:
        lib = load_gfe_library(_lib_path())
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")

    t = [0.1 * i for i in range(81)]
    y = [0.65 * (2.718281828459045 ** (-0.8 * x)) + 0.35 * (2.718281828459045 ** (-0.12 * x)) for x in t]
    out = fit_soe_kernel(lib, t, y)

    assert len(out["fit"]) == len(t)
    assert len(out["gamma_fit"]) > 0
    assert len(out["w_fit"]) == len(out["gamma_fit"])
    assert all(v == v for v in out["fit"])  # NaN check without extra deps
