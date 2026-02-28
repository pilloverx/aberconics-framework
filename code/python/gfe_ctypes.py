"""Minimal ctypes wrapper for gfe_c_api.h."""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, byref, c_char, c_double, c_int, c_size_t
from pathlib import Path
from typing import Dict, List


GFE_C_STATUS_OK = 0
GFE_C_STATUS_INVALID_ARGUMENT = 1
GFE_C_STATUS_RUNTIME_ERROR = 2
GFE_C_STATUS_BUFFER_TOO_SMALL = 3

GFE_C_FIT_BACKEND_NNLS_PG = 0
GFE_C_FIT_BACKEND_PRONY = 1

GFE_C_COMPUTE_BACKEND_CPU_REF = 0
GFE_C_COMPUTE_BACKEND_CPU_BLAS = 1
GFE_C_COMPUTE_BACKEND_GPU_CUDA = 2
GFE_C_COMPUTE_BACKEND_GPU_HIP = 3
GFE_C_COMPUTE_BACKEND_GPU_SYCL = 4


class GfeCSoeFitOptions(ctypes.Structure):
    _fields_ = [
        ("backend", c_int),
        ("n_basis", c_int),
        ("gamma_min", c_double),
        ("gamma_max", c_double),
        ("threshold", c_double),
        ("lr", c_double),
        ("iters", c_int),
        ("prony_order", c_int),
        ("prony_ridge", c_double),
        ("prony_root_imag_tol", c_double),
        ("prony_root_min", c_double),
        ("prony_root_max", c_double),
        ("prony_enforce_nonnegative_weights", c_int),
        ("prony_max_root_iters", c_int),
        ("prony_root_tol", c_double),
    ]


class GfeCBackendConfig(ctypes.Structure):
    _fields_ = [
        ("preferred", c_int),
        ("allow_fallback", c_int),
        ("fallback", c_int),
    ]


class GfeCFitResultView(ctypes.Structure):
    _fields_ = [
        ("gamma_fit", POINTER(c_double)),
        ("gamma_fit_capacity", c_size_t),
        ("gamma_fit_size", POINTER(c_size_t)),
        ("w_fit", POINTER(c_double)),
        ("w_fit_capacity", c_size_t),
        ("w_fit_size", POINTER(c_size_t)),
        ("fit", POINTER(c_double)),
        ("fit_capacity", c_size_t),
        ("fit_size", POINTER(c_size_t)),
    ]


def _default_library_path() -> Path:
    ext = {"linux": "so", "darwin": "dylib", "win32": "dll"}
    suffix = ext.get(os.sys.platform, "so")
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "code" / "c++ core" / "build-shared" / f"libgfe_core.{suffix}"


def load_gfe_library(path: str | None = None) -> ctypes.CDLL:
    lib_path = path or os.environ.get("GFE_CORE_LIB")
    if not lib_path:
        lib_path = str(_default_library_path())
    lib = ctypes.CDLL(lib_path)

    lib.gfe_c_default_soe_fit_options.argtypes = []
    lib.gfe_c_default_soe_fit_options.restype = GfeCSoeFitOptions

    lib.gfe_c_default_backend_config.argtypes = []
    lib.gfe_c_default_backend_config.restype = GfeCBackendConfig

    lib.gfe_c_fit_soe_kernel.argtypes = [
        POINTER(c_double),
        c_size_t,
        POINTER(c_double),
        c_size_t,
        POINTER(GfeCSoeFitOptions),
        POINTER(GfeCBackendConfig),
        POINTER(GfeCFitResultView),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_fit_soe_kernel.restype = c_int
    return lib


def fit_soe_kernel(lib: ctypes.CDLL, t: List[float], data: List[float]) -> Dict[str, List[float]]:
    if len(t) != len(data):
        raise ValueError("t and data must have the same length")
    n = len(t)
    if n == 0:
        raise ValueError("t and data must be non-empty")

    t_arr = (c_double * n)(*t)
    d_arr = (c_double * n)(*data)

    opts = lib.gfe_c_default_soe_fit_options()
    opts.backend = GFE_C_FIT_BACKEND_NNLS_PG
    opts.n_basis = 16
    opts.threshold = 0.0
    opts.iters = 5000

    backend = lib.gfe_c_default_backend_config()
    backend.preferred = GFE_C_COMPUTE_BACKEND_GPU_CUDA
    backend.allow_fallback = 1
    backend.fallback = GFE_C_COMPUTE_BACKEND_CPU_REF

    cap = max(64, n)
    gamma_buf = (c_double * cap)()
    w_buf = (c_double * cap)()
    fit_buf = (c_double * n)()
    gamma_size = c_size_t(0)
    w_size = c_size_t(0)
    fit_size = c_size_t(0)

    gamma_size_p = ctypes.pointer(gamma_size)
    w_size_p = ctypes.pointer(w_size)
    fit_size_p = ctypes.pointer(fit_size)

    out = GfeCFitResultView(
        gamma_buf,
        cap,
        gamma_size_p,
        w_buf,
        cap,
        w_size_p,
        fit_buf,
        n,
        fit_size_p,
    )
    err = ctypes.create_string_buffer(512)

    status = lib.gfe_c_fit_soe_kernel(
        t_arr,
        n,
        d_arr,
        n,
        byref(opts),
        byref(backend),
        byref(out),
        err,
        ctypes.sizeof(err),
    )
    if status != GFE_C_STATUS_OK:
        raise RuntimeError(f"gfe_c_fit_soe_kernel failed ({status}): {err.value.decode('utf-8')}")

    g = [gamma_buf[i] for i in range(gamma_size.value)]
    w = [w_buf[i] for i in range(w_size.value)]
    fit = [fit_buf[i] for i in range(fit_size.value)]
    return {"gamma_fit": g, "w_fit": w, "fit": fit}


if __name__ == "__main__":
    lib = load_gfe_library()
    t_vals = [0.1 * i for i in range(81)]
    y_vals = [0.65 * (2.718281828459045 ** (-0.8 * t)) + 0.35 * (2.718281828459045 ** (-0.12 * t)) for t in t_vals]
    result = fit_soe_kernel(lib, t_vals, y_vals)
    print("ctypes call ok")
    print(f"modes={len(result['gamma_fit'])} fit_len={len(result['fit'])}")
