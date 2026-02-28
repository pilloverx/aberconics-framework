# Python Wrapper (ctypes)

This folder provides a minimal Python `ctypes` wrapper over the C ABI in:
- `code/c++ core/include/gfe/gfe_c_api.h`

## Files

- `gfe_ctypes.py`: typed `ctypes` structs/signatures + `fit_soe_kernel(...)` helper
- `ctypes_smoke.py`: proof-of-call script
- `tests/test_gfe_ctypes.py`: tiny pytest smoke test

## Build Shared Library

```bash
cmake -S "code/c++ core" -B "code/c++ core/build-shared" -DBUILD_SHARED_LIBS=ON
cmake --build "code/c++ core/build-shared" -j
```

## Run Smoke Script

```bash
export GFE_CORE_LIB="$(pwd)/code/c++ core/build-shared/libgfe_core.so"
python3 code/python/ctypes_smoke.py
```

## Run Pytest

```bash
python3 -m pip install pytest
export GFE_CORE_LIB="$(pwd)/code/c++ core/build-shared/libgfe_core.so"
pytest -q code/python/tests
```

## Minimal Example

```python
from gfe_ctypes import load_gfe_library, fit_soe_kernel

lib = load_gfe_library()  # uses GFE_CORE_LIB or default build-shared path
t = [0.1 * i for i in range(81)]
y = [0.65 * (2.718281828459045 ** (-0.8 * x)) + 0.35 * (2.718281828459045 ** (-0.12 * x)) for x in t]
out = fit_soe_kernel(lib, t, y)
print(len(out["gamma_fit"]), len(out["fit"]))
```
