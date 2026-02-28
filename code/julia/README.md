# Julia Usage

This directory contains:
- `src/GFE.jl`: Julia-native GFE reference module
- `src/GFE_CAPI.jl`: minimal Julia `ccall` wrapper for the C ABI
- `examples/`: experiments and wrapper smoke scripts

## Julia-native tests

```bash
cd code/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. src/test/runtests.jl
```

## C-ABI wrapper smoke

Build shared library first:

```bash
cmake -S "code/c++ core" -B "code/c++ core/build-shared" -DBUILD_SHARED_LIBS=ON
cmake --build "code/c++ core/build-shared" -j
```

Then run:

```bash
export GFE_CORE_LIB="$(pwd)/code/c++ core/build-shared/libgfe_core.so"
julia code/julia/examples/05_capi_ccall_smoke.jl
```

Wrapper validation test is included in `src/test/runtests.jl` and runs when `GFE_CORE_LIB` is set to a valid shared library.
