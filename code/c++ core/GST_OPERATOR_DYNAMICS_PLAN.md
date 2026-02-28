# GST + Operator-Coupled Augmented Dynamics Plan

## Goal
Extend the current C++ SOE toolkit into a dynamics layer aligned with generalized GFE structure:
- GST layer (geometric spectral transform abstraction)
- Operator-coupled auxiliary memory channels
- Stable coupling modes (Formulation A/B/C-ready)

Out of scope for first implementation phase:
- theorem-assumption tooling (kept as placeholder interfaces)
- energy/Hamiltonian bookkeeping (kept as placeholder interfaces)

## Design Principles
- Keep core numerics decoupled from application physics.
- Separate spatial/operator side (`L[u]`, optional GST) from temporal memory side (`chi` channels).
- Make coupling explicit via policy enums to avoid accidental unstable sign conventions.
- Keep API minimal and composable so Aberconics and non-Aberconics apps can share it.

## Proposed Module Split

### 1) `gfe_core` (already exists)
- SOE fitting and spectral metrics.
- Parameter packing/unpacking.

### 2) `gfe_dynamics` (new)
- Time-stepping of augmented local system:
  - base state `u`
  - memory channels `chi[l]`
- Operator callbacks and coupling policies.
- Integrator-neutral stepping interface (explicit Euler / RK start; extend later).

### 3) `gfe_gst` (new abstraction layer)
- Optional GST adapter for applications that use field-dependent spectral bases.
- No hard dependency on a specific eigensolver in phase 1.

## Proposed Core Types

```cpp
namespace gfe {

using State = std::vector<double>;

struct DynamicsConfig {
    double dt;
    std::size_t memory_channels;
};

enum class CouplingForm {
    InputDriven,       // Formulation A
    NegativeFeedback,  // Formulation B
    ResonantSecondOrder // Formulation C
};

struct MemoryKernelParams {
    Vec gamma; // size L, positive
    Vec w;     // size L
};

struct StepResult {
    State u_next;
    Vec chi_next;
};

} // namespace gfe
```

## Operator-Coupled API

Implement dynamics with user-supplied operators:

```cpp
struct OperatorCallbacks {
    // Core operator term L[u] in theorem notation.
    std::function<State(const State& u, double t)> apply_L;

    // Optional nonlinear term N[u] (can be zero).
    std::function<State(const State& u, double t)> apply_N;

    // External forcing/input x(t) or f(t).
    std::function<State(double t)> forcing;
};
```

For scalar models, support lightweight overloads using `double`.

## Memory Channel Update (Unified)

General per-channel update:

\[
\dot{\chi}_\ell = -\gamma_\ell \chi_\ell + drive_\ell(u,t)
\]

Where `drive` depends on coupling policy:
- `InputDriven`: `drive = forcing` (or external input signal)
- `NegativeFeedback`: `drive = u` or selected state projection
- `ResonantSecondOrder`: drive coupled to resonant state variable (e.g., `u` or velocity term depending on model)

## Base-State Update Policy

Define canonical update function:

\[
\dot{u} = F_{\text{base}}(u,t) + \sum_\ell c_\ell(\chi_\ell)
\]

with policy-specific coupling:
- Formulation A: `+ sum(w_l * chi_l)` with input-driven `chi`
- Formulation B: `- sum(w_l * chi_l)` (dissipative)
- Formulation C: second-order state decomposition (introduce velocity state `v`)

## GST Integration Strategy

Phase 1: GST as adapter interface, not full geometric solver.

```cpp
struct GSTAdapter {
    // Project physical state to modal coefficients.
    virtual State project(const State& u, double t) const = 0;

    // Reconstruct physical state from modal coefficients.
    virtual State reconstruct(const State& coeffs, double t) const = 0;

    virtual ~GSTAdapter() = default;
};
```

How it is used:
1. if GST present: project `u -> coeffs`
2. apply operator in modal/physical form per app
3. reconstruct as needed before coupling and output

This keeps the framework ready for Laplace-Beltrami/evolving-basis implementations later.

## Placeholders (Explicitly Deferred)

### Theorem-assumption tooling placeholder
Create non-blocking validation hooks:

```cpp
struct AssumptionChecks {
    bool kernel_approximable = true; // placeholder
    bool regularity_ok = true;       // placeholder
    std::string note;
};
```

No hard runtime gating in phase 1.

### Energy/Hamiltonian bookkeeping placeholder
Create optional observer hook:

```cpp
struct EnergyObserver {
    virtual void on_step(double t, const State& u, const Vec& chi) {}
    virtual ~EnergyObserver() = default;
};
```

Default behavior: no-op.

## Minimal Implementation Sequence

1. Add `gfe_dynamics.hpp/.cpp`
- implement `step_augmented(...)` for scalar/vector states
- implement Formulations A and B first

2. Add `CouplingForm` guardrails
- default to `NegativeFeedback`
- require explicit opt-in for risky couplings in advanced APIs

3. Add `GSTAdapter` optional path
- null adapter = plain state-space dynamics
- adapter path only transforms state representation

4. Add tests
- deterministic toy systems for A/B/C behavior
- sign-check tests ensuring Formulation B uses dissipative sign
- regression tests on small Lorenz-like reduced examples

5. Add placeholders
- `AssumptionChecks` and `EnergyObserver` non-blocking hooks

## Risks and Mitigations
- Risk: API over-generalization too early.
  - Mitigation: ship A/B first; keep C and GST adapter optional.
- Risk: sign mistakes in coupling.
  - Mitigation: encode coupling forms as enums, not free-form booleans.
- Risk: mismatch with future theorem tooling.
  - Mitigation: placeholder structs now; evolve without breaking runtime API.

## Deliverable Definition for Next Iteration
- New `gfe_dynamics` API committed.
- Formulation A/B stepping implemented and tested.
- `GSTAdapter` interface present with one identity adapter.
- Placeholder hooks compiled and documented, but not mandatory.
