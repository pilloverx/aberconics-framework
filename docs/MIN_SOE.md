Aberconics / GFE — Hierarchical Complexity & Renormalisation Flow
HIERARCHICAL COMPLEXITY
IN BIOLOGICAL SYSTEMS
How the Aberconics / GFE Framework Handles Multi-Scale Memory
D. K. Ahorlu
GFE / GGFE / Aberconics Framework
Abstract
Biological systems exhibit non-Markovian dynamics across at least eight orders of magnitude
in time — from femtosecond bond vibrations to multi-day homeostatic rhythms. Each level of
organisation carries its own memory structure, and these levels interact in a nested,
bidirectional hierarchy. This document formalises how the Aberconics / GFE framework
addresses this challenge through three complementary mechanisms: (i) Sum-of-Exponentials
embedding of memory kernels at each level, supported by the complete monotonicity (CM)
condition that guarantees physical admissibility; (ii) a hierarchical Memory-Integrated
Network (MIN) architecture with an explicit coupling assembly rule governing both bottom-up
and top-down interactions; and (iii) a renormalisation flow connecting levels, under which CM
and the SOE structure are preserved. The effective dimension D_eff is proposed as a
conserved-but-renormalised quantity that tracks memory complexity as one coarse-grains
across scales. Together, these elements yield a unified, mathematically rigorous, and
computationally tractable language for multi-scale biological modelling.
1. The Physical Foundation: Why CM Holds at Every Level
Before introducing the hierarchical architecture, we must establish why the framework's central
mathematical property — complete monotonicity of the memory kernel — propagates up the
biological hierarchy rather than being an artefact of the molecular-level derivation.
1.1 Complete Monotonicity as a Stability Condition
In the Mori-Zwanzig derivation of the GLE at the molecular level, the memory kernel K(t) is
provably CM because it is the autocorrelation function of a bath that is Gaussian, stationary, and in
thermal equilibrium. The fluctuation-dissipation theorem ensures K(t) = (1/kT) F(t), where F(t) is
the random force autocorrelation — which is positive definite by construction.
D. K. Ahorlu | GFE Framework | p. 1Aberconics / GFE — Hierarchical Complexity & Renormalisation Flow
At higher levels — cells, tissues, organisms — the bath is neither in equilibrium nor Gaussian. The
FDT breaks down. The standard CM proof no longer applies. Yet biological regulatory systems
return to homeostasis after perturbation. This stability is the key:
Theorem (CM from Stability):
Let x(t) satisfy a stable linear system near a fixed point with response kernel K(t). Then K(t) is
the Laplace transform of a positive Borel measure on [0,∞) — i.e., K(t) is completely
monotone.
Proof sketch: K(t) is the impulse response of a stable, causal, passive system. By
Bochner-Bernstein, a function is the Laplace transform of a positive measure iff it is CM.
Stability (all poles in left half-plane) combined with passivity (non-negative energy
dissipation) implies exactly this.
The biological implication is direct: pathological states — cancer, neurodegeneration, arrhythmia —
are precisely states where stability breaks down and CM fails. The CM condition is therefore not just
a mathematical convenience. It is the mathematical signature of functional biological organisation.
1.2 SOE Embedding and the Auxiliary Variable System
By Bernstein's representation theorem, every CM kernel admits the decomposition:
K(t) = ∫0∞ e−rt dμ(γ) ≈ Σl wl e−γlt
where μ is a positive measure (the spectral measure) and the approximation replaces it with a finite
atomic measure at rates {γl}. This is not merely a fitting trick. Each exponential term corresponds to
an auxiliary variable χl(t) satisfying:
dχl/dt = -γlχl + u(t)
The GLE with memory kernel K(t) is therefore exactly equivalent to a finite-dimensional Markovian
system in the extended state space (u, χ1, χ2, ..., χ ). The dimension of this extension is L, the
number of SOE terms. The memory has been made explicit.
The effective memory dimension quantifies how many independent channels are active:
D_eff = L · exp(H_mem)
where
H_mem = -Σl w̃l log w̃l
with w̃l = wl / Σ wl. D_eff = 1 for a single-exponential (Markovian) kernel. D_eff = L for a uniform
spectrum (maximally non-Markovian). Power-law kernels drive D_eff → ∞ as observation time
grows, signalling infinite-dimensional memory.
2. The Duality Theorem: Why Non-Markovian Models Are Complete
The theoretical justification for the entire programme rests on a duality result established in GFE II
(Theorem 4.4). This should be understood as the foundational result from which all applications
follow.
D. K. Ahorlu | GFE Framework | p. 2Aberconics / GFE — Hierarchical Complexity & Renormalisation Flow
Theorem 4.4 (GFE II — Nonlinearity-Memory Duality):
(i) Reduction: Any Lipschitz nonlinear Markovian system on a high-dimensional state space,
when projected onto a lower-dimensional observable subspace, generates a linear
non-Markovian system with a CM memory kernel. The kernel encodes all information about
the eliminated degrees of freedom that is observable from the projected subspace.
(ii) Embedding: Conversely, any linear non-Markovian system with CM kernel K(t) can be
embedded into a higher-dimensional linear Markovian system via the SOE auxiliary variable
construction. The embedding is exact when L is finite and K(t) has a finite atomic spectral
measure.
Corollary: The non-Markovian description is not an approximation of the Markovian one — it
is exactly equivalent, with the memory kernel carrying all discarded information about hidden
degrees of freedom.
The biological interpretation is profound. When we observe a protein conformational change, a
synaptic response, or a gene regulatory circuit, we are necessarily working in a projected subspace —
we cannot track every atom or every signalling molecule. Theorem 4.4 guarantees that the memory
kernel of our observable is the exact reduced description of the full system. Fitting K(t) from data is
therefore not a phenomenological act — it is recovering the exact projected dynamics.
This also resolves why cooperative binding, neural firing thresholds, and gene regulatory bistability
can all be described within the linear non-Markovian framework. Each nonlinear phenomenon at
the full level appears as a specific structure in K(t) at the projected level. The nonlinearity is folded
into the memory, not discarded.
3. The Biological Hierarchy and Its Memory Structure
The following table characterises the memory kernel at each level of biological organisation. The
D_eff values marked with † are theoretical predictions to be validated; those with ‡ are derived from
or consistent with existing experimental data.
Level
Timescale
Memory Source
D_eff
Observable
Bond / solventfs–psVelocity autocorr.1–3 ‡MD FACF
Residue /
backboneps–nsDihedral
transitions3–10 ‡NMR R1/R2/NOE
Domain / loopns–μsConformational
exch.5–15 ‡NMR Rex, CPMG
Protein
complexμs–msAssembly/allostery8–25 †smFRET, SPR
Ion channelms–sOpen/close/inactiv
ate2–6 †Patch clamp
Neuronms–100sSynaptic + intrinsic10–25 †Intracellular
D. K. Ahorlu | GFE Framework | p. 3Aberconics / GFE — Hierarchical Complexity & Renormalisation Flow
Level
Timescale
Memory Source
D_eff
Observable
Neural
population100ms–minPlasticity,
adaptation25–100 †LFP, EEG
Cell /
signallings–minFeedback loops15–50 †Live-cell imaging
Tissue / organmin–hrsMechanical
memory20–80 †Rheology
Organismhrs–daysHomeostatic
feedback≤80† per pathwayPhysiology
‡ Derived from or consistent with NMR/MD data in the literature. † Theoretical predictions of the renormalisation flow.
4. Hierarchical MIN Architecture with Explicit Coupling
We formalise the hierarchical Memory-Integrated Network as a nested operator structure.
Critically, we specify the coupling assembly rule — the mechanism by which the slow (outer) level
modulates the fast (inner) level's dynamics, which was the principal gap in previous formulations.
4.1 The Nested Volterra Structure
A hierarchical MIN is defined as a composition of causal integral operators:
M[f] = Σl ∏_j νl,j,1 (νl,j,2 (⋯ νl,j,r [f] ⋯ ))
where each ν is a causal Volterra operator with its own CM kernel K_ν(t). The inner layers
correspond to fast processes (ion channel fluctuations, fast backbone dynamics); the outer layers
correspond to slow processes (synaptic plasticity, gene regulation). SOE embedding converts each ν
into a finite-dimensional ODE, making the full hierarchy computationally tractable.
4.2 Bottom-Up Coupling: Fast Drives Slow
At level n, the coarse-grained observable X_n evolves under a GLE whose forcing term includes the
output of the level below:
dX_n/dt = -∇V_n(X_n) - Σl wln χln + F_n(X_{n-1}, t)
where F_n is a coupling function of the level-(n-1) output, and {χln} are the level-n auxiliary
variables. This is bottom-up coupling: fast processes create the effective forcing and noise at the
slow level. It is the standard Mori-Zwanzig coarse-graining, and it preserves CM at level n provided
K_{n-1}(t) is CM.
4.3 Top-Down Coupling: Slow Modulates Fast — The Assembly Rule
Top-down modulation is the harder case and the one previously unspecified. The slow variable
X_n(t) modulates the memory kernel of the fast level n-1. We formalise this through a
parameter-dependent kernel:
K_{n-1}(t ; X_n) = Σl wl(X_n) · e^{-γl(X_n) t}
D. K. Ahorlu | GFE Framework | p. 4Aberconics / GFE — Hierarchical Complexity & Renormalisation Flow
Both the weights wl and the rates γl at level n-1 are functions of the slow variable X_n. This is the
biologically correct picture: the slow conformational state of a protein modulates the fast backbone
fluctuations; the slow gene regulatory state modulates the fast ion channel kinetics; the slow
mechanical state of tissue modulates the fast cellular dynamics.
The CM condition requires that for each fixed value of X_n, the kernel K_{n-1}(t; X_n) remains
CM. This imposes the constraint:
wl(X_n) ≥ 0
and
γl(X_n) > 0
for all X_n
These are simple inequality constraints on the coupling functions. They can be enforced by
parameterising wl = exp(αl(X_n)) and γl = exp(βl(X_n)) with unconstrained α, β — or by NNLS at
each update step. The result is a CM-preserving top-down coupling that is both biologically
meaningful and mathematically consistent.
Coupling Assembly Rule (complete specification):
Level n drives level n-1 via forcing: F_{n-1}(t) ← g(X_n(t))
Level n modulates level n-1 kernel: K_{n-1}(t) ← K_{n-1}(t; X_n(t))
Level n-1 drives level n via output: F_n(t) = h(X_{n-1}(t))
Coupling constraints:
wl(X_n) ≥ 0, γl(X_n) > 0 (CM preserved)
g and h are learned from data using local Hebbian rules (cross-level correlation).
The full hierarchy is a coupled ODE system with total dimension Σ_n L_n.
5. Renormalisation Flow Across the Biological Hierarchy
This is the new theoretical contribution that was absent from previous formulations. We establish a
precise mathematical relationship between the memory kernel at level n and the kernel at level n+1,
and show that this relationship constitutes a renormalisation group flow in the space of CM kernels.
5.1 The Coarse-Graining Operation
Moving from level n to level n+1 involves integrating out the fast degrees of freedom at level n,
retaining only the slow observable X_n. This is a coarse-graining or Mori-Zwanzig projection. The
resulting kernel at level n+1 is:
K_{n+1}(t) = K_n(t) ∗ K_bathn(t) + K_directn(t)
where * denotes temporal convolution, K_bathn(t) is the bath correlation function at level n, and
K_directn(t) captures direct memory contributions that survive the coarse-graining. This is the
renormalisation map R_n:
R_n : K_n ↦ K_{n+1} = K_n ∗ K_bathn + K_directn
5.2 Preservation of CM Under Coarse-Graining
The key structural result justifying the use of the same framework at every level:
D. K. Ahorlu | GFE Framework | p. 5Aberconics / GFE — Hierarchical Complexity & Renormalisation Flow
Proposition (CM is preserved under R_n):
If K_n(t) is CM and K_bathn(t) is CM, then K_{n+1}(t) = K_n ∗ K_bathn is CM.
Proof: The convolution of two CM functions is CM. This follows from the fact that the
Bernstein class is closed under pointwise products (since K_n(t) · K_bathn(t) is CM for CM
functions), and convolution corresponds to multiplication of Laplace transforms. The product
of two completely monotone Laplace transforms is completely monotone. □
Corollary: CM propagates upward through the entire biological hierarchy. Provided each level
is individually stable (its bath is CM), the effective kernel at every higher level is also CM.
5.3 Renormalisation of SOE Parameters
Under R_n, the SOE parameters transform in a computable way. If K_n(t) = Σl wl exp(-γl t) and
K_bathn(t) = Σ_m u_m exp(-δ_m t), then their convolution is:
K_{n+1}(t) = Σl Σ_m (wl u_m / (γl - δ_m)) · [e^{-δ_m t} - e^{-γl t}]
≠ δ_m)
(γl
This is still a sum of exponentials with L × M terms — but many will have small weights and can be
pruned. The effective number of channels at level n+1 is therefore at most L × M, but typically much
smaller after pruning. The rates at level n+1 are the union of {-γl} and {-δ_m} — the slow bath rates
dominate, the fast signal rates are suppressed.
This is the renormalisation flow in parameter space: each level inherits the slow channels from the
level below and generates new slow channels from the bath correlation. Fast channels are integrated
out. The SOE spectrum shifts toward slower timescales as we ascend the hierarchy.
5.4 Renormalisation of D_eff
How does D_eff transform under R_n? This is the central diagnostic of the renormalisation flow.
Because K_{n+1} has at most L × M channels but typically fewer, and because the weight
distribution shifts toward the new slow channels, we can derive bounds:
D_effn+1 ≤ D_effn · D_eff(K_bathn)
with equality when all channels have identical weight (maximum entropy case). In biological
systems, the bath correlation is typically dominated by a small number of slow modes —
D_eff(K_bath) ~ 2-5. So D_eff grows sub-multiplicatively with each level, consistent with the
observed moderate D_eff values at each biological scale.
Critically, D_eff at level n+1 is not simply D_eff at level n plus new channels. It is a renormalised
quantity whose entropy reflects the new weight distribution after coarse-graining. Channels that
were nearly degenerate at level n may merge into a single effective channel at level n+1, reducing
D_eff. New slow channels from the bath may increase it.
Summary: Renormalisation Flow Properties
D. K. Ahorlu | GFE Framework | p. 6Aberconics / GFE — Hierarchical Complexity & Renormalisation Flow
1. CM is preserved at every level (Proposition above).
2. SOE structure is preserved at every level (convolution of SOEs is SOE).
3. Fast channels are suppressed; slow channels accumulate upward.
4. D_eff grows sub-multiplicatively: D_eff^{n+1} ≤ D_eff^n · D_eff(bath^n).
5. Pathological states (cancer, neurodegeneration) correspond to fixed points or divergences
of the flow where D_eff grows without bound or collapses to 1.
6. Spectral Units as Common Currency Across Scales
The spectral units derived from K(t) provide a universal vocabulary that allows comparison across
levels, systems, and experimental modalities. They are well-defined at every level of the hierarchy
and transform predictably under the renormalisation flow.
Spectral Unit
Definition and Biological Meaning
D_effEffective number of independent memory channels. Quantifies
dynamical complexity. Conserved-but-renormalised across
levels.
M_cap (mean depth)Mean memory timescale Σ w̃l/γl. Sets the dominant integration
window of the system. Shifts to slower values at higher levels.
M_scale (span)log10(γ_max / γ_min). Number of decades covered. Grows
monotonically up the hierarchy.
H_mem (entropy)Shannon entropy of normalised weights. Maximum when all
channels contribute equally. High H_mem = rich, distributed
memory.
d_s (fractal exponent)Spectral dimension from FACF power-law tail. d_s < 1 signals
infinite-dimensional memory approaching a branch cut.
These units enable statements that cross disciplinary boundaries. A condensate made of proteins
with D_eff = 15 has quantifiably richer memory than one with D_eff = 3 — and the renormalisation
flow predicts how that cellular-level D_eff emerges from the molecular-level kernels of its
constituent proteins.
7. Handling Unbounded Complexity: The Infinite-Dimensional Limit
Certain biological phenomena — the rheology of cytoplasm, the richness of neural population
dynamics, evolutionary adaptation — exhibit memory that is effectively infinite-dimensional. The
framework diagnoses and handles this case explicitly.
A power-law kernel K(t) = t^{-α} with 0 < α < 1 is CM, but its Laplace transform has a branch cut on
the negative real axis rather than discrete poles. The SOE approximation converges, but D_eff → ∞
D. K. Ahorlu | GFE Framework | p. 7Aberconics / GFE — Hierarchical Complexity & Renormalisation Flow
as the number of terms L grows with observation time. No finite-dimensional embedding captures
the full dynamics; any practical model is necessarily truncated.
The fractal exponent d_s = 2α provides the diagnostic:
K(t) ~ t^{-d_s/2}
⇒
J(ω) ~ ω^{d_s/2-1}
T_obs^{1-d_s/2}
⇒
D_eff(T_obs) ~
This allows principled truncation: choose L such that the truncation error in J(ω) at the highest
frequency of interest is below a specified tolerance. The error bound is:
ε(L) = ∫_{γ_L}^{∞} dμ(γ) / (γ2 + ω_max2) ≤ μ([[γ_L, ∞)]) / ω_max2
This is the renormalisation group fixed point of the flow for fractal systems: the kernel is
scale-invariant, and every level of the hierarchy sees the same power-law memory with the same
exponent, only stretched in time. Turbulent blood flow, cytoplasmic viscoelasticity, and certain
neural dynamics belong to this universality class.
8. Blueprint for Multi-Scale Biological Modelling
The following pipeline integrates all elements of the framework into a practical protocol for
modelling any biological system exhibiting hierarchical non-Markovian dynamics.
•​ Step 1 — Data acquisition. Gather time-series data at each relevant scale: single-channel
currents, NMR relaxation, spike trains, live-cell imaging, tissue rheology. Identify the
observable at each level and the relevant timescale range.
•​ Step 2 — Kernel extraction. For each level, extract K(t) via the appropriate method:
FACF from MD; R1/R2/NOE inversion (two-stage pipeline) from NMR; impulse response
from electrophysiology. Run the CM filter. Fit SOE via joint NNLS with Rex extraction if
chemical exchange contributes. Compute spectral units.
•​ Step 3 — Renormalisation consistency check. Verify that D_eff grows
sub-multiplicatively between levels. Check that M_scale increases monotonically. If D_eff
decreases between levels, this signals either channel merging (physical) or fitting artefact
(methodological) — use Prony diagnostics to distinguish.
•​ Step 4 — Hierarchical MIN construction. Couple levels using the assembly rule:
bottom-up via forcing F_n(X_{n-1}), top-down via parameter modulation K_{n-1}(t; X_n).
Enforce CM preservation at each coupling: wl(X_n) ≥ 0, γl(X_n) > 0. Learn coupling
functions from cross-level correlation data using local Hebbian rules.
•​ Step 5 — Simulation. Integrate the coupled auxiliary variable ODE system. Total
dimension = Σ_n L_n, typically 20–100 for a 3–4 level model. Run Formulation B dynamics
(Aberconics u, m, v decomposition) for each level's observable.
•​ Step 6 — Tuning and inverse design. Tune individual fields (u, m, v) at any level to
explore the dynamical neighbourhood of the real system. Specify a target behaviour at level
n+1 and invert the renormalisation map to determine which level-n kernel parameters
D. K. Ahorlu | GFE Framework | p. 8Aberconics / GFE — Hierarchical Complexity & Renormalisation Flow
produce it. For sequence-level design, invert the Rouse eigenvalue problem to recover
residue composition constraints.
•​ Step 7 — Validation. Compare predictions at each level against held-out experimental
data. Use spectral units as the primary validation metric: predicted D_eff, M_cap, and d_s
should match observed relaxation spectra. Mismatches diagnose specific modelling failures:
D_eff too low → insufficient memory channels; M_scale too narrow → missing slow or fast
channels; d_s wrong → incorrect fractal structure.
9. Conclusions
The Aberconics / GFE framework addresses biological hierarchical complexity not by treating each
level independently, but by establishing a single mathematical object — the CM memory kernel and
its SOE representation — that is well-defined, physically meaningful, and computationally tractable
at every level of biological organisation.
Three results carry the weight of the formal argument. The CM-from-stability theorem justifies
applying the framework above the equilibrium molecular level. The nonlinearity-memory duality
theorem guarantees that linear non-Markovian models are complete descriptions of projected
nonlinear systems. The renormalisation flow shows that CM and the SOE structure propagate
upward under coarse-graining, with D_eff growing sub-multiplicatively and the spectral weight
shifting systematically toward slower timescales.
The result is not a collection of level-specific models connected by ad hoc interfaces. It is a tower of
equivalent theories, one at each scale, connected by mathematically derived transformations that
preserve the framework's core properties. The spectral units provide the common language in which
measurements at any level can be compared and predictions at other levels derived.
The immediate frontier is the experimental validation of the renormalisation flow: measuring D_eff
at the residue, domain, and complex levels for a single protein system and verifying the predicted
sub-multiplicative scaling. The CaM pipeline we have built — extracting K(t) from NMR data at the
residue level — is the starting point for that measurement.
D. K. Ahorlu | GFE / GGFE / Aberconics Framework
D. K. Ahorlu | GFE Framework | p. 9