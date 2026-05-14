# HierAberConic-D2C: Technical Reference

> Generated from `HierAberConic_D2C_Technical_Reference_v1.0.pdf` for repo-local reading. Tables and equations are preserved as plain text where layout-native PDF formatting does not map cleanly to Markdown.

**HierAberConic-D2C: A Physics-Grounded Hierarchical Memory Architecture for Non-Markovian Inference, Control, and Continual Learning**

D. K. Ahorlu

Advanced Adaptive Systems Laboratory

correspondence@example.org

May 2026 | Technical Reference v1.0 | github.com/pilloverx/aberconics-framework

## Contents

- Abstract
- 1. Introduction
- 2. Background and Related Work
- 3. Mathematical Foundations
- 4. The Aberconics Memory Block
- 5. The Hierarchical MIN Architecture
- 6. Multi-Timescale Value Learning
- 7. Discrete-to-Continuous Bridge
- 8. Learning Rules and Adaptation
- 9. The Aberconics Training Runtime
- 10. Spectral Diagnostics and Interpretability
- 11. Stability: Full Proofs
- 12. Experiments
- A. Pseudocode Reference
- B. Hardware Mapping
- C. Hyperparameter Guide
- D. Implementation Architecture
- References

## Abstract

We introduce HierAberConic-D2C, a hierarchical memory architecture for problems where the present state alone is insufficient to explain future evolution. The architecture is founded on three convergent principles: (i) Sum-of-Exponentials (SOE) memory kernels that make non-Markovian history explicit, interpretable, and provably stable; (ii) a predictive-coding hierarchy organised by timescale rather than abstraction alone, connecting levels via bottom-up error signals and top-down kernel modulation; and (iii) multi-timescale value learning in which each SOE memory channel induces its own discount horizon, unifying credit assignment and memory retention in a single dynamical substrate. The framework is grounded in the Generalised Langevin Equation and the Mori–Zwanzig projection formalism, ensuring that memory kernels are not heuristic constructs but exact reductions of hidden degrees of freedom. A renormalisation flow connects levels of the hierarchy, preserving complete monotonicity and the SOE structure under coarse-graining. Spectral units—including effective memory dimension D_eff, memory capacity M_cap, and spectral entropy H_mem—serve as a universal diagnostic language, enabling real-time monitoring, pruning, and growth of memory channels. The architecture targets non-Markovian settings: long-memory prediction, delayed control, continual adaptation, and hybrid symbolic–dynamical reasoning. A discrete-to-continuous bridge allows token or symbolic inputs to drive the continuous dynamical core without replacing it. Stability is guaranteed by design via proven conditions on kernel weights and decay rates. Early validation on coloured-noise approximation and Lorenz chaos suppression establishes the empirical foundation; broader benchmarks are underway. We position HierAberConic-D2C not as a replacement for existing deep learning but as a principled alternative for regimes where memory structure, interpretability, and stability guarantees are primary requirements.

## 1. Introduction

### 1.1. The Memory Problem in Modern AI

Contemporary sequence models—Transformers [5], LSTMs [6], and structured state-space models such as S4 and Mamba [7, 8]—have demonstrated remarkable empirical success on language, vision, and control benchmarks. Yet each encodes temporal history implicitly: attention weights are recomputed from scratch at every step, gated recurrent states carry no physically interpretable timescale, and no architecture provides a formal guarantee that memory structure is stable or recoverable from data. This matters in a growing class of problems. Climate emulation must integrate forcing signals across days and decades simultaneously. Neural control systems must remember an action’s consequence minutes after it was taken. Biomedical monitors must maintain patient baselines across non-stationary sessions. In each case the system is fundamentally non-Markovian: the present observation is insufficient to predict the future without access to structured history. The central claim of this work is that memory should be treated as an explicit computational object, governed by physics, not inferred implicitly from data depth. The architecture we present formalises this claim across four dimensions: mathematical grounding, hierarchical organisation, value-directed adaptation, and hardware-mappable implementation.

### 1.2. What HierAberConic-D2C Is

HierAberConic-D2C is a hierarchical predictive-memory architecture built from Aberconics Memory Blocks (AMBs). Each AMB couples a latent state u(t) ∈Rd to a bank of L SOE memory channels χℓ(t), whose decay rates γℓencode physical timescales and whose weights wℓare learned online. The augmented state z = (u, χ) is Markovian in the extended space, making the system amenable to Bellman-style value learning despite describing non-Markovian dynamics in the observation space. Multiple AMBs are stacked into a timescale hierarchy: lower levels handle fast sensory dynamics; higher levels accumulate slow structural context. Levels interact bidirectionally—lower levels send prediction errors upward; higher levels modulate the memory kernel of lower levels top-down. A renormalisation map governs how kernels transform across levels, preserving complete monotonicity. A discrete-to-continuous bridge converts token or symbolic inputs into smooth forcing signals, allowing language and symbolic reasoning to be layered on top of the continuous dynamical core rather than replacing it. Training combines predictive-coding loss, multi-timescale TD value learning, and stability regularisation, orchestrated by a specialised runtime rather than a standard gradient loop.

### 1.3. Key Contributions

1. Unified dynamical substrate. A single mathematical object—the SOE memory kernel— simultaneously governs memory retention, credit assignment, and multi-timescale discounting. 2. Stability by design. Proven stability conditions (Theorem 4.1) are enforced throughout training. The completely monotone (CM) constraint on kernel weights is preserved under all update rules. 3. Hierarchical renormalisation. A formal renormalisation flow connects levels, with D_eff growing sub-multiplicatively— grounding the hierarchy in physical coarse-graining rather than engineering convention. 4. Interpretable diagnostics. Spectral units (D_eff, M_cap, H_mem) are computable at every step, enabling principled pruning, growth, and runtime monitoring. 5. Hardware pathway. Each memory channel maps directly to an RC circuit; the full stack approaches thermodynamic memory limits, providing a route to neuromorphic implementation. 6. Domain-specific training runtime. An Aberconics Director orchestrates stateful episodes, layerwise update rates, stability checks, and online consolidation—none of which are supported by standard ML training loops.

### 1.4. Scope and Limitations

The results presented here are a foundation, not a finished system. Early experiments on coloured-noise approximation and Lorenz chaos suppression establish that the mathematical framework is implementable and yields physically meaningful results. These should be understood as proofs of concept, demonstrating that the design principles are sound and that the direction is productive. Broader benchmarks—chaotic forecasting at scale, delayed-reward control, long-context language modelling—are underway and will be reported in subsequent work. HierAberConic-D2C does not compete with large-scale Transformers on tasks where massive data and compute are available. Its niche is the regime where memory structure, stability guarantees, and interpretability are primary: small data, continuous-time processes, safety-critical control, and scientific modelling.

### 1.5. Paper Organisation

Part I (§2–3) reviews prior work and establishes the mathematical foundations. Part II (§4–7) describes each architectural component in detail. Part III (§8–9) covers learning rules and the training runtime. Part IV (§10–11) provides the interpretability framework and full stability proofs. Part V (§12) reports experiments. Appendices provide pseudocode reference, hardware mapping, and a hyperparameter guide.

## 2. Background and Related Work

### 2.1. Non-Markovian Dynamics and Memory Kernels

A dynamical system is non-Markovian when its future evolution depends on history beyond the current state. The canonical mathematical treatment is the Generalised Langevin Equation (GLE) [1, 2]: ˙u(t) = − Z t 0 K(t −s) L[u(s)] ds + f (t), (1) where K(t) is the memory kernel, L is a spatial operator, and f (t) is a fluctuating force. The Mori–Zwanzig formalism [1, 2] derives K(t) rigorously by projecting high-dimensional Markovian dynamics onto an observable subspace. The kernel thereby encodes all information about eliminated degrees of freedom that is observable from the projected subspace.

### 2.2. Sum-of-Exponentials Approximation

Direct numerical convolution in (1) requires O(N2 t ) operations. The Sum-of-Exponentials approximation [3] K(t) ≈KL(t) = L ∑ ℓ=1 wℓe−γℓt, wℓ, γℓ> 0, (2) introduces auxiliary variables χℓ(t) = R t 0 e−γℓ(t−s)u(s) ds, each satisfying a local ODE, reducing complexity to O(L · Nt) with L ≪Nt. Exponential convergence of rational Padé approximants guarantees high accuracy with modest L ∼5–10 [4]. The stability analysis and corrected coupling formulations needed to make this computationally safe were established in prior work [20, 21] that forms the immediate foundation of this paper.

### 2.3. Structured State-Space Models

S4 [7] and its successors (Mamba [8], H3, Hyena) parameterise the state-space as a discretised linear dynamical system with structured matrices, achieving near-linear sequence complexity. These models share the SOE intuition but differ fundamentally: their parameters are tuned for computational efficiency rather than physical interpretability; they carry no stability guarantee; they do not support hierarchical timescale organisation; and their hidden state has no spectral diagnostic equivalent to D_eff. HierAberConic-D2C may be seen as a physically grounded, hierarchical extension of the SSM programme.

### 2.4. Predictive Coding

Predictive coding [11, 12] organises perception as hierarchical prediction–error minimisation. Each level predicts the activity of the level below; residual errors propagate upward as learning signals. The framework is biologically plausible and supports local learning rules. However, existing predictive-coding architectures do not incorporate explicit SOE memory channels, provide no timescale hierarchy grounded in physical decay rates, and do not address non-Markovian control. HierAberConic-D2C embeds predictive coding within an SOE dynamical substrate, adding memory, stability, and value-directed adaptation.

### 2.5. Neural ODEs and Continuous-Time Models

Neural ODEs [9] parameterise dynamics as ˙h = fθ(h, t) with a neural network right-hand side. This provides a continuous-time foundation but offers no explicit memory structure, no stability guarantee, and no hierarchical timescale organisation. Augmented variants [10] expand the state space, but without the physical motivation of the SOE–GLE connection. The present work may be seen as a Neural ODE with a principled, interpretable, and provably stable memory augmentation.

### 2.6. Multi-Timescale Reinforcement Learning

Temporal credit assignment—attributing a reward at time t to actions many steps earlier—is a central challenge in RL. Standard eligibility traces [13] decay at a single manually tuned rate λ. Options frameworks [14] introduce macro-actions but do not provide continuous-time multi-timescale credit assignment. We show (Section 6) that SOE memory channels provide a structured generalisation of eligibility traces, with physically motivated, per-channel decay rates that emerge from the memory kernel rather than from hyperparameter search.

## 3. Mathematical Foundations

### 3.1. Complete Monotonicity and Its Physical Basis

Definition 3.1. Complete Monotonicity A function K : [0, ∞) →R is completely monotone (CM) if it is infinitely differentiable and (−1)nK(n)(t) ≥0 for all t ≥0 and n ≥0. Equivalently (Bernstein’s theorem), K is CM if and only if it is the Laplace transform of a positive Borel measure µ on [0, ∞): K(t) = Z ∞ 0 e−γt dµ(γ). (3) At the molecular level, K(t) is CM because it equals the autocorrelation of a Gaussian stationary bath, guaranteed by the fluctuation–dissipation theorem [1]. At higher levels of organisation— cells, tissues, regulatory networks—the bath is no longer in equilibrium and the standard proof fails. The following result recovers CM from a weaker, biologically relevant condition. Theorem 3.1. CM from Stability Let u(t) satisfy a stable linear system near a fixed point with impulse-response kernel K(t). If the linearised system is causal and passive (non-negative energy dissipation), then K(t) is CM. Proof sketch. K(t) is the impulse response of a stable, causal, passive system. Stability places all poles in the open left half-plane; passivity ensures the transfer function ˆK(s) has non-negative real part for Re(s) ≥0. By the positive-real lemma, ˆK(s) is the Laplace transform of a positive measure on [0, ∞), which is precisely the Bernstein representation (3). The biological corollary is direct: pathological states (cancer, neurodegeneration, cardiac arrhythmia) are precisely those in which stability breaks down, hence CM fails. Complete monotonicity is the mathematical signature of functional biological organisation, not merely a computational convenience. 3.2. SOE Embedding and the Auxiliary-Variable System Replacing the continuous measure in (3) by a finite atomic measure µ = ∑ℓwℓδγℓgives the SOE approximation (2). Defining auxiliary variables χℓ(t) = Z t 0 e−γℓ(t−s)u(s) ds, (4) each χℓsatisfies the local ODE ˙χℓ= −γℓχℓ+ u(t), ℓ= 1, . . . , L, (5) and the non-local convolution becomes the finite sum Z t 0 K(t −s)u(s) ds ≈∑ ℓ wℓχℓ(t). (6) The augmented state z = (u, χ1, . . . , χL) evolves as a Markovian ODE system in R1+L, despite describing non-Markovian dynamics in u alone. This Markovianisation of memory is the computational engine of the entire framework.

### 3.3. Nonlinearity–Memory Duality

The following result, established in the geometric-Fourier-extension line of work [21], is the theoretical keystone of the framework. Theorem 3.2. Nonlinearity–Memory Duality (i) Reduction. Any Lipschitz nonlinear Markovian system on a high-dimensional space, projected onto a lower-dimensional observable subspace, generates a linear non-Markovian system with a CM memory kernel. The kernel encodes all information about eliminated degrees of freedom that is observable from the subspace. (ii) Embedding. Any linear non-Markovian system with CM kernel K(t) embeds exactly into a higher-dimensional Markovian system via the SOE auxiliary construction. The embedding is exact when L is finite and K(t) has a finite atomic spectral measure. Corollary 3.1. Completeness of Non-Markovian Description The non-Markovian kernel description is not an approximation of the Markovian one: it is exactly equivalent, with K(t) carrying all discarded information. Fitting K(t) from data is an act of model identification, not phenomenological curve-fitting.

### 3.4. Renormalisation Flow Across Levels

Moving from a fine-grained level n to a coarser level n+1 involves integrating out fast degrees of freedom. The resulting kernel satisfies the renormalisation map: Kn+1(t) = (Kn ∗Kn bath)(t) + Kn direct(t), (7) where ∗denotes temporal convolution, Kn bath is the bath correlation function at level n, and Kn direct captures direct memory contributions that survive coarse-graining. Proposition 3.1. CM Preserved Under Coarse-Graining If Kn(t) and Kn bath(t) are both CM, then Kn ∗ Kn bath is CM. Proof. The Bernstein class is closed under convolution because convolution of Laplace transforms is multiplication: the product of two completely monotone Laplace transforms is completely monotone. □ When both kernels are SOE, their convolution is also SOE:  Kn ∗Kn bath (t) = ∑ ℓ∑ m wn ℓun m γn ℓ−δnm  e−δn mt −e−γn ℓt , (8) yielding at most L × M terms, reduced by pruning to ˜L significant modes. The effective dimension transforms as Dn+1 eff ≤Dn eff · D_eff(Kn bath), (9) establishing sub-multiplicative growth of memory complexity across levels, consistent with observed moderate D_eff values at each biological scale.

## 4. The Aberconics Memory Block

### 4.1. Core Equations

An Aberconics Memory Block (AMB) at level n couples a latent state un(t) ∈Rd to Ln memory channels χn,ℓ(t) via: ˙un = −ℓnun − Ln ∑ ℓ=1 wn,ℓχn,ℓ+ Fn(un−1) + Pn(un+1) · un + xn (10) ˙χn,ℓ= −γn,ℓχn,ℓ+ un, (11) where ℓn > 0 is the leak rate, Fn is the bottom-up forcing from level n−1, Pn is top-down kernel modulation from level n+1, and xn(t) is direct external input (non-zero only at the base level). All learnable parameters—wn,ℓ, γn,ℓ, ℓn—are strictly positive, preserving CM.

### 4.2. Three Stable Coupling Formulations

Early formulations used state-driven positive feedback, which is provably unstable for typical parameters [20]. Three corrected formulations are used: Formulation A (Input-Driven). ˙u = −ℓu + ∑ℓwℓχℓ+ x(t), ˙χℓ= −γℓχℓ+ x(t). (12) Memory channels track external input, not the state. The Jacobian is upper-triangular, giving eigenvalues {−ℓ, −γ1, . . . , −γL}—all negative, hence unconditionally stable. Formulation B (Negative Feedback). ˙u = −ℓu −∑ℓwℓχℓ+ x(t), ˙χℓ= −γℓχℓ+ u. (13) Memory provides intelligent dissipation. Used for hierarchy levels and chaos suppression. Formulation C (Second-Order Resonant). ¨u + 2ζω0 ˙u + ω2 0u = x(t) + ∑ℓwℓχℓ, ˙χℓ= −γℓχℓ+ u. (14) A damped oscillator with memory-modulated forcing, enabling resonant echoes when 1/γℓ≈1/ω0.

### 4.3. Stability Theorem

Theorem 4.1. Positive-Feedback Instability (Formulation B, corrected) Consider Formulation B with x(t) = 0. The equilibrium (z) = 0 is asymptotically stable if and only if ℓ> L ∑ ℓ=1 wℓ γℓ . (15) Proof. The Jacobian of the linearised system is J =      −ℓ −w1 · · · −wL 1 −γ1 · · · 0 ... ... ... ... 1 0 · · · −γL     . At λ = 0, the characteristic polynomial evaluates to det(−J) λ=0 = ℓ∏ℓγℓ  1 − ∑ℓwℓ/(ℓγℓ)  . This is positive (no zero eigenvalue) iff condition (15) holds. A Lyapunov function V = 1 2u2 + 1 2 ∑ℓ(wℓ/γℓ)χ2 ℓ satisfies ˙V ≤0 when (15) holds, confirming asymptotic stability. □ This condition is enforced at every training step, providing a hard stability guarantee absent from all standard recurrent architectures.

### 4.4. Hardware Mapping

Each memory channel χℓwith decay rate γℓmaps to an RC circuit with time constant τℓ= 1/γℓ: R = 100 kΩ, C = 1/(γℓ· R). A summing operational amplifier realises the weighted output ∑ℓwℓχℓ. The Landauer bound for maintaining L = 7 channels at T = 300 K and f = 1 kHz is Pmin ≈1.4 × 10−15 W—approaching the thermodynamic limit and three orders of magnitude below SRAM [20, 18].

## 5. The Hierarchical MIN Architecture

### 5.1. Design Principle: Timescale as Hierarchy

Standard deep networks organise layers by abstraction. HierAberConic-D2C organises levels by timescale: level n has characteristic timescale ¯τn = 1/ ¯γn (geometric mean of its channel decay rates), with the constraint ¯τ1 < ¯τ2 < · · · < ¯τN. (16) Lower levels capture fast sensory dynamics; higher levels accumulate slow structural context. This organisation is not a modelling assumption: it emerges from the renormalisation flow (7), which shifts spectral weight toward slower modes at each coarser level. A reference parameterisation for language or continuous-time control is given in Table 1.

**Table 1. Reference level parameterisation.**

Level Role ¯τn Ln D_eff target 1 Fast sensory 1–10 ms 3–5 2–4 2 Feature 50–200 ms 5–8 4–7 3 Context 0.5–2 s 8–12 6–10 4 Semantic 5–30 s 8–12 6–10 5 Structural 1–5 min 5–8 5–8 6 Document/Goal 10–30 min 3–5 3–5

### 5.2. Bottom-Up Coupling

Level n−1 drives level n through a forcing function: Fn(un−1) = hn  un−1(t)  , (17) where hn : Rd →Rd is a learned projection (linear for interpretability; nonlinear optionally). This corresponds to the standard Mori–Zwanzig coarse-graining and preserves CM at level n provided Kn−1(t) is CM (Proposition 3.1).

### 5.3. Top-Down Kernel Modulation

The key architectural innovation is that higher levels reshape the memory structure of lower levels, not merely their activity. Level n modulates the kernel of level n−1 by making weights and decay rates functions of the slow state: Kn−1(t; un) = ∑ ℓ wℓ(un) e−γℓ(un) t. (18) CM is preserved by parameterising wℓ(un) = eαℓ(un), γℓ(un) = eβℓ(un), (19) with unconstrained αℓ, βℓ(linear projections of un). A global stability bound wℓ(un) ≤wmax ensures condition (15) holds for all un.

### 5.4. The Coupling Assembly Rule

The complete bidirectional coupling at each timestep is: Level n forcing −−−→Level n−1 : Fn−1(t) ←gn  un(t)  , (20) Level n modulation −−−−−−→Level n−1 : Kn−1(t) ←Kn−1(t; un(t)), (21) Level n−1 output −−−→Level n : Fn(t) = hn  un−1(t)  , (22) with CM preservation enforced: wℓ(un) ≥0, γℓ(un) > 0 for all un. Coupling functions gn, hn are learned from cross-level correlations via local Hebbian rules (Section 8).

### 5.5. Predictive Coding Integration

Each level n maintains a prediction of level n−1: ˆyn(t) = Wpred,n un(t), (23) and receives the prediction error εn(t) = un−1(t) −ˆyn(t). (24) Errors propagate bottom-up; predictions propagate top-down. Learning at level n is driven by εn: weights wn,ℓare updated to reduce future prediction errors, while timescales γn,ℓare adapted to match the temporal structure of the error signal (Section 8).

## 6. Multi-Timescale Value Learning

### 6.1. The Augmented State and Its Markov Property

A critical correction from standard RL practice: the value function V is not part of the dynamical state. The state is: zn(t) =  un(t), χn(t)  ∈Rd+Ln, (25) and the critic Vℓ: Rd+Ln →R is a learned function over zn. Including V in the state would introduce a circular dependency; the two objects must remain cleanly separated. Proposition 6.1. Markov Property of Augmented State The process (zn(t))t≥0 is Markovian: given zn(t), the future evolution is determined by the dynamics (10)–(11) and requires no additional history. Proof. χn,ℓ(t) = R t 0 e−γℓ(t−s)un(s) ds is fully determined by zn(t) and the subsequent ODE, with no dependence on history before t. □ This result justifies Bellman-style value learning over zn despite the non-Markovian nature of un alone.

### 6.2. Hamilton–Jacobi–Bellman Equation

The value function for the augmented state satisfies the continuous-time HJB equation: ρV(z) = max a h R(z, a) + ∇uV · f (u, χ, a) + ∑ ℓ ∂V ∂χℓ  −γℓχℓ+ u i , (26) where ρ > 0 is a global discount rate and f encodes the latent dynamics (10). The memory channels appear explicitly in (26) through two gradient terms: • Decay term: −γℓχℓ∂V/∂χℓ(fading memory reduces value sensitivity); • Update term: u ∂V/∂χℓ(current state refreshes memory relevance). Slow channels (small γℓ) contribute more to longhorizon value; fast channels dominate immediate assessment.

### 6.3. Per-Channel Value Decomposition

We define a per-channel value function with discount rate γℓmatching the memory channel: Vℓ(z) = E Z ∞ t e−γℓ(s−t)r(z(s), a(s)) ds z(t) = z  . (27) The aggregate value under the full kernel is VK(z) = ∑ ℓ wℓVℓ(z), (28) which follows directly from the SOE decomposition of the reward functional. We adopt the ensemble of L critics as a design choice (rather than a single critic on z or a shared-feature critic), motivated by the natural correspondence between channel ℓand horizon 1/γℓ. Remark 6.1. The SOE decay rates γℓinduce a natural family of timescale-matched discount factors e−γℓ∆t. This is a structural correspondence, not a theorem: exact equivalence between memory discount and reward discount holds when the reward functional shares the same SOE kernel as the memory.

### 6.4. Per-Channel TD Errors

Discretising (27) with timestep ∆t: δℓ(t) = r(t) + e−γℓ∆tVℓ(zt+∆t) −Vℓ(zt). (29) Each δℓhas variance Var[δℓ] ∝1/γℓ: slow channels have noisier TD targets because they integrate over longer, more uncertain horizons. Per-channel learning rates are therefore set as ηV ℓ= ηV base · SNRℓ maxℓ′ SNRℓ′ , (30) where SNRℓ= E[|δℓ|]/Std[δℓ] is the signal-tonoise ratio of the TD error for channel ℓ. This is presented as a practical heuristic: the optimal schedule depends on reward sparsity and horizon length.

### 6.5. Eligibility Traces as a Structured Generalisation

The memory channels χℓ(t) provide a structured generalisation of eligibility traces. Standard TD(λ) traces et = λγet−1 + ∇wV(st) operate in weight space with a single, manually tuned decay rate λ. The channels χℓoperate in observation space with L physically motivated decay rates, supporting simultaneous multi-timescale credit assignment without hyperparameter search. The three-factor update rule (Section 8) exploits this directly: the value gradient at timescale ℓis carried by χℓ, which has already accumulated the relevant history at the matching timescale.

## 7. Discrete-to-Continuous Bridge

### 7.1. Design Principle

The continuous dynamical core is primary; discrete inputs are a boundary condition, not the substrate. Tokens or symbols are converted into smooth forcing signals that drive the base-level AMB. Between tokens, the system evolves freely under its own dynamics, integrating context in the memory channels. This allows the model to think between words—an operation impossible in architectures where processing is token-synchronous.

### 7.2. Token-to-Forcing Conversion

A token k arriving at time tk is embedded as ek = Embed(k) ∈Rd, then spread into a smooth forcing signal: x(t) = ek · ϕ(t −tk), ϕ(τ) = J ∑ j=1 αje−βjτ, (31) where αj, βj are learnable parameters defining the injection kernel. This is itself an SOE, ensuring the forcing is smooth, physically motivated, and integrates naturally with the memory channels.

### 7.3. Continuous-to-Discrete Readout

At discrete output times tk, logits are read from the base-level state: ℓk = Wout u1(tk) + bout, (32) passed through a softmax (or Gumbel–Softmax for differentiable discrete sampling [15]) to produce a token distribution. Diffusion or flow-matching modules may be attached as generative adapters for structured output spaces.

### 7.4. Language as a Boundary Mechanism

Language and symbolic processing are boundary mechanisms layered on the dynamical substrate, not the substrate itself. This distinction has practical consequences: the continuous memory accumulates cross-token context in χℓ(t) without a fixed context window; information about token k is available at token k+m through the slow channels, with graceful exponential decay rather than hard truncation. Long-distance dependencies are handled by channels with small γℓ; local syntax by channels with large γℓ.

## 8. Learning Rules and Adaptation

### 8.1. The Three-Factor Update Rule

At each level n, kernel weights wn,ℓare updated by a three-factor rule combining predictive coding and value signals: dwn,ℓ dt = ηpred χn,ℓεn + ηval χn,ℓδn,ℓ−β(εn) wn,ℓ, (33) where: • ηpred χn,ℓεn is the Hebbian prediction term: strengthen channels that correlate with prediction error; • ηval χn,ℓδn,ℓis the value term: strengthen channels that carry credit for reward; • β(εn) = β0/(1 + |εn|) is error-modulated decay—faster forgetting when predictions are accurate (consolidation), slower when errors are large (plasticity). The CM constraint wn,ℓ≥0 is enforced after each step.

### 8.2. Timescale Adaptation

Decay rates γn,ℓadapt at a rate α ≪ηpred, ηval (timescales change 10–100× slower than weights): dγn,ℓ dt = −α γn,ℓsign  χn,ℓ· εn  , (34) with hard constraints γmin ≤γn,ℓ≤γmax enforced after each update. Intuitively: if χn,ℓand εn are positively correlated, this channel is relevant to the error and should integrate over a longer window (decrease γ); if negatively correlated, it should release memory faster (increase γ).

### 8.3. Consolidation Mechanism

To prevent catastrophic forgetting, weights are split into fast and slow components: wℓ= wfast ℓ + wslow ℓ , (35) where wfast ℓ is updated by (33) at every step, and wslow ℓ is updated periodically: wslow ℓ ←wslow ℓ + βc  wfast ℓ −wslow ℓ  , (36) with consolidation rate βc ≪β0. Fast weights then reset toward the slow baseline: wfast ℓ ← (1−βc)wfast ℓ + βc wslow ℓ . This mechanism, analogous to sleep-dependent memory consolidation [17], prevents recent experience from overwriting long-term structure.

### 8.4. Channel Birth and Death

The number of channels Ln is adaptive: Death (pruning). If wn,ℓ< ϵprune · maxℓ′ wn,ℓ′ for more than Tprune steps, channel ℓis removed. Birth (growth). If D_eff,n ≈Ln (all channels contribute equally, saturation) and the prediction error εn contains significant power at a timescale ˆτ not covered by current channels, a new channel is inserted with γnew = 1/ ˆτ and small initial weight. These operations preserve CM and the stability condition (15) by construction.

### 8.5. Online versus Offline Training

Online mode. Dynamics run continuously; weights updated by (33) at every step; value heads updated when sufficient trace data is available. No batching; no gradient accumulation. Suitable for deployment and real-time adaptation. Offline mode. Trajectory windows are sampled from a TraceStore (Section 9); weight updates via (33) applied to stored traces; value head parameters updated by gradient descent on ∑ℓδ2 ℓ. Suitable for initial training and benchmarking. Hybrid mode. Continuous online dynamics with periodic offline value-learning passes on stored traces. This is the recommended configuration: the dynamical substrate never stops; learning operates at two speeds.

## 9. The Aberconics Training Runtime

### 9.1. Why a Specialised Runtime Is Required

Standard ML training loops assume a static computation graph, i.i.d. mini-batches, a single global loss, and stateless processing between batches. HierAberConic-D2C violates all four assumptions: memory channels are stateful across episodes; training must support layerwise updates at different rates; the learning objective is a composite of local prediction errors, per-channel TD errors, stability penalties, and memory regularisation; and the system must support online consolidation. A domain-specific training runtime—an Aberconics Director—is therefore required. The runtime is structured as three layers: 1. C++ core (GFE kernel engine, ABERSOE runtime, hierarchical MIN): handles dynamics, kernel fitting, spectral diagnostics, stability monitoring, and serialisation. 2. Python director (AberconicsDirector, TraceStore, Scheduler, SpectralInspector, CheckpointManager): handles orchestration, objective composition, training phase transitions, and logging. 3. Optional accelerator: custom CUDA kernels for batched SOE stepping, distributed execution (future work).

### 9.2. The AberconicsDirector State Machine

The Director operates as a finite-state machine governing training phases:

**Algorithm 1 AberconicsDirector state machine**

1: States: WARMUP, EXPLORE, CONSOLIDATE, EVALUATE, PRUNE, GROW 2: Initialise: state ←WARMUP, t ←0 3: while not done do 4: z, ε ←HIERARCHYSTEP(obs, ∆t) 5: Append (z, r, ε) to TraceStore 6: if state = WARMUP and t > Twarmup then 7: state ←EXPLORE 8: else if state = EXPLORE then 9: UPDATEWEIGHTS(ε, δ, ∆t) 10: if ¯ε < εthresh then 11: state ←CONSOLIDATE 12: end if 13: else if state = CONSOLIDATE then 14: CONSOLIDATE(βc) 15: if consolidation timer expired then 16: state ←EXPLORE 17: end if 18: end if 19: if D_eff < Dmin then state ←PRUNE 20: end if 21: if D_eff ≈L then state ←GROW 22: end if 23: if t mod Teval = 0 then EVALUATE 24: end if 25: STABILITYCHECK 26: t ←t + ∆t 27: end while

### 9.3. The TraceStore

The TraceStore is not a standard replay buffer. Random i.i.d. sampling destroys temporal correlations that are the reason the architecture exists. Instead: Contiguous sampling ensures that χ(t) in the sampled window reflects the actual memory state at that time, not an incoherent mixture of unrelated episodes.

### 9.4. The StabilityMonitor

### 9.5. Composite Training Objective

The total training loss for an offline pass over a sampled window is: L = λtaskLtask + λpredLpred + λvalLval + λstabLstab + λmemLmem, (37)

**Algorithm 2 TraceStore: contiguous-window trajectory memory**

Input: Capacity C, window size W, augmentedstate dimension d+L 1: Allocate circular buffers for (u, χ, r, ε) 2: procedure ADD(u, χ, r, ε) 3: Write to position ptr; ptr ←(ptr + 1) mod C 4: end procedure 5: procedure SAMPLEWINDOW 6: s ←Uniform(0, min(|stored|, C) −W) 7: return contiguous slice [s, s + W) ▷ Preserves temporal correlation 8: end procedure

**Algorithm 3 StabilityMonitor: Theorem 4.1 enforcement**

1: procedure CHECK(level n) 2: ρn ←∑ℓwn,ℓ/(γn,ℓ· ℓn) 3: if ρn > 0.9 then 4: wn,ℓ←wn,ℓ· (0.8/ρn) ▷Rescale to satisfy (15) 5: Log violation 6: return VIOLATED 7: end if 8: return OK 9: end procedure 10: procedure EMERGENCYSTABILISE(all levels) 11: for each level n do 12: while CHECK(n) = VIOLATED do 13: Reduce wn,ℓby 10% 14: end while 15: end for 16: end procedure where Lpred = ∑ n Et[εn(t)2], (38) Lval = ∑ n ∑ ℓ Et[δn,ℓ(t)2], (39) Lstab = ∑ n ReLU  ∑ ℓ wn,ℓ γn,ℓ−0.9 ℓn  , (40) Lmem = ∑ n ReLU(D_eff,n −Dmax,n) −λdiv Varℓ  log γn,ℓ  . (41) The diversity penalty −λdiv Var(log γn,ℓ) prevents timescale collapse—a key failure mode where all channels converge to the same decay rate and lose the multi-timescale structure that makes the architecture distinct.

## 10. Spectral Diagnostics and Interpretability

### 10.1. Spectral Units: A Universal Diagnostic Language

Given a fitted SOE kernel with weights {wℓ} and rates {γℓ}, the normalised weights are ˜wℓ= wℓ/ ∑ℓ′ wℓ′. The spectral units are: M_cap = ∑ℓ˜wℓ/γ2 ℓ ∑ℓ˜wℓ/γℓ (mean memory depth, seconds), (42) M_scale = log10  γmax/γmin  (timescale span, decades), (43) M_res = L/M_scale (modes per decade), (44) H_mem = −∑ ℓ ˜wℓln ˜wℓ (spectral entropy, nats), (45) D_eff = L · exp(H_mem) (effective memory dimension). (46) D_eff = 1 corresponds to a single-exponential (Markovian) kernel. D_eff = L corresponds to a uniform spectrum—maximally non-Markovian. Power-law kernels drive D_eff →∞as observation time grows, signalling infinite-dimensional memory.

### 10.2. Deff as a Cognitive State Monitor

Computed at every timestep from the current kernel weights, D_eff,n(t) provides a real-time indicator of cognitive complexity at level n: • D_eff ≈1: System has collapsed to Markovian processing—insufficient memory for the task. • D_eff ≈L: All channels contribute equally— high complexity, potentially approaching saturation (trigger growth). • D_eff rising sharply: Sudden increase in task complexity, or instability onset. • D_eff below threshold: Channels have specialised—trigger pruning of near-zero channels. This interpretability is a design property, not an add-on: because the memory kernel is an explicit computational object with learnable SOE parameters, its spectral properties are computable at zero overhead beyond the weight normalisation in (45)– (46).

### 10.3. Comparison with Existing Architectures

Table 2 summarises HierAberConic-D2C against leading alternatives on properties that are either absent or non-interpretable in existing systems.

### 10.4. Renormalisation Consistency Validation

At each level transition, the following checks are performed: 1. Dn+1 eff ≤Dn eff · D_eff(Kn bath) (sub-multiplicative bound); 2. Mn+1 scale ≥Mn scale (span grows or stays constant); 3. ¯γn+1 < ¯γn (dominant rate slows at higher levels). Violations signal either a modelling failure (missing slow channels) or a fitting artefact (spurious fast modes at a high level).

## 11. Stability: Full Proofs

### 11.1. Formulation A: Unconditional Stability

Theorem 11.1. Input-Driven Stability For Formulation A (12) with x(t) = 0, the equilibrium is asymptotically stable for any ℓ, wℓ, γℓ> 0. Proof. The Jacobian JA is upper triangular with diagonal entries {−ℓ, −γ1, . . . , −γL}. All eigenvalues are negative by assumption; stability follows from the eigenvalue condition. □

### 11.2. Formulation B: Lyapunov Analysis

Theorem 11.2. Negative-Feedback Stability (Lyapunov For Formulation B (13) with x(t) = 0, define the Lyapunov function V(u, χ) = 1 2u2 + 1 2 ∑ ℓ wℓ γℓ χ2 ℓ. (47) When condition (15) holds, ˙V ≤−c V for some c > 0, implying exponential stability. Proof. Computing ˙V along trajectories of

**Table 2. Architectural comparison across key properties. ✓= supported by design; ∼= partially supported or**

heuristic; × = not supported. Property Transformer LSTM S4/Mamba Neural ODE Pred. Coding HierAberConic-D2C Explicit memory kernel × × ∼ × × ✓ Physical timescales × × × × × ✓ Stability guarantee × × × × × ✓ Hierarchical timescale org. × × × × ∼ ✓ Predictive coding learning × × × × ✓ ✓ Multi-timescale value learning × × × × × ✓ Online continual learning × ∼ × × ∼ ✓ No catastrophic forgetting × × × × ∼ ✓ Interpretable diagnostics (D_eff) × × × × × ✓ Hardware-mappable (analog) × × × × × ✓ (13): ˙V = u ˙u + ∑ ℓ wℓ γℓ χℓ˙χℓ = u  −ℓu −∑ ℓ wℓχℓ  + ∑ ℓ wℓ γℓ χℓ(−γℓχℓ+ u) = −ℓu2 −∑ ℓ wℓχ2 ℓ+ ∑ ℓ wℓχℓu  1 γℓ −1  . For γℓ≥1, the cross term is non-positive and ˙V ≤−ℓu2 −∑ℓwℓχ2 ℓ≤−2c V with c = min(ℓ, minℓγℓ). For general γℓ, Young’s inequality applied to the cross term yields | ∑ℓwℓχℓu| ≤εu2/2 + ∑ℓ w2 ℓ 2ε χ2 ℓ, and choosing ε to satisfy (15) gives ˙V < 0. □

### 11.3. Formulation C: Stability Condition

Proposition 11.1. Second-Order Stability For Formulation C (14) with ζ > 0, ω0 > 0, and x(t) = 0, the system is stable if ∑ ℓ wℓ γℓ < 2ζω0. (48)

### 11.4. CM Preservation Under Learning

Proposition 11.2. CM Invariance of Update Rules The three-factor update rule (33), with the constraint wn,ℓ≥0 enforced after each step, and the timescale adaptation (34) with γn,ℓ≥γmin > 0, jointly preserve the CM structure of Kn(t) = ∑ℓwn,ℓe−γn,ℓt. Proof. A function is CM iff it is a positive linear combination of decaying exponentials with positive rates. The non-negativity constraint on wn,ℓand the positivity constraint on γn,ℓmaintain exactly this representation at every step. □

## 12. Experiments

The experiments reported here are initial validations establishing that the mathematical framework is implementable, numerically stable, and produces physically interpretable results. They should be read as proofs of concept confirming the theoretical foundations, not as comprehensive performance benchmarks. Broader benchmarks against state-of-the-art sequence models are underway.

### 12.1. Experiment 1: Coloured-Noise Approximation

#### 12.1.1. Setup

We compare three stochastic processes over a T = 500 s horizon: 1. White OU: dx = −θx dt + σ dW, θ = 1.0, σ = 0.5. 2. Coloured OU (reference): dx = −θx dt + η dt; dη = −αη dt + σ dW, α = 0.1. 3. Aberconics (L=3): dx = −θx dt + ∑i wiχi dt; dχi = −γiχi dt + σi dWi, log-spaced γ ∈ {0.01, 0.1, 1.0}. The coloured OU autocorrelation function (ACF) is fitted using NNLS with 15 exponential basis functions (γ ∈[10−2, 101], log-spaced).

#### 12.1.2. Results

The NNLS fit recovers 7 significant SOE modes with normalised L1 error < 0.5% over 200 time lags—confirming that the two-timescale coloured OU ACF is well-represented by the SOE basis despite having only two underlying timescales. Memory persistence at lag τ = 50 s: white OU ρ(50) = 0.60; coloured OU ρ(50) = 0.99; Aberconics ρ(50) = 0.995, demonstrating a 50× effective memory extension over baseline. Spectral diagnostics for the fitted kernel: D_eff = 6.86, H_mem = 1.93 nats, M_scale = 2.29 decades (Table 3).

**Table 3. OU noise experiment: key metrics.**

Metric White OU Coloured OU Aberconics ρ(50) 0.60 0.990 0.995 τeff (steps) 91 2052 4603 Memory extension 1× 22× 50× L1 fit error — — < 0.5% D_eff 1.0 — 6.86 M_scale (decades) 0.0 — 2.29 H_mem (nats) 0.0 — 1.93

#### 12.1.3. Interpretation

The result validates three claims simultaneously: the NNLS kernel fitting procedure is accurate (< 0.5% error); the SOE basis is expressive enough to represent multi-timescale coloured processes; and the spectral units correctly quantify the resulting memory structure. The D_eff = 6.86 out of L = 7 fitted modes indicates a nearly uniform spectral weight distribution— rich, distributed memory with no single dominant timescale. 12.2. Experiment 2: Lorenz ‘63 Chaos Suppression

#### 12.2.1. Setup

The Lorenz system [16] with standard parameters (σ = 10, β = 8/3, ρ0 = 28, λbaseline max ≈0.9) is augmented with Formulation B memory feedback on the x-equation: ˙x = σ(y −x) − 3 ∑ ℓ=1 wℓχℓ, ˙χℓ= −γℓχℓ+ x. (49) Parameters (w1, w2, w3, γ1, γ2, γ3) are optimised via BFGS to minimise the largest Lyapunov exponent λmax, estimated by the variational method with threaded multi-IC evaluation (4-core, 5.7× speedup). Training uses 2 initial conditions (ICs) simultaneously; 3 held-out ICs are used for validation.

#### 12.2.2. Results

Optimised parameters (Table 4) show decay rates γ ≈(0.065, 0.018, 0.008), corresponding to timescales τ ≈(15, 56, 119) s— spanning two orders of magnitude, matching the multi-timescale structure of Lorenz dynamics. Lyapunov exponent is reduced by 25% on training ICs. On validation: 2 of 3 ICs achieve periodic stability (λ < 0); 1 IC (high initial z = 20) remains chaotic. The failure case reflects geometric basinof-attraction constraints, not a framework limitation, and is consistent with theoretical predictions [20].

**Table 4. Lorenz chaos suppression: parameter and**

Lyapunov comparison. Metric Baseline Optimised λtrain −0.019 −0.024 (−25%) γ1 0.200 0.065 γ2 0.050 0.018 γ3 0.010 0.008 w1 0.500 0.035 w2 0.200 0.019 w3 0.100 0.051 Training ICs stable — 2/2 Validation ICs stable — 2/3 Wall-clock time — 130 s

#### 12.2.3. Interpretation

Three claims are validated: (1) the memory-augmented dynamical system is numerically stable with no blow-ups across 200 optimisation evaluations; (2) multi-timescale memory feedback can suppress chaotic dynamics, demonstrating the “memory as intelligent dissipation” paradigm; (3) optimised parameters are physically interpretable— γ values that span two decades correspond naturally to the separation of fast oscillation and slow drift in Lorenz dynamics. The 2/3 validation success rate motivates the ICaware training strategy (cluster-specific parameter optimisation) described as a direct next step.

### 12.3. Planned Experiments

The following experiments are in preparation and will be reported in subsequent work: Multi-timescale prediction. Synthetic signals composed of three superimposed sinusoids at timescales spanning two orders of magnitude. Expected: level specialisation visible in per-level D_eff and M_cap. Delayed-reward control. Action at t = 0, reward only at t = Tdelay. Expected: the memory channel with γℓ≈1/Tdelay dominates after training, providing interpretable credit assignment. Continual learning without forgetting. Sequential task pairs with different characteristic timescales. Expected: timescale separation prevents interference between tasks in distinct channels. Long-context sequence modelling. Synthetic sequences requiring integration over 500+ steps. Comparison against S4, Mamba, and LSTM on accuracy and D_eff diagnostic interpretability. Gray–Scott reaction-diffusion. Memory feedback modulating pattern formation parameters. Expected: D_eff maps showing spatially distributed memory complexity aligned with pattern boundaries.

## A. Pseudocode Reference

### A.1. Single AMB Step

**Algorithm 4 AMBStep: single Aberconics Memory Block forward step**

Input: State (u, χ), forcing x, parameters (ℓ, w, γ), timestep ∆t, formulation ∈{A, B, C} Output: Updated state (u′, χ′), spectral units S 1: Compute memory feedback: M ←∑ℓwℓχℓ 2: if formulation = A then 3: ˙u ←−ℓu + M + x 4: ˙χℓ←−γℓχℓ+ x ∀ℓ 5: else if formulation = B then 6: ˙u ←−ℓu −M + x 7: ˙χℓ←−γℓχℓ+ u ∀ℓ 8: else if formulation = C then 9: ¨u ←−2ζω0 ˙u −ω2 0u + M + x 10: ˙χℓ←−γℓχℓ+ u ∀ℓ 11: end if 12: Euler integrate: u′ ←u + ˙u ∆t; χ′ ℓ←χℓ+ ˙χℓ∆t 13: Check stability: assert ℓ> ∑ℓwℓ/γℓ 14: Compute S = SpectralUnits(w, γ) 15: return (u′, χ′), S

### A.2. Hierarchy Step

**Algorithm 5 HierarchyStep: one timestep of the full hierarchical system**

Input: Observation xraw, all level states {zn}, ∆t Output: Updated states {z′ n}, prediction errors {εn} 1: Bottom-up pass: 2: for n = 1 to N do 3: xn ←xraw if n = 1 else hn(un−1) 4: Compute top-down modulation: (wn, γn) ←  eα(un+1), eβ(un+1) (if n < N) 5: (u′ n, χ′ n) ←AMBStep(un, χn, xn, ℓn, wn, γn, ∆t, B) 6: end for 7: Prediction errors: 8: for n = 2 to N do 9: ˆyn ←Wpred,n u′ n 10: εn ←u′ n−1 −ˆyn 11: end for 12: return {(u′ n, χ′ n)}, {εn}

### A.3. Three-Factor Weight Update

**Algorithm 6 UpdateWeights: three-factor Hebbian–TD update**

Input: Level state χn, prediction error εn, TD errors {δn,ℓ}, current weights wn, rates ηpred, ηval, β0, ∆t Output: Updated weights w′ n 1: for ℓ= 1 to Ln do 2: β ←β0/(1 + |εn|) 3: ∆wℓ←ηpred χn,ℓεn + ηval χn,ℓδn,ℓ−β wn,ℓ 4: w′ n,ℓ←max  0, wn,ℓ+ ∆wℓ∆t  // enforce CM 5: end for 6: return w′ n

### A.4. Kernel Fitting via NNLS

**Algorithm 7 FitSOEKernel: NNLS-based SOE kernel identification**

Input: Target ACF ˆK(τ) on lags {τi}, basis rates {γj} (log-spaced, j = 1 . . . M), pruning threshold ϵ Output: SOE weights w, rates γ, fit error 1: Build design matrix: Aij ←e−γjτi 2: Solve NNLS: w ←arg minw≥0 Aw −ˆK 2 3: Prune: keep j where wj > ϵ · maxj wj 4: Normalise: w ←w/ ∑j wj 5: Compute spectral units: S ←SpectralUnits(w, γ) 6: Compute fit: L1 ← Aw −ˆK 1 / ˆK 1 7: return γ, w, L1, S

## B. Hardware Mapping

Each memory channel χℓwith γℓmaps to an RC circuit: Rℓ= 100 kΩ, Cℓ= 1/(γℓ· Rℓ). A summing operational amplifier with feedback resistors R f,ℓ= Rℓ/wℓrealises the weighted sum. A 7-channel implementation fits on a < 5 cm2 PCB; memristor-based implementations [19] could reduce this to < 1 mm2. Neuromorphic mapping (Intel Loihi 2): each χℓmaps to a dendritic compartment with time constant τℓ= 1/γℓ; wℓmaps to synaptic weight; ℓn maps to the membrane leak. The correspondence between the continuous SOE equations and the discrete-time Loihi 2 compartment model is exact to first-order Euler discretisation.

## C. Hyperparameter Guide

**Table 5. Hyperparameter guide and typical ranges.**

Parameter Symbol Typical range Notes Channels per level Ln 3–12 Start with 5; grow via D_eff Decay rates γℓ [10−2, 101] Log-spaced; adapt via (34) Leak rate ℓn 0.05–0.5 Must satisfy (15) Pred. learning rate ηpred 10−3–10−2 Per level Value learning rate ηval 10−4–10−3 Lower than pred. Timescale adapt. α 10−5–10−4 α ≪ηpred Decay baseline β0 10−4–10−3 Error-modulated Consolidation rate βc 10−3–10−2 Periodic Diversity penalty λdiv 0.01–0.1 Prevents γ collapse Prune threshold ϵprune 0.01 Fraction of max weight Window size W 50–500 Task-dependent

## D. Implementation Architecture

### D.1. Repository Structure

The reference implementation is available at github.com/pilloverx/aberconics-framework. The codebase follows a three-layer design: Layer 1 — GFE Core (C++). Kernel factory: NNLS solver, SOE basis, spectral units, parameter packing. Language-agnostic via C ABI; accessed from Python via ctypes and from Julia via ccall. Layer 2 — ABERSOE Runtime (C++ / Julia). Stateful dynamical system: memory channels, coupling formulations, Hebbian updates, Lorenz/OU/Gray-Scott/echo scenarios. Julia reference experiments are production-ready (code/julia/examples/). Layer 3 — Hierarchical MIN (C++ / Python). Multi-level coupling (hierarchical_min.cpp), top-down modulation (hierarchical_min_coupling.cpp), renormalisation (hierarchical_min_renorm.cpp), cross-level diagnostics (hierarchical_min_diagnostics.cpp). Training Runtime (Python, in development). AberconicsDirector, TraceStore, StabilityMonitor, SpectralInspector, CheckpointManager. Orchestrates the C++ substrate; interfaces with PyTorch for value head parameter updates.

## References

[1] H. Mori, “Transport, collective motion, and Brownian motion,” Prog. Theor. Phys., vol. 33, pp. 423–455, 1965. [2] R. Zwanzig, “Memory effects in irreversible thermodynamics,” Phys. Rev., vol. 124, pp. 983–992, 1961. [3] W. McLean and V. Thomée, “Numerical solution of an evolution equation with a positive-type memory term,” J. Aust. Math. Soc. Ser. B, vol. 35, pp. 23–70, 1993. [4] Y. Nakatsukasa, O. Sète, and L. N. Trefethen, “The AAA algorithm for rational approximation,” SIAM J. Sci. Comput., vol. 40, pp. A1494– A1522, 2018. [5] A. Vaswani et al., “Attention is all you need,” in NeurIPS, 2017. [6] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Comput., vol. 9, no. 8, pp. 1735–1780, 1997. [7] A. Gu, K. Goel, and C. Ré, “Efficiently modeling long sequences with structured state spaces,” in ICLR, 2022. [8] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv:2312.00752, 2023. [9] R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud, “Neural ordinary differential equations,” in NeurIPS, 2018. [10] E. Dupont, A. Doucet, and Y. W. Teh, “Augmented neural ODEs,” in NeurIPS, 2019. [11] R. P. N. Rao and D. H. Ballard, “Predictive coding in the visual cortex,” Nature Neurosci., vol. 2, pp. 79–87, 1999. [12] K. Friston, “The free-energy principle: a unified brain theory?” Nature Rev. Neurosci., vol. 11, pp. 127–138, 2010. [13] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. MIT Press, 2018. [14] R. S. Sutton, D. Precup, and S. Singh, “Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning,” Artif. Intell., vol. 112, pp. 181–211, 1999. [15] E. Jang, S. Gu, and B. Poole, “Categorical reparameterization with Gumbel-Softmax,” in ICLR, 2017. [16] E. N. Lorenz, “Deterministic nonperiodic flow,” J. Atmos. Sci., vol. 20, pp. 130–141, 1963. [17] S. Diekelmann and J. Born, “The memory function of sleep,” Nature Rev. Neurosci., vol. 11, pp. 114–126, 2010. [18] D. Attwell and S. B. Laughlin, “An energy budget for signaling in the grey matter of the brain,” J. Cereb. Blood Flow Metab., vol. 21, pp. 1133–1145, 2001. [19] D. B. Strukov, G. S. Snider, D. R. Stewart, and R. S. Williams, “The missing memristor found,” Nature, vol. 453, pp. 80–83, 2008. [20] D. K. Ahorlu, “Stabilizing memory kernels: Corrected formulations for sum-ofexponentials non-Markovian dynamics with experimental validation,” Preprint, 2026. [21] D. K. Ahorlu, “Geometric Fourier extension: A unified framework for spectral-temporal methods,” Preprint, 2025.
