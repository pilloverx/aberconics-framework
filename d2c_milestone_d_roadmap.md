# Implementation Roadmap: D2C Milestone D & Symbolic Reasoning

This plan refines the experimental roadmap for the Aberconics D2C (Discrete-to-Continuous) transition, specifically focusing on **Milestone D: Long-context and token bridge**. It incorporates rigorous scientific standards by turning demonstration ideas into testable hypotheses with baselines, metrics, and ablations.

## Objective
To demonstrate that explicit multiscale memory combined with continuous latent evolution enables superior symbolic retrieval and structured reasoning compared to standard discrete-sequence models.

## Experimental Roadmap

The experiments will be executed in the following order to build a progressive evidence base:

### 1. Symbolic Retrieval over Long Gaps (Induction Head Challenge)
*   **Hypothesis:** Explicit multiscale memory can retrieve symbolic relations across long temporal gaps without an attention mechanism.
*   **Task:** Input stream `[A] ... [gap] ... [A]`; model must predict `[B]` (where `[B]` originally followed the first `[A]`).
*   **Baselines:** Transformer (small attention window), RNN/LSTM, and Mamba-style SSM.
*   **Metrics:** Next-symbol accuracy vs. gap length, $D_{eff}$ during retrieval, and channel usage by timescale.
*   **Ablation:** Test without slow channels or with fixed $\gamma$.

### 2. Token-to-Forcing Bridge (The "Thinking Between Tokens" Test)
*   **Hypothesis:** Continuous latent evolution between discrete tokens allows the model to process context during "silent" intervals, improving latent-state quality for subsequent predictions.
*   **Task:** Map tokens to forcing vectors $\mathbf{u}(t)$, evolve for $N$ "thought steps" in continuous time, and decode the next token.
*   **Baselines:** Standard token-synchronous RNN/LSTM.
*   **Metrics:** Token prediction accuracy, latent-state retention across no-input intervals, and comparison of $D_{eff}$ vs. hidden state entropy.
*   **Ablation:** Test with zero-length intervals (token-synchronous mode).

### 3. Hierarchical Document Modeling (Structural Abstraction)
*   **Hypothesis:** A timescale-based hierarchy (as per the Renormalization Flow) captures nested semantic regularities better than flat or abstraction-only hierarchies.
*   **Task:** Process long event streams with nested structure (e.g., local syntax transitions vs. global document themes).
*   **Baselines:** Flat LSTM or flat SSM.
*   **Metrics:** Level-wise prediction error, semantic consistency across layers ($D_{eff}$ alignment), and cross-level transfer score.
*   **Ablation:** Test without top-down kernel modulation.

### 4. Temporal Logic and Delayed Inference (Rule Composition)
*   **Hypothesis:** Resonant memory channels (Formulation C) can compose logical rules across time without explicit symbolic state tracking.
*   **Task:** `IF (X occurred earlier) AND (Y occurs now), THEN predict (Z)`.
*   **Baselines:** LSTM with cell-state persistence.
*   **Metrics:** Delayed-rule accuracy, noise tolerance (jitter), and minimum memory horizon required.
*   **Ablation:** Test without resonant (Formulation C) coupling.

## Implementation Steps

### Phase 1: Infrastructure Preparation
1.  **Refactor Reporting:** Move plotting and spectral diagnostic logic from `lorenz63.py` to `d2c/reports.py`.
2.  **Generalize Director:** Update `AberconicsDirector` to handle arbitrary sequence windows and token forcing inputs.
3.  **Implement Token Bridge:** Create `d2c/experiments/symbolic_induction.py` with the initial token-to-forcing mapping logic.

### Phase 2: Hypothesis Testing (Sequential)
1.  Execute **Experiment 1 (Symbolic Retrieval)** and establish the $D_{eff}$ baseline for symbolic load.
2.  Execute **Experiment 2 (Token Bridge)** to validate "thinking between tokens".
3.  Execute **Experiment 3 (Document Hierarchy)** using multi-level `hierarchical_min` core.
4.  Execute **Experiment 4 (Temporal Logic)** using Formulation C (Resonant) kernel specs.

### Phase 3: Documentation and Analysis
1.  Generate comprehensive reports for each experiment including baseline comparisons and ablation results.
2.  Update `Context.md` with findings and updated architecture refinements.

## Verification & Testing
*   **Statistical Significance:** All experiments to be run over $N=20$ random seeds.
*   **Stability Monitoring:** All runs must report zero Theorem 4.1 violations (auto-rescaling events are acceptable but must be logged).
*   **Metric Validation:** Verify that $D_{eff}$ correlates with task complexity across all experiments.


│  To resume this session: gemini --resume ab669558-caa1-4812-b017-ff90e1efbd0a                                                │