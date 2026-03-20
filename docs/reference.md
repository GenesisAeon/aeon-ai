# API Reference

Complete mathematical and programmatic reference for `aeon-ai` v0.2.0.

---

## 1. Fieldtheory Lagrangian

### Definition

$$L(S_A, S_V, \delta, t) = \frac{S_A \cdot S_V}{S_A + S_V} - \frac{1 + \delta}{t^2}$$

**Parameters:**

| Symbol | Type | Range | Description |
|--------|------|-------|-------------|
| $S_A$ | float | $\mathbb{R}$ | Auditory signal amplitude |
| $S_V$ | float | $\mathbb{R}$ | Visual signal amplitude |
| $\delta$ | float | $\mathbb{R}_{\geq 0}$ | Deformation / curvature parameter |
| $t$ | float | $\mathbb{R}_{> 0}$ | Time step |

**Special cases:**

- $S_A + S_V \approx 0$: harmonic term set to 0 (numerical guard)
- $t \to \infty$: $L \to S_A S_V / (S_A + S_V)$ (temporal penalty vanishes)
- $\delta = 0$: standard undistorted Lagrangian

### Gradients

$$\frac{\partial L}{\partial S_A} = \frac{S_V^2}{(S_A + S_V)^2}$$

$$\frac{\partial L}{\partial S_V} = \frac{S_A^2}{(S_A + S_V)^2}$$

$$\frac{\partial L}{\partial t} = \frac{2(1 + \delta)}{t^3}$$

Note: $\partial L / \partial t > 0$ always (increasing $t$ reduces penalty).

### Python API

```python
from aeon_ai.aeon_layer import lagrangian, lagrangian_gradient, AeonLayer

# Scalar computation
L = lagrangian(s_a=0.8, s_v=0.6, delta=0.1, t=1.0)

# Gradients
grad = lagrangian_gradient(s_a=0.8, s_v=0.6, delta=0.1, t=1.0)
# → {"dL/ds_a": ..., "dL/ds_v": ..., "dL/dt": ...}

# Layer (wraps advanced_weighting_systems if installed)
layer = AeonLayer(delta=0.1, epsilon=1e-8)
out   = layer.forward(s_a=0.8, s_v=0.6, t=1.0)
grad  = layer.gradient(s_a=0.8, s_v=0.6, t=1.0)
sd    = layer.state_dict()
layer.load_state_dict(sd)
layer.reset_history()

# Factory (with optional AWS base layer)
layer = AeonLayer.from_advanced_weighting_systems(delta=0.05)
```

---

## 2. CREP Quality Metric

### Harmonic Mean (primary)

$$\text{CREP} = \frac{4}{\dfrac{1}{C} + \dfrac{1}{R} + \dfrac{1}{E} + \dfrac{1}{P}}$$

Returns 0 if any dimension is 0.

### Weighted Mean (variant)

$$\text{CREP}_w = w_C \cdot C + w_R \cdot R + w_E \cdot E + w_P \cdot P$$

subject to $w_C + w_R + w_E + w_P = 1$.

### Dimensions

| Dim | Symbol | Heuristic | Range |
|-----|--------|-----------|-------|
| Coherence | $C$ | Inverse normalised variance of signal | $[0,1]$ |
| Resonance | $R$ | Lag-1 autocorrelation mapped to $[0,1]$ | $[0,1]$ |
| Emergence | $E$ | Normalised Shannon entropy of signal | $[0,1]$ |
| Poetics   | $P$ | Type-token ratio of text vocabulary | $[0,1]$ |

**Coherence:**

$$C = \frac{1}{1 + \sigma^2 \cdot s}$$

where $\sigma^2$ is the signal variance and $s$ is `entropy_scale`.

**Resonance:**

$$R = \frac{r_1 + 1}{2}, \quad r_1 = \frac{\sum_{i=1}^{n-1}(x_i - \bar{x})(x_{i-1} - \bar{x})}{(n-1)\,\text{Var}(x)}$$

**Emergence** (normalised Shannon entropy):

$$E = -\frac{s}{H_{\max}} \sum_i p_i \log_2 p_i, \quad H_{\max} = \log_2 n$$

**Poetics** (type-token ratio):

$$P = \frac{|\text{unique tokens}|}{|\text{total tokens}|}$$

### Python API

```python
from aeon_ai.crep_evaluator import CREPEvaluator, CREPScore

ev = CREPEvaluator(weights=(0.25, 0.25, 0.25, 0.25), entropy_scale=1.0)

score = ev.evaluate(
    signal=[0.3, 0.7, 0.5],
    text="aeon of mirrors",
    external={"coherence": 0.9},   # optional per-dimension overrides
)

score.harmonic_mean   # primary CREP score
score.weighted_mean   # weighted variant
score.as_dict()       # flat dict

# Batch
scores = ev.evaluate_batch([{"signal": [0.5]}, {"text": "genesis"}])
ev.reset()
```

---

## 3. UTAC-Logistic

### Definition

$$\text{UTAC}(x) = \frac{L}{1 + e^{-k(x - x_0)}}$$

| Symbol | Param | Default | Description |
|--------|-------|---------|-------------|
| $L$ | `carrying_capacity` | 1.0 | Saturation ceiling |
| $k$ | `growth_rate` | 1.0 | Steepness of transition |
| $x_0$ | `midpoint` | 0.0 | Inflection point |

**Limiting behaviour:**

- $x \to +\infty$: $\text{UTAC}(x) \to L$
- $x \to -\infty$: $\text{UTAC}(x) \to 0$
- $x = x_0$: $\text{UTAC}(x_0) = L/2$

### Python API

```python
from aeon_ai.mirror_core import utac_logistic

val = utac_logistic(x=0.5, carrying_capacity=1.0, growth_rate=2.0, midpoint=0.0)
```

---

## 4. Mirror Machine

### Phase Pipeline

```
input → INIT → REFLECT → INTEGRATE → EMIT → output
```

**INIT:**
$$e_0 = H_b(\text{sigmoid}(x)), \quad H_b(p) = -p\log_2 p - (1-p)\log_2(1-p)$$

**REFLECT** (depth $d$):
$$x_i = \tanh\!\left(x_{i-1} \cdot (1 + e_{i-1} \cdot 0.1 \cdot i)\right), \quad i = 1, \ldots, d$$

**INTEGRATE:**
$$m_{\text{merged}} = x_d \cdot (1 - \lambda) + m_{\text{prev}} \cdot \lambda$$
$$x_{\text{int}} = m_{\text{merged}} \cdot \text{UTAC}(m_{\text{merged}})$$
$$m \leftarrow x_{\text{int}}$$

where $\lambda$ is `memory_decay`.

**EMIT:** pass-through of $x_{\text{int}}$; memory state persisted.

### Python API

```python
from aeon_ai.mirror_core import MirrorCore, MirrorPhase, ReflectionState, utac_logistic

core = MirrorCore(
    depth=3,
    utac_capacity=1.0,
    utac_growth=1.0,
    utac_midpoint=0.0,
    memory_decay=0.9,
)
state = core.reflect(value=0.7, entropy=0.4)
# state.phase == MirrorPhase.EMIT
# state.output_val: float
# state.entropy: float

core.trace      # list[ReflectionState] — all four phases
core.memory     # current context memory
core.reset()
core.state_dict()
```

---

## 5. Sigillin Bridge

### Activation Score

For a sigil with triggers $T = \{t_1, \ldots, t_n\}$ and weight $w$:

$$A(\text{text}) = w \cdot \frac{|\{t \in T : t \text{ matches text}\}|}{|T|}$$

Combined poetic expansion:
$$\text{expand}(\text{text}) = \text{text} \;⟨\; \text{intent}_1 \cdot \text{intent}_2 \cdots ⟩$$

ordered by descending activation score.

### Built-in Sigils

| ID | Name | Phase | Intent |
|----|------|-------|--------|
| `GENESIS` | Genesis Anchor | C | Origin and first-cause initiation |
| `HEIMKEHR` | Homecoming | E | Return, completion, and integration |
| `FRAKTALURS` | Fractal Memory | R | Recursive self-reference and memory persistence |
| `RESO_ECHO` | Resonance Echo | R | Harmonic persistence and symbolic reverberation |
| `MIRROR` | Mirror Gate | P | Self-reflection and inversion |
| `AEON` | Aeon | E | Timeless symbolic intelligence and becoming |

### Python API

```python
from aeon_ai.sigillin_bridge import SigillinBridge, Sigil

bridge = SigillinBridge(load_defaults=True)

# Registration
bridge.register(Sigil(id="MY_SIGIL", name="...", intent="...", triggers=[r"\bmysymbol\b"]))
bridge.unregister("MY_SIGIL")

# Activation
activations = bridge.activate("mirror aeon genesis")   # {id: score}
top         = bridge.top_sigil("mirror self-reflection")
expanded    = bridge.poetic_expansion("aeon rises")
```

---

## 6. Field Bridge

### Composite Field Strength

$$F = \frac{C_f \cdot R_f}{1 + T}$$

where $C_f$ = field coherence, $R_f$ = field resonance, $T$ = topological tension.

### Modulation Factor

$$M = F \cdot \left(1 - \frac{H}{2}\right)$$

where $H$ = field entropy.

### Delta Adjustment

$$\delta' = \delta_{\text{base}} + T \cdot H$$

### Internal Oscillator (standalone fallback)

$$T(t) = 0.5 \cdot |\sin(\omega t)|$$

$$H(t) = H_0 + 0.1 \cdot \sin(0.01 t), \quad \text{clamped to } [0, 1]$$

$$R_f(t) = R_0 \cdot (1 + 0.1 \cos(\omega t)), \quad \text{clamped to } [0, 1]$$

### Python API

```python
from aeon_ai.field_bridge import FieldBridge, CosmicMoment, MediumMode

bridge = FieldBridge(
    base_entropy=0.3,
    tension_frequency=0.1,
    coherence_base=0.8,
    resonance_base=0.75,
)

moment = bridge.sample_moment()          # CosmicMoment
factor = bridge.modulation_factor(moment)
delta  = bridge.adjust_delta(0.0, moment)
ent    = bridge.inject_entropy(moment)

moment.field_strength   # F
moment.as_dict()
bridge.moment_history   # list[CosmicMoment]
bridge.reset()
```

---

## 7. Orchestrator

### Pipeline Equations (one run)

1. $\text{moment} = \text{FieldBridge.sample\_moment}()$
2. $\delta' = \delta_{\text{base}} + T \cdot H$
3. $L = \text{AeonLayer.forward}(S_A, S_V, t;\, \delta')$
4. $r = \text{MirrorCore.reflect}(L,\, H)$
5. $\text{crep} = \text{CREPEvaluator.evaluate}(\text{signal}, \text{text})$
6. $\text{act} = \text{SigillinBridge.activate}(\text{text})$
7. $M = F \cdot (1 - H/2)$
8. $\text{output} = r_{\text{out}} \cdot M$

### Python API

```python
from aeon_ai.agents import Orchestrator, OrchestratorResult

orch = Orchestrator(
    delta=0.0,
    mirror_depth=2,
    crep_weights=(0.25, 0.25, 0.25, 0.25),
    base_entropy=0.3,
    load_sigils=True,
)

result = orch.run(
    s_a=0.8,
    s_v=0.6,
    t=1.0,
    sigil_text="mirror aeon genesis",
    signal=[0.8, 0.6],
    crep_external=None,
    metadata={"model": "trans"},
)

result.lagrangian_out        # float: L value
result.reflection            # ReflectionState (EMIT phase)
result.crep_score            # CREPScore
result.sigil_activations     # {id: score}
result.cosmic_moment         # CosmicMoment
result.modulation            # float: M
result.final_output          # float: reflection.output_val * M
result.as_dict()             # nested dict

orch.run_batch([{"s_a": 0.5, "s_v": 0.5}, ...])
orch.state_dict()
orch.reset()
```

---

---

## 8. Phase-Transition Detection *(v0.2.0)*

### UTAC Trigger Function

$$\Phi_{\text{trigger}}(H) = \frac{L}{1 + e^{-k(H - H_0)}}$$

| Symbol | Param | Default | Description |
|--------|-------|---------|-------------|
| $H$ | `entropy` | — | Current Shannon entropy input |
| $H_0$ | `entropy_threshold` | 0.37 | Pivot entropy threshold |
| $k$ | `utac_growth` | 6.0 | UTAC sigmoid steepness |
| $L$ | `utac_capacity` | 1.0 | Carrying capacity |

A **UTAC_TRIGGER** event fires when $\Phi(H) \geq \theta_{\text{ceil}}$ (`utac_trigger_ceil`, default 0.85).

### Collapse Condition

$$\Delta_{\text{collapse}} = \left| x_n - x_{n-1} \right| < \epsilon_{\text{stab}}$$

AND entropy below threshold:

$$H_n < H_0$$

### Phase-Label Mapping

| UTAC value range | Label |
|-----------------|-------|
| $\Phi < 0.3$ | `STABLE` |
| $0.3 \leq \Phi < 0.7$ | `TRANSITIONING` |
| $\Phi \geq 0.7$ | `COLLAPSE_RISK` |

### Python API

```python
from aeon_ai.phase_detector import (
    PhaseDetector, PhaseTransitionEvent, TransitionType,
    detect_phases_from_core, entropy_phase_label,
)
from aeon_ai.mirror_core import MirrorCore

# Create detector
detector = PhaseDetector(
    entropy_threshold=0.37,
    stability_floor=1e-4,
    utac_trigger_ceil=0.85,
    utac_capacity=1.0,
    utac_growth=6.0,
)

# Process a complete mirror trace
core = MirrorCore(depth=2)
state = core.reflect(0.6, entropy=0.4)
events = detector.process_trace(core.trace)

# Per-state detection
ev = detector.detect_transition(state)   # PhaseTransitionEvent | None

# Collapse detection
is_collapsed = detector.detect_collapse(core.trace)

# UTAC query
val     = detector.utac_value_at(0.45)  # float
fired   = detector.utac_trigger_check(0.45)  # bool

# Force a transition event
ev = detector.force_transition(MirrorPhase.INIT, MirrorPhase.EMIT, entropy=0.5)

# One-shot convenience function
events = detect_phases_from_core(core.trace, entropy_threshold=0.37)

# Entropy → label
label = entropy_phase_label(0.5)  # "STABLE" | "TRANSITIONING" | "COLLAPSE_RISK"

# Introspection
detector.transition_history    # list[PhaseTransitionEvent]
detector.state_dict()
detector.reset()
```

---

## 9. Native Self-Reflection Loop *(v0.2.0)*

### Iteration Equations

At each iteration $i = 0, \ldots, 6$:

1. **Lagrangian:** $L_i = \text{AeonLayer.forward}(S_A^{(i)}, S_V^{(i)}, t)$

2. **Mirror:** $r_i = \text{MirrorCore.reflect}(L_i,\, H_{i-1})$

3. **CREP:** $\text{CREP}_i = \text{CREPEvaluator.evaluate}([S_A^{(i)}, S_V^{(i)}, L_i, r_i])$

4. **Sigillin:** $\text{act}_i = \text{SigillinBridge.activate}(\text{text})$

5. **Update:**
$$S_A^{(i+1)} = S_A^{(i)} + \eta_i \cdot \frac{\partial L}{\partial S_A}$$
$$S_V^{(i+1)} = S_V^{(i)} + \eta_i \cdot \frac{\partial L}{\partial S_V}$$

where $\eta_i = \eta_0 \cdot (1 + 0.1 \cdot \max(\text{act}_i))$.

6. **Entropy update:**
$$H_i = \text{clip}\!\left(\text{CREP}_i \cdot H_0 + (1 - \text{CREP}_i) \cdot r_i^{\text{entropy}},\;[0.01, 0.99]\right)$$

7. **Convergence check** (requires $i \geq 1$):
$$|\text{CREP}_i - \text{CREP}_{i-1}| < \varepsilon = 10^{-4}$$

### Python API

```python
from aeon_ai.self_reflection import SelfReflector, ReflectionLoopResult, MAX_ITER

reflector = SelfReflector(
    delta=0.0,
    mirror_depth=2,
    step_size=0.05,
    crep_weights=(0.25, 0.25, 0.25, 0.25),
    sigil_text="aeon mirror genesis",
)

result = reflector.self_reflect(
    s_a=0.7,
    s_v=0.6,
    t=1.0,
    entropy_threshold=0.37,   # H_0
    sigil_text=None,           # use constructor default
    metadata={"experiment": "run1"},
)

# MAX_ITER = 7 (hard upper bound)
result.converged            # bool
result.total_iterations     # int, <= MAX_ITER
result.final_crep           # CREPScore
result.final_lagrangian     # float
result.final_s_a, result.final_s_v  # updated amplitudes
result.snapshots            # list[IterationSnapshot]
result.as_dict()            # JSON-serialisable dict

# Per-snapshot inspection
snap = result.snapshots[0]
snap.iteration, snap.s_a, snap.s_v
snap.lagrangian_value, snap.crep_score
snap.reflection             # ReflectionState
snap.sigil_activations      # {id: score}
snap.entropy, snap.converged
snap.as_dict()

# History
reflector.loop_history      # list[ReflectionLoopResult]
reflector.state_dict()
reflector.reset()
```

## 10. CLI Reference

```
aeon [OPTIONS] COMMAND [ARGS]...

Commands:
  reflect         Run the AeonAI self-reflection pipeline
  detect-phase    Detect phase-transition state for entropy value  (v0.2.0)
  info            Display package information and component status
  sigils          List all registered sigils

aeon reflect [OPTIONS]
  --models TEXT        Comma-separated model ids (default: trans)
  --sigil TEXT         Poetic sigil / trigger text (default: "")
  --entropy FLOAT      External entropy hint in [0,1] (default: 0.3)
  --s-a FLOAT          Auditory amplitude S_A (default: 0.7)
  --s-v FLOAT          Visual amplitude S_V (default: 0.6)
  --delta FLOAT        Lagrangian deformation parameter delta (default: 0.0)
  --time-step FLOAT    Time step t > 0 (default: 1.0)
  --loop               Run native self-reflection closed loop (v0.2.0)
  --phases             Show real-time phase-transition events (v0.2.0)
  --visualize          Render mandala visualisation + sonification
  --json               Output raw JSON result

aeon detect-phase [OPTIONS]                                           (v0.2.0)
  --entropy FLOAT      Field entropy in [0,1] to analyse (default: 0.37)
  --threshold FLOAT    Entropy pivot H0 (default: 0.37)
  --utac-ceil FLOAT    UTAC trigger ceiling in (0,1] (default: 0.85)
  --json               Output raw JSON
```

---

## 11. Contract Tests (GenesisAeon Stack)

When the optional stack packages are installed, the following contracts are enforced:

### advanced-weighting-systems contract

`AeonLayer.from_advanced_weighting_systems()` must:
- Return an instance implementing `forward(s_a, s_v) -> float`
- Return an instance implementing `state_dict() -> dict`

### fieldtheory contract

`FieldBridge._try_fieldtheory(t)` must return a dict with:
- `timestamp`, `entropy`, `tension`, `coherence`, `resonance`, `medium`

### mirror-machine contract

`MirrorCore` must:
- Accept `depth >= 1`
- Produce exactly 4 `ReflectionState` objects per `reflect()` call
- Persist memory across calls

### phase-detector contract (v0.2.0)

`PhaseDetector` must:
- Return `None` from `detect_transition()` when no condition fires
- Emit `PhaseTransitionEvent` with correct `TransitionType`
- Operate standalone when `mirror-machine` package is absent

### self-reflector contract (v0.2.0)

`SelfReflector.self_reflect()` must:
- Execute at most `MAX_ITER = 7` iterations
- Return a `ReflectionLoopResult` with `converged` flag
- Couple CREP score, Lagrangian gradient, and Sigillin activations at each step
- Clip updated signal amplitudes to $[-10, 10]$
