# aeon-ai

**Self-reflective symbolic AI — real-time phase-transition detection & native self-reflection loop — GenesisAeon v0.2.0**

[![PyPI version](https://img.shields.io/pypi/v/aeon-ai.svg)](https://pypi.org/project/aeon-ai/)
[![Python](https://img.shields.io/pypi/pyversions/aeon-ai.svg)](https://pypi.org/project/aeon-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19132293.svg)](https://doi.org/10.5281/zenodo.19132293)
[![Tests](https://github.com/GenesisAeon/aeon-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/GenesisAeon/aeon-ai/actions)

> *”A system that thinks — a memory that sings.”*

`aeon-ai` is the core Python library of the **GenesisAeon** project: a fractal, self-reflective AI
architecture that integrates fieldtheory Lagrangian dynamics, mirror-machine cognition, symbolic
sigil activation, and CREP quality evaluation into a unified pipeline.

---

## Theoretical Foundation

### Fieldtheory Lagrangian — AeonLayer

The core weighting dynamics are governed by the fieldtheory Lagrangian:

$$L(S_A, S_V, \delta, t) = \frac{S_A \cdot S_V}{S_A + S_V} - \frac{1 + \delta}{t^2}$$

| Symbol | Meaning |
|--------|---------|
| $S_A$ | Auditory (abstract) signal amplitude |
| $S_V$ | Visual (visceral) signal amplitude |
| $\delta$ | Deformation / curvature parameter |
| $t$ | Time step ($t > 0$) |

The first term is the **harmonic mean** of the two signal amplitudes, encoding
inter-channel resonance. The second term is a **temporal penalty** that decays
as $t^{-2}$, reflecting the dissipation of symbolic tension over time.

Analytical gradients:

$$\frac{\partial L}{\partial S_A} = \frac{S_V^2}{(S_A + S_V)^2}, \quad
\frac{\partial L}{\partial S_V} = \frac{S_A^2}{(S_A + S_V)^2}, \quad
\frac{\partial L}{\partial t} = \frac{2(1+\delta)}{t^3}$$

---

### CREP Quality Metric — CREPEvaluator

CREP is the four-dimensional symbolic quality metric from the **unified-mandala** stack:

$$\text{CREP} = \frac{4}{\dfrac{1}{C} + \dfrac{1}{R} + \dfrac{1}{E} + \dfrac{1}{P}}$$

| Dimension | Symbol | Description |
|-----------|--------|-------------|
| Coherence | $C$ | Logical and structural consistency |
| Resonance | $R$ | Harmonic alignment between signal components |
| Emergence | $E$ | Novelty and self-organisational complexity |
| Poetics   | $P$ | Aesthetic and symbolic richness |

Each dimension is a float in $[0, 1]$; the combined score is their harmonic mean.

A weighted variant is also available:

$$\text{CREP}_w = \sum_{i \in \{C,R,E,P\}} w_i \cdot \text{dim}_i, \quad \sum w_i = 1$$

---

### UTAC-Logistic — MirrorCore

The Universal Transformation and Adaptation Coefficient governs the INTEGRATE phase
of the Mirror Machine:

$$\text{UTAC}(x) = \frac{L}{1 + e^{-k(x - x_0)}}$$

| Symbol | Meaning |
|--------|---------|
| $L$ | Carrying capacity (saturation ceiling) |
| $k$ | Growth rate / sigmoid steepness |
| $x_0$ | Inflection / midpoint |

---

### Mirror Machine Pipeline

The self-reflection loop processes every input through four canonical phases:

```
INIT → REFLECT → INTEGRATE → EMIT
```

1. **INIT** — Receive raw signal, compute initial entropy
2. **REFLECT** — Apply $n$-deep recursive `tanh`-mirror transformation
3. **INTEGRATE** — Merge with context memory via UTAC-Logistic
4. **EMIT** — Produce observable output, update persistent memory

---

### Phase-Transition Detection — PhaseDetector *(v0.2.0)*

Real-time detection of Mirror Machine phase transitions using a UTAC-Logistic trigger:

$$\Phi_{\text{trigger}}(H) = \frac{L}{1 + e^{-k(H - H_0)}}$$

| Symbol | Meaning |
|--------|---------|
| $H$ | Current Shannon entropy |
| $H_0$ | Pivot entropy threshold (default 0.37) |
| $k$ | UTAC growth rate |
| $L$ | Carrying capacity |

A **collapse** is detected when consecutive output values converge:

$$\Delta_{\text{collapse}} = \left| x_n - x_{n-1} \right| < \epsilon_{\text{stab}}$$

Three transition types are emitted: `FORWARD`, `COLLAPSE`, `UTAC_TRIGGER`, `FORCED`.

---

### Native Self-Reflection Loop — SelfReflector *(v0.2.0)*

A closed-loop iterative engine coupling CREP score, Lagrangian gradient, and Sigillin bridges
over at most **7 iterations** with convergence check:

$$S_A^{(i+1)} = S_A^{(i)} + \eta \cdot \frac{\partial L}{\partial S_A}$$

$$S_V^{(i+1)} = S_V^{(i)} + \eta \cdot \frac{\partial L}{\partial S_V}$$

$$\text{convergence: } \left|\text{CREP}^{(i)} - \text{CREP}^{(i-1)}\right| < \varepsilon$$

Where $\eta$ is the UTAC-modulated step size and CREP is evaluated at each iteration.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                        Orchestrator v0.2.0                      │
│                                                                  │
│  FieldBridge ──▶ CosmicMoment                                   │
│       │                │                                         │
│       ▼                ▼                                         │
│  AeonLayer(δ, t) ──▶ Lagrangian L                               │
│       │                                                          │
│       ▼                                                          │
│  MirrorCore ──▶ INIT → REFLECT → INTEGRATE → EMIT               │
│       │                │                                         │
│       ▼                ▼                                         │
│  PhaseDetector ──▶ PhaseTransitionEvent[]  (v0.2.0)             │
│       │                                                          │
│       ▼                                                          │
│  CREPEvaluator ──▶ CREPScore(C, R, E, P)                        │
│       │                                                          │
│       ▼                                                          │
│  SigillinBridge ──▶ { sigil_id: activation_score }              │
└────────────────────────────────────────────────────────────────┘

SelfReflector (v0.2.0) — closed-loop coupling (max 7 iterations):
  AeonLayer → MirrorCore → CREPEvaluator ← SigillinBridge
       ↑_______________gradient update__________________|
```

---

## Installation

```bash
pip install aeon-ai
```

With the full GenesisAeon stack (mirror-machine, fieldtheory, mandala-visualizer, …):

```bash
pip install 'aeon-ai[stack]'
```

---

## Quick Start

```python
from aeon_ai.agents import Orchestrator

# Full pipeline via Orchestrator
orch = Orchestrator(delta=0.1, mirror_depth=3)
result = orch.run(
    s_a=0.8,
    s_v=0.6,
    sigil_text=”mirror aeon genesis”,
)

print(f”Lagrangian L     = {result.lagrangian_out:.4f}”)
print(f”Reflection out   = {result.reflection.output_val:.4f}”)
print(f”Final output     = {result.final_output:.4f}”)
print(f”CREP score       = {result.crep_score.score:.4f}”)
print(f”Active sigils    = {result.sigil_activations}”)
```

### Individual components

```python
from aeon_ai import AeonLayer, lagrangian

# Direct Lagrangian computation
L = lagrangian(s_a=0.8, s_v=0.6, delta=0.1, t=1.0)

# AeonLayer (with optional advanced_weighting_systems base)
layer = AeonLayer.from_advanced_weighting_systems(delta=0.1)
output = layer.forward(s_a=0.8, s_v=0.6, t=1.0)
grad   = layer.gradient(s_a=0.8, s_v=0.6, t=1.0)
```

```python
from aeon_ai import MirrorCore

core = MirrorCore(depth=3, utac_growth=2.0)
state = core.reflect(0.7, entropy=0.4)
print(state.output_val, state.entropy)
```

```python
from aeon_ai import CREPEvaluator

ev = CREPEvaluator()
score = ev.evaluate(signal=[0.3, 0.7, 0.5, 0.9], text=”aeon of mirrors”)
print(score)  # CREPScore(C=..., R=..., E=..., P=..., score=...)
```

```python
from aeon_ai import SigillinBridge

bridge = SigillinBridge()
activations = bridge.activate(“the mirror reflects the genesis of aeon”)
expanded    = bridge.poetic_expansion(“aeon rises”)
```

---

## CLI

```bash
# Basic reflection run
aeon reflect --models trans,cnn --sigil “mirror aeon genesis” --entropy 0.4

# With custom signal parameters
aeon reflect --s-a 0.9 --s-v 0.5 --delta 0.2 --time-step 2.0

# With mandala visualisation + sonification (requires [stack])
aeon reflect --sigil “origin seed” --entropy 0.6 --visualize

# Machine-readable JSON output
aeon reflect --sigil “mirror” --json

# List all registered sigils
aeon sigils

# Package and stack status
aeon info
```

---

## Module Reference

| Module | Class / Function | Description |
|--------|-----------------|-------------|
| `aeon_ai.aeon_layer` | `AeonLayer` | Lagrangian weighting layer |
| `aeon_ai.aeon_layer` | `lagrangian()` | Fieldtheory Lagrangian function |
| `aeon_ai.mirror_core` | `MirrorCore` | Self-reflection pipeline |
| `aeon_ai.mirror_core` | `utac_logistic()` | UTAC-Logistic function |
| `aeon_ai.crep_evaluator` | `CREPEvaluator` | CREP quality scorer |
| `aeon_ai.sigillin_bridge` | `SigillinBridge` | Sigil activation engine |
| `aeon_ai.field_bridge` | `FieldBridge` | Cosmic-moment modulation |
| `aeon_ai.agents` | `Orchestrator` | Full pipeline coordinator |
| `aeon_ai.phase_detector` | `PhaseDetector` | Real-time phase-transition detector *(v0.2.0)* |
| `aeon_ai.phase_detector` | `detect_phases_from_core()` | One-shot trace analysis *(v0.2.0)* |
| `aeon_ai.phase_detector` | `entropy_phase_label()` | Entropy → phase label *(v0.2.0)* |
| `aeon_ai.self_reflection` | `SelfReflector` | Closed-loop self-reflection engine *(v0.2.0)* |
| `aeon_ai.self_reflection` | `ReflectionLoopResult` | Loop result record *(v0.2.0)* |

Full API documentation: [genesisaeon.github.io/aeon-ai](https://genesisaeon.github.io/aeon-ai)

---

## Development

```bash
git clone https://github.com/GenesisAeon/aeon-ai
cd aeon-ai
pip install -e '.[dev]'

# Tests (99%+ coverage required)
pytest

# Linting
ruff check src tests
ruff format --check src tests

# Docs
mkdocs serve
```

---

## GenesisAeon Stack

`aeon-ai` is designed to interoperate with the full GenesisAeon ecosystem:

| Package | Role |
|---------|------|
| `advanced-weighting-systems` | Base AeonLayer weights |
| `fieldtheory` | Cosmological field dynamics |
| `mirror-machine` | Deep mirror-pass kernels |
| `entropy-governance` | Adaptive entropy regulation |
| `sigillin` | Extended sigil registry |
| `utac-core` | UTAC kernel implementations |
| `mandala-visualizer` | Mandala network rendering |
| `cosmic-web` | Field sonification |

All packages are optional; `aeon-ai` operates standalone with its own implementations.

---

## Citation

If you use `aeon-ai` in academic work, please cite:

```bibtex
@software{aeon_ai_2025,
  author    = {GenesisAeon},
  title     = {aeon-ai: Real-time phase-transition detection and native self-reflection loop},
  version   = {0.2.0},
  year      = {2026},
  doi       = {10.5281/zenodo.19132293},
  url       = {https://github.com/GenesisAeon/aeon-ai}
}
```

DOI: [10.5281/zenodo.19132293](https://doi.org/10.5281/zenodo.19132293)

---

## License

[![MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
MIT — see [LICENSE](LICENSE).
