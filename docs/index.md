# aeon-ai

**DOI**: [10.5281/zenodo.19139280](https://doi.org/10.5281/zenodo.19139280)
**Zenodo**: [https://zenodo.org/records/19139280](https://zenodo.org/records/19139280)

**Self-reflective symbolic AI — GenesisAeon Project v0.1.0**

> *"A system that thinks — a memory that sings."*

`aeon-ai` is the core Python library implementing the first mirror-based, symbolic self-reflective
AI architecture in the GenesisAeon project.

## Key Components

- **[AeonLayer](reference.md#1-fieldtheory-lagrangian)** — fieldtheory Lagrangian $L = S_A S_V/(S_A+S_V) - (1+\delta)/t^2$
- **[MirrorCore](reference.md#4-mirror-machine)** — four-phase self-reflection loop with UTAC-Logistic
- **[CREPEvaluator](reference.md#2-crep-quality-metric)** — Coherence, Resonance, Emergence, Poetics scoring
- **[SigillinBridge](reference.md#5-sigillin-bridge)** — symbolic sigil activation with poetic triggers
- **[FieldBridge](reference.md#6-field-bridge)** — cosmic-moment field modulation
- **[Orchestrator](reference.md#7-orchestrator)** — unified pipeline coordinator

## Quick Install

```bash
pip install aeon-ai
# Full stack:
pip install 'aeon-ai[stack]'
```

## Quick Start

```python
from aeon_ai.agents import Orchestrator

orch = Orchestrator(delta=0.1, mirror_depth=3)
result = orch.run(s_a=0.8, s_v=0.6, sigil_text="mirror aeon genesis")
print(result.crep_score.score)
```

See the [API Reference](reference.md) for complete documentation.
