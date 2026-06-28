# Changelog

All notable changes to `aeon-ai` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] – 2026-06-28

### Changed

- **License**: relicensed as dual-licensed — source code under
  **GPL-3.0-or-later** ([`LICENSE-CODE`](./LICENSE-CODE)), documentation
  under **CC BY 4.0** ([`LICENSE-DOCS`](./LICENSE-DOCS)). Previously
  released versions remain available under the original MIT license.
- Bumped GenesisAeon-ecosystem dependency pins (`mirror-machine`,
  `entropy-governance`, `sigillin`, `utac-core`, `mandala-visualizer`,
  `cosmic-web`, `advanced-weighting-systems`, `fieldtheory`) to `>=1.0.0`.

### Added

- Standardized release tooling for the GenesisAeon v1.0.0 ecosystem
  milestone: `RELEASE_GUIDE.md`, `.github/ISSUE_TEMPLATE/`,
  `.github/PULL_REQUEST_TEMPLATE.md`.
- Updated `.zenodo.json` metadata for the v1.0.0 release.

## [0.1.0] – 2026-03-19

### Added

- **AeonLayer** – resonance container with fieldtheory Lagrangian dynamics and analytical gradients
- **MirrorCore** – UTAC-Logistic mirror with MirrorPhase transition support
- **CREPEvaluator** – harmonic-mean quality metric (Coherence, Resonance, Emergence, Poetics)
- **SigillinBridge** – 6 built-in Genesis sigils with symbolic reflection engine
- **FieldBridge** – CosmicMoment detection, MediumMode encoding, entropy-bridge
- **Orchestrator** (`aeon_ai.agents`) – unified-mandala neural adapter combining all components
- **CLI** (`aeon reflect`, `aeon info`, `aeon sigils`) powered by Typer + Rich
- Full test suite: 195 tests, 99.71% branch coverage
- ruff-clean code (E/W/F/I/N/UP/B/C4/SIM/TCH/ANN/D/PT)
- mkdocs-material documentation with KaTeX math rendering
- `stack` optional-dependencies for full GenesisAeon integration

[0.1.0]: https://github.com/GenesisAeon/aeon-ai/releases/tag/v0.1.0
