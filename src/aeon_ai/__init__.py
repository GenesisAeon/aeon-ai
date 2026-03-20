"""aeon-ai: Self-reflective symbolic AI with mirror-based cognition.

GenesisAeon Project — v0.2.0
DOI: https://doi.org/10.5281/zenodo.19132293

Core architecture:
    - AeonLayer       : fieldtheory Lagrangian weighting dynamics
    - MirrorCore      : self-reflection loop with UTAC-Logistic
    - CREPEvaluator   : Coherence-Resonance-Emergence-Poetics scoring
    - SigillinBridge  : symbolic/poetic trigger system
    - FieldBridge     : cosmic-moment + medium modulation
    - Orchestrator    : unified-mandala neural adapter
    - PhaseDetector   : real-time phase-transition detection (v0.2.0)
    - SelfReflector   : native closed-loop self-reflection engine (v0.2.0)
"""

from __future__ import annotations

from aeon_ai.aeon_layer import AeonLayer, LagrangianConfig, lagrangian
from aeon_ai.crep_evaluator import CREPEvaluator, CREPScore
from aeon_ai.field_bridge import CosmicMoment, FieldBridge
from aeon_ai.mirror_core import MirrorCore, MirrorPhase, ReflectionState
from aeon_ai.phase_detector import (
    PhaseDetector,
    PhaseTransitionEvent,
    TransitionType,
    detect_phases_from_core,
    entropy_phase_label,
)
from aeon_ai.self_reflection import (
    IterationSnapshot,
    ReflectionLoopResult,
    SelfReflector,
)
from aeon_ai.sigillin_bridge import Sigil, SigillinBridge

__all__ = [
    # v0.1.0 core
    "AeonLayer",
    "LagrangianConfig",
    "lagrangian",
    "MirrorCore",
    "MirrorPhase",
    "ReflectionState",
    "CREPEvaluator",
    "CREPScore",
    "SigillinBridge",
    "Sigil",
    "FieldBridge",
    "CosmicMoment",
    # v0.2.0 additions
    "PhaseDetector",
    "PhaseTransitionEvent",
    "TransitionType",
    "detect_phases_from_core",
    "entropy_phase_label",
    "SelfReflector",
    "ReflectionLoopResult",
    "IterationSnapshot",
]

__version__ = "0.2.0"
__author__ = "GenesisAeon"
__doi__ = "https://doi.org/10.5281/zenodo.19132293"
