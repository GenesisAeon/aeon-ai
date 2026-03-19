"""aeon_ai.agents: Orchestrator adapter and agent infrastructure.

Provides the :class:`~aeon_ai.agents.orchestrator.Orchestrator` which
coordinates AeonLayer, MirrorCore, CREPEvaluator, SigillinBridge, and
FieldBridge into a unified symbolic-AI pipeline.
"""

from __future__ import annotations

from aeon_ai.agents.orchestrator import Orchestrator, OrchestratorResult

__all__ = ["Orchestrator", "OrchestratorResult"]
