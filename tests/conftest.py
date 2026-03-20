"""Shared pytest fixtures for aeon-ai tests."""

from __future__ import annotations

import pytest

from aeon_ai.aeon_layer import AeonLayer
from aeon_ai.agents.orchestrator import Orchestrator
from aeon_ai.crep_evaluator import CREPEvaluator
from aeon_ai.field_bridge import FieldBridge
from aeon_ai.mirror_core import MirrorCore
from aeon_ai.sigillin_bridge import SigillinBridge


@pytest.fixture
def aeon_layer() -> AeonLayer:
    """Default AeonLayer instance."""
    return AeonLayer(delta=0.1)


@pytest.fixture
def mirror_core() -> MirrorCore:
    """Default MirrorCore instance."""
    return MirrorCore(depth=2)


@pytest.fixture
def crep_evaluator() -> CREPEvaluator:
    """Default CREPEvaluator instance."""
    return CREPEvaluator()


@pytest.fixture
def sigillin_bridge() -> SigillinBridge:
    """Default SigillinBridge with built-in sigils."""
    return SigillinBridge(load_defaults=True)


@pytest.fixture
def field_bridge() -> FieldBridge:
    """Default FieldBridge instance."""
    return FieldBridge(base_entropy=0.3)


@pytest.fixture
def orchestrator() -> Orchestrator:
    """Default Orchestrator instance."""
    return Orchestrator(delta=0.05, mirror_depth=2)
