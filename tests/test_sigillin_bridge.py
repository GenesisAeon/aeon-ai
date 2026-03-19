"""Tests for aeon_ai.sigillin_bridge — Sigil + SigillinBridge."""

from __future__ import annotations

import pytest

from aeon_ai.sigillin_bridge import Sigil, SigillinBridge

# ---------------------------------------------------------------------------
# Sigil
# ---------------------------------------------------------------------------


class TestSigil:
    def test_creation(self) -> None:
        s = Sigil(id="TEST", name="Test Sigil", intent="A test", triggers=["foo"])
        assert s.id == "TEST"
        assert s.weight == 1.0

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="id must not be empty"):
            Sigil(id="", name="x", intent="x")

    def test_invalid_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            Sigil(id="X", name="X", intent="X", weight=1.5)

    def test_matches_trigger(self) -> None:
        s = Sigil(id="X", name="X", intent="X", triggers=[r"\bfoo\b"])
        assert s.matches("foo bar")

    def test_no_match_empty_triggers(self) -> None:
        s = Sigil(id="X", name="X", intent="X", triggers=[])
        assert not s.matches("anything")

    def test_case_insensitive_match(self) -> None:
        s = Sigil(id="X", name="X", intent="X", triggers=[r"\baeon\b"])
        assert s.matches("AEON is alive")

    def test_case_sensitive_mismatch(self) -> None:
        s = Sigil(id="X", name="X", intent="X", triggers=[r"\baeon\b"])
        assert not s.matches("AEON", case_sensitive=True)

    def test_activation_score_zero_no_match(self) -> None:
        s = Sigil(id="X", name="X", intent="X", triggers=[r"\bfoo\b"], weight=0.8)
        assert s.activation_score("bar baz") == 0.0

    def test_activation_score_full_match(self) -> None:
        s = Sigil(id="X", name="X", intent="X", triggers=[r"\bfoo\b"], weight=0.8)
        assert abs(s.activation_score("foo") - 0.8) < 1e-9

    def test_activation_score_partial_match(self) -> None:
        s = Sigil(id="X", name="X", intent="X", triggers=[r"\bfoo\b", r"\bbar\b"], weight=1.0)
        score = s.activation_score("foo baz")
        assert abs(score - 0.5) < 1e-9

    def test_activation_score_no_triggers(self) -> None:
        s = Sigil(id="X", name="X", intent="X", triggers=[])
        assert s.activation_score("foo") == 0.0


# ---------------------------------------------------------------------------
# SigillinBridge
# ---------------------------------------------------------------------------


class TestSigillinBridge:
    def test_default_sigils_loaded(self, sigillin_bridge: SigillinBridge) -> None:
        assert len(sigillin_bridge.sigils) > 0

    def test_genesis_sigil_present(self, sigillin_bridge: SigillinBridge) -> None:
        assert "GENESIS" in sigillin_bridge.sigils

    def test_mirror_sigil_present(self, sigillin_bridge: SigillinBridge) -> None:
        assert "MIRROR" in sigillin_bridge.sigils

    def test_aeon_sigil_present(self, sigillin_bridge: SigillinBridge) -> None:
        assert "AEON" in sigillin_bridge.sigils

    def test_register_custom_sigil(self, sigillin_bridge: SigillinBridge) -> None:
        custom = Sigil(id="CUSTOM", name="Custom", intent="Testing", triggers=[r"\bcustom\b"])
        sigillin_bridge.register(custom)
        assert "CUSTOM" in sigillin_bridge.sigils

    def test_register_overwrites(self, sigillin_bridge: SigillinBridge) -> None:
        s1 = Sigil(id="DUP", name="Dup1", intent="First")
        s2 = Sigil(id="DUP", name="Dup2", intent="Second")
        sigillin_bridge.register(s1)
        sigillin_bridge.register(s2)
        assert sigillin_bridge.sigils["DUP"].name == "Dup2"

    def test_unregister(self, sigillin_bridge: SigillinBridge) -> None:
        sigillin_bridge.unregister("GENESIS")
        assert "GENESIS" not in sigillin_bridge.sigils

    def test_unregister_missing_raises(self, sigillin_bridge: SigillinBridge) -> None:
        with pytest.raises(KeyError):
            sigillin_bridge.unregister("NONEXISTENT_SIGIL")

    def test_activate_mirror_text(self, sigillin_bridge: SigillinBridge) -> None:
        activations = sigillin_bridge.activate("the mirror reflects everything")
        assert "MIRROR" in activations
        assert activations["MIRROR"] > 0.0

    def test_activate_returns_only_nonzero(self, sigillin_bridge: SigillinBridge) -> None:
        activations = sigillin_bridge.activate("zzz yyy xyz no trigger here")
        # All values should be positive
        assert all(v > 0.0 for v in activations.values())

    def test_activate_empty_text(self, sigillin_bridge: SigillinBridge) -> None:
        activations = sigillin_bridge.activate("")
        assert isinstance(activations, dict)

    def test_top_sigil_mirror(self, sigillin_bridge: SigillinBridge) -> None:
        top = sigillin_bridge.top_sigil("mirror self-reflection gateway")
        assert top is not None
        assert top.id == "MIRROR"

    def test_top_sigil_none_on_no_match(self, sigillin_bridge: SigillinBridge) -> None:
        # Register fresh bridge with no defaults, guaranteed no match
        bridge = SigillinBridge(load_defaults=False)
        bridge.register(Sigil(id="X", name="X", intent="X", triggers=[r"\bspecific_token_xyz\b"]))
        assert bridge.top_sigil("completely unrelated text") is None

    def test_poetic_expansion_adds_intent(self, sigillin_bridge: SigillinBridge) -> None:
        expanded = sigillin_bridge.poetic_expansion("mirror genesis aeon")
        assert "⟨" in expanded

    def test_poetic_expansion_no_match_returns_original(self) -> None:
        bridge = SigillinBridge(load_defaults=False)
        text = "zzz yyy no match"
        assert bridge.poetic_expansion(text) == text

    def test_sigils_property_is_copy(self, sigillin_bridge: SigillinBridge) -> None:
        orig_len = len(sigillin_bridge.sigils)
        copy = sigillin_bridge.sigils
        copy.clear()
        assert len(sigillin_bridge.sigils) == orig_len

    def test_no_defaults(self) -> None:
        bridge = SigillinBridge(load_defaults=False)
        assert len(bridge.sigils) == 0

    def test_repr(self, sigillin_bridge: SigillinBridge) -> None:
        assert "SigillinBridge" in repr(sigillin_bridge)
