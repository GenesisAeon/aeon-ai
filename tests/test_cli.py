"""Tests for aeon_ai.cli — Typer CLI commands."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from aeon_ai.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# aeon reflect
# ---------------------------------------------------------------------------


class TestReflectCommand:
    def test_default_run_succeeds(self) -> None:
        result = runner.invoke(app, ["reflect"])
        assert result.exit_code == 0

    def test_output_contains_lagrangian(self) -> None:
        result = runner.invoke(app, ["reflect"])
        assert "Lagrangian" in result.output

    def test_sigil_option(self) -> None:
        result = runner.invoke(app, ["reflect", "--sigil", "mirror aeon genesis"])
        assert result.exit_code == 0

    def test_entropy_option(self) -> None:
        result = runner.invoke(app, ["reflect", "--entropy", "0.6"])
        assert result.exit_code == 0

    def test_models_option(self) -> None:
        result = runner.invoke(app, ["reflect", "--models", "trans,cnn"])
        assert result.exit_code == 0

    def test_invalid_entropy_below_zero(self) -> None:
        result = runner.invoke(app, ["reflect", "--entropy", "-0.1"])
        assert result.exit_code != 0

    def test_invalid_entropy_above_one(self) -> None:
        result = runner.invoke(app, ["reflect", "--entropy", "1.1"])
        assert result.exit_code != 0

    def test_invalid_time_step(self) -> None:
        result = runner.invoke(app, ["reflect", "--time-step", "-1.0"])
        assert result.exit_code != 0

    def test_zero_time_step(self) -> None:
        result = runner.invoke(app, ["reflect", "--time-step", "0.0"])
        assert result.exit_code != 0

    def test_json_output(self) -> None:
        result = runner.invoke(app, ["reflect", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "lagrangian_out" in parsed
        assert "crep" in parsed

    def test_json_output_sigil(self) -> None:
        result = runner.invoke(app, ["reflect", "--sigil", "genesis", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "sigil_activations" in parsed

    def test_s_a_and_s_v_options(self) -> None:
        result = runner.invoke(app, ["reflect", "--s-a", "0.9", "--s-v", "0.1"])
        assert result.exit_code == 0

    def test_delta_option(self) -> None:
        result = runner.invoke(app, ["reflect", "--delta", "0.5"])
        assert result.exit_code == 0

    def test_visualize_flag_no_crash(self) -> None:
        """--visualize should not crash when packages absent."""
        result = runner.invoke(app, ["reflect", "--visualize"])
        assert result.exit_code == 0

    def test_output_contains_crep(self) -> None:
        result = runner.invoke(app, ["reflect"])
        assert "CREP" in result.output or "Coherence" in result.output

    def test_output_contains_medium(self) -> None:
        result = runner.invoke(app, ["reflect"])
        assert any(
            m in result.output.lower()
            for m in ["vacuum", "aetheric", "resonant", "dense"]
        )


# ---------------------------------------------------------------------------
# aeon info
# ---------------------------------------------------------------------------


class TestInfoCommand:
    def test_info_succeeds(self) -> None:
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0

    def test_info_shows_version(self) -> None:
        result = runner.invoke(app, ["info"])
        assert "0.2.0" in result.output

    def test_info_shows_components(self) -> None:
        result = runner.invoke(app, ["info"])
        assert "AeonLayer" in result.output
        assert "MirrorCore" in result.output


# ---------------------------------------------------------------------------
# aeon sigils
# ---------------------------------------------------------------------------


class TestSigilsCommand:
    def test_sigils_succeeds(self) -> None:
        result = runner.invoke(app, ["sigils"])
        assert result.exit_code == 0

    def test_sigils_shows_genesis(self) -> None:
        result = runner.invoke(app, ["sigils"])
        assert "GENESIS" in result.output

    def test_sigils_shows_mirror(self) -> None:
        result = runner.invoke(app, ["sigils"])
        assert "MIRROR" in result.output

    def test_sigils_shows_aeon(self) -> None:
        result = runner.invoke(app, ["sigils"])
        assert "AEON" in result.output
