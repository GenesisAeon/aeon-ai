"""aeon-ai CLI: symbolic self-reflection at the command line.

Usage::

    aeon reflect --models trans,cnn --sigil "mirror aeon genesis" --entropy 0.4
    aeon reflect --models trans --sigil "origin" --entropy 0.6 --visualize
    aeon info
"""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="aeon",
    help="AeonAI — self-reflective symbolic AI CLI (GenesisAeon v0.1.0)",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_models(models_str: str) -> list[str]:
    """Split comma-separated model names into a list.

    Args:
        models_str: Comma-separated model identifiers, e.g. ``"trans,cnn"``.

    Returns:
        List of stripped, lower-cased model name strings.
    """
    return [m.strip().lower() for m in models_str.split(",") if m.strip()]


def _build_orchestrator(entropy: float, delta: float) -> Orchestrator:  # type: ignore[name-defined]  # noqa: F821
    """Build an Orchestrator with entropy-derived field settings.

    Args:
        entropy: External entropy hint.
        delta:   Lagrangian deformation parameter.

    Returns:
        Configured :class:`~aeon_ai.agents.Orchestrator` instance.
    """
    from aeon_ai.agents import Orchestrator

    return Orchestrator(delta=delta, base_entropy=entropy)


def _try_visualize(result: OrchestratorResult) -> None:  # type: ignore[name-defined]  # noqa: F821
    """Attempt to render mandala visualisation and sonification.

    Delegates to ``mandala-visualizer`` and ``cosmic-web`` if installed.
    Prints a notice when those packages are absent.

    Args:
        result: Pipeline result to visualise.
    """
    visualized = False
    try:
        import mandala_visualizer  # type: ignore[import-untyped]

        mandala_visualizer.render(result.as_dict())
        visualized = True
    except ImportError:
        pass

    try:
        import cosmic_web  # type: ignore[import-untyped]

        cosmic_web.sonify(result.cosmic_moment.as_dict())
        visualized = True
    except ImportError:
        pass

    if not visualized:
        rprint(
            "[yellow]Tip:[/yellow] Install [bold]mandala-visualizer[/bold] and "
            "[bold]cosmic-web[/bold] for full visualisation & sonification:\n"
            "  pip install 'aeon-ai[stack]'"
        )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def reflect(
    models: Annotated[
        str,
        typer.Option("--models", "-m", help="Comma-separated model ids, e.g. trans,cnn"),
    ] = "trans",
    sigil: Annotated[
        str,
        typer.Option("--sigil", "-s", help="Poetic sigil / trigger text"),
    ] = "",
    entropy: Annotated[
        float,
        typer.Option("--entropy", "-e", help="External entropy hint ∈ [0, 1]"),
    ] = 0.3,
    s_a: Annotated[
        float,
        typer.Option("--s-a", help="Auditory signal amplitude S_A"),
    ] = 0.7,
    s_v: Annotated[
        float,
        typer.Option("--s-v", help="Visual signal amplitude S_V"),
    ] = 0.6,
    delta: Annotated[
        float,
        typer.Option("--delta", help="Lagrangian deformation parameter δ"),
    ] = 0.0,
    time_step: Annotated[
        float,
        typer.Option("--time-step", "-t", help="Time step for Lagrangian (> 0)"),
    ] = 1.0,
    visualize: Annotated[
        bool,
        typer.Option("--visualize", "-v", help="Render mandala visualisation + sonification"),
    ] = False,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Output raw JSON result"),
    ] = False,
) -> None:
    """Run the AeonAI self-reflection pipeline.

    Processes input signals through the full aeon-ai symbolic pipeline:
    AeonLayer → MirrorCore → CREPEvaluator → SigillinBridge → FieldBridge.
    """
    if not 0.0 <= entropy <= 1.0:
        rprint("[red]Error:[/red] --entropy must be in [0, 1]")
        raise typer.Exit(1)
    if time_step <= 0.0:
        rprint("[red]Error:[/red] --time-step must be > 0")
        raise typer.Exit(1)

    model_list = _parse_models(models)
    orch = _build_orchestrator(entropy=entropy, delta=delta)

    if not output_json:
        rprint(
            Panel(
                f"[bold cyan]AeonAI Reflection[/bold cyan]\n"
                f"Models: {model_list}  |  Sigil: {sigil!r}  |  Entropy: {entropy}",
                title="[bold magenta]GenesisAeon[/bold magenta]",
                expand=False,
            )
        )

    try:
        result = orch.run(
            s_a=s_a,
            s_v=s_v,
            t=time_step,
            sigil_text=sigil,
            metadata={"models": model_list},
        )
    except ValueError as exc:
        rprint(f"[red]Pipeline error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if output_json:
        print(json.dumps(result.as_dict(), indent=2, default=str))
        return

    # Rich table output
    table = Table(title="Reflection Result", show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Lagrangian L", f"{result.lagrangian_out:.6f}")
    table.add_row("Reflection Output", f"{result.reflection.output_val:.6f}")
    table.add_row("Final Output (modulated)", f"{result.final_output:.6f}")
    table.add_row("Entropy", f"{result.reflection.entropy:.4f}")
    table.add_row("CREP Score", f"{result.crep_score.score:.4f}")
    table.add_row("  Coherence (C)", f"{result.crep_score.coherence:.4f}")
    table.add_row("  Resonance (R)", f"{result.crep_score.resonance:.4f}")
    table.add_row("  Emergence (E)", f"{result.crep_score.emergence:.4f}")
    table.add_row("  Poetics (P)", f"{result.crep_score.poetics:.4f}")
    table.add_row("Field Strength", f"{result.cosmic_moment.field_strength:.4f}")
    table.add_row("Modulation Factor", f"{result.modulation:.4f}")
    table.add_row("Medium", result.cosmic_moment.medium.value)

    if result.sigil_activations:
        activations_str = ", ".join(
            f"{sid}:{score:.2f}" for sid, score in sorted(result.sigil_activations.items())
        )
        table.add_row("Active Sigils", activations_str)
    else:
        table.add_row("Active Sigils", "(none)")

    rprint(table)

    if visualize:
        _try_visualize(result)


@app.command()
def info() -> None:
    """Display aeon-ai package information and component status."""
    from aeon_ai import __doi__, __version__

    components = {
        "AeonLayer": "aeon_ai.aeon_layer",
        "MirrorCore": "aeon_ai.mirror_core",
        "CREPEvaluator": "aeon_ai.crep_evaluator",
        "SigillinBridge": "aeon_ai.sigillin_bridge",
        "FieldBridge": "aeon_ai.field_bridge",
        "Orchestrator": "aeon_ai.agents.orchestrator",
    }

    stack_packages = [
        "advanced_weighting_systems",
        "fieldtheory",
        "mirror_machine",
        "entropy_governance",
        "sigillin",
        "utac_core",
        "mandala_visualizer",
        "cosmic_web",
    ]

    rprint(
        Panel(
            f"[bold]aeon-ai[/bold] v{__version__}\n"
            f"DOI: {__doi__}\n"
            f"[italic]Self-reflective symbolic AI — GenesisAeon Project[/italic]",
            title="[bold magenta]AeonAI Info[/bold magenta]",
        )
    )

    table = Table(title="Core Components", header_style="bold blue")
    table.add_column("Component", style="cyan")
    table.add_column("Module", style="dim")
    table.add_column("Status", style="green")

    for name, module in components.items():
        try:
            __import__(module)
            status = "[green]✓[/green]"
        except ImportError:
            status = "[red]✗[/red]"
        table.add_row(name, module, status)

    rprint(table)

    stack_table = Table(title="[stack] Optional Dependencies", header_style="bold blue")
    stack_table.add_column("Package", style="cyan")
    stack_table.add_column("Available", style="green")

    for pkg in stack_packages:
        try:
            __import__(pkg)
            avail = "[green]✓[/green]"
        except ImportError:
            avail = "[yellow]–[/yellow]"
        stack_table.add_row(pkg, avail)

    rprint(stack_table)


@app.command()
def sigils() -> None:
    """List all registered sigils in the SigillinBridge registry."""
    from aeon_ai.sigillin_bridge import SigillinBridge

    bridge = SigillinBridge(load_defaults=True)

    table = Table(title="Registered Sigils", header_style="bold blue")
    table.add_column("ID", style="cyan bold")
    table.add_column("Name", style="white")
    table.add_column("Phase", style="magenta")
    table.add_column("Weight", style="yellow")
    table.add_column("Intent", style="dim")

    for sid, sigil in sorted(bridge.sigils.items()):
        table.add_row(
            sid,
            sigil.name,
            sigil.phase or "—",
            f"{sigil.weight:.2f}",
            sigil.intent[:60] + ("…" if len(sigil.intent) > 60 else ""),
        )

    rprint(table)


def main() -> None:
    """Entry point for the ``aeon`` CLI command."""
    app()


if __name__ == "__main__":
    main()
