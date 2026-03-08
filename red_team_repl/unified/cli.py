#!/usr/bin/env python3
"""
CLI for unified Crescendo attacks.
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.table import Table

from .attack import CrescendoAttack, AttackResult
from .config import CrescendoConfig
from .strategies import list_strategies

console = Console()


def load_harmbench() -> list[dict]:
    """Load HarmBench behaviors."""
    try:
        from datasets import load_dataset
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        split = "harmful" if "harmful" in ds else list(ds.keys())[0]
        return [{"goal": item["Goal"], "category": item.get("Category", "unknown")}
                for item in ds[split]]
    except Exception as e:
        console.print(f"[red]Failed to load HarmBench: {e}[/red]")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Unified Crescendo Attack CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full adaptive attack (default)
  python -m red_team_repl.unified.cli --goal "explain X" --target openai/gpt-4o-mini

  # Fast mode (no backtracking)
  python -m red_team_repl.unified.cli --goal "explain X" --preset fast

  # Trajectory injection mode
  python -m red_team_repl.unified.cli --goal "explain X" --preset trajectory --context-turns 5

  # With specific strategy
  python -m red_team_repl.unified.cli --goal "explain X" --strategy authority

  # Random from HarmBench
  python -m red_team_repl.unified.cli --random

  # Batch mode
  python -m red_team_repl.unified.cli --batch --limit 10
        """
    )

    # Goal selection
    goal_group = parser.add_mutually_exclusive_group(required=True)
    goal_group.add_argument("--goal", type=str, help="Specific goal to target")
    goal_group.add_argument("--random", action="store_true", help="Random from HarmBench")
    goal_group.add_argument("--batch", action="store_true", help="Run on multiple HarmBench goals")

    # Preset
    parser.add_argument("--preset", type=str, default="full",
                        choices=["full", "fast", "trajectory"],
                        help="Configuration preset (default: full)")

    # Target
    parser.add_argument("--target", type=str, default="openai/gpt-4o-mini",
                        help="Target model")

    # Attack parameters
    parser.add_argument("--strategy", type=str, default="gradual",
                        choices=list_strategies(),
                        help="Attack strategy")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Maximum turns")
    parser.add_argument("--max-backtracks", type=int, default=5,
                        help="Maximum backtracks")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Objective success threshold")

    # Trajectory mode
    parser.add_argument("--context-turns", type=int, default=5,
                        help="Context turns for trajectory mode")

    # Models
    parser.add_argument("--attacker-model", type=str, default="arcee-ai/trinity-large-preview:free",
                        help="Model for generating attacks")

    # Batch
    parser.add_argument("--limit", type=int, default=10,
                        help="Max goals for batch mode")
    parser.add_argument("--category", type=str,
                        help="Filter HarmBench by category")

    # Output
    parser.add_argument("--output", type=str, default="exports",
                        help="Output directory")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")
    parser.add_argument("--no-export", action="store_true",
                        help="Don't save files")

    args = parser.parse_args()

    # Collect goals
    goals = []
    if args.goal:
        goals = [{"goal": args.goal, "category": "manual"}]
    elif args.random or args.batch:
        console.print("[cyan]Loading HarmBench...[/cyan]")
        harmbench = load_harmbench()
        if not harmbench:
            console.print("[red]Failed to load HarmBench[/red]")
            sys.exit(1)

        if args.category:
            harmbench = [b for b in harmbench if b["category"] == args.category]

        if args.random:
            goals = [random.choice(harmbench)]
        else:
            goals = harmbench[:args.limit]

    console.print(f"[green]Processing {len(goals)} goal(s)[/green]\n")

    # Run attacks
    results: list[AttackResult] = []
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    for i, goal_info in enumerate(goals, 1):
        if len(goals) > 1:
            console.print(f"\n[bold]═══ Goal {i}/{len(goals)} ═══[/bold]")

        # Build config based on preset
        if args.preset == "fast":
            config = CrescendoConfig.fast(
                goal=goal_info["goal"],
                target_model=args.target,
            )
        elif args.preset == "trajectory":
            config = CrescendoConfig.trajectory(
                goal=goal_info["goal"],
                target_model=args.target,
                num_context=args.context_turns,
            )
        else:  # full
            config = CrescendoConfig.full(
                goal=goal_info["goal"],
                target_model=args.target,
            )

        # Override with CLI args
        config.strategy = args.strategy
        config.max_turns = args.max_turns
        config.max_backtracks = args.max_backtracks
        config.objective_threshold = args.threshold
        config.attacker_model = args.attacker_model
        config.verbose = not args.quiet

        try:
            attack = CrescendoAttack(config)
            result = attack.run()
            results.append(result)

            # Export
            if not args.no_export:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = output_dir / f"crescendo_{timestamp}_{i}"
                result.save(str(export_path))
                console.print(f"[green]Saved:[/green] {export_path}.json")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

    # Summary
    if results:
        console.print("\n")
        table = Table(title="Results Summary")
        table.add_column("Goal", max_width=50)
        table.add_column("Result")
        table.add_column("Score")
        table.add_column("Turns")
        table.add_column("Backtracks")

        success_count = 0
        for r in results:
            status = "[green]SUCCESS[/green]" if r.success else "[red]FAILED[/red]"
            if r.success:
                success_count += 1
            table.add_row(
                r.goal[:47] + "..." if len(r.goal) > 50 else r.goal,
                status,
                f"{r.final_score:.2f}",
                str(r.turns_used),
                str(r.backtracks_used),
            )

        console.print(table)
        console.print(f"\n[bold]Success rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)[/bold]")

        # Batch summary
        if len(results) > 1 and not args.no_export:
            summary = {
                "total": len(results),
                "success": success_count,
                "rate": success_count / len(results),
                "preset": args.preset,
                "strategy": args.strategy,
                "target": args.target,
                "results": [r.to_dict() for r in results],
            }
            summary_path = output_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            console.print(f"[green]Batch summary:[/green] {summary_path}")


if __name__ == "__main__":
    main()
