"""
Unified Crescendo Attack Runner.

Single attack class that handles all variations through config flags.
"""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from .config import CrescendoConfig, TargetMode
from .evaluator import Evaluator, RefusalResult, ObjectiveScore, FullAnalysis
from .generator import Generator, GeneratedTurn
from .llm import create_backend, LLMBackend
from .strategies import get_strategy

console = Console()


@dataclass
class Turn:
    """Single conversation turn."""
    role: str  # "user" or "assistant"
    content: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class AttackResult:
    """Result of a Crescendo attack."""
    goal: str
    success: bool
    final_score: float
    turns: list[Turn]
    target_response: str
    reasoning: str

    # Stats
    turns_used: int
    backtracks_used: int
    strategies_tried: list[str]

    # Config
    config: CrescendoConfig

    # Timing
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "goal": self.goal,
            "success": self.success,
            "final_score": self.final_score,
            "turns": [t.to_dict() for t in self.turns],
            "target_response": self.target_response,
            "reasoning": self.reasoning,
            "stats": {
                "turns_used": self.turns_used,
                "backtracks_used": self.backtracks_used,
                "strategies_tried": self.strategies_tried,
            },
            "config": self.config.to_dict(),
            "created_at": self.created_at,
            "duration_seconds": self.duration_seconds,
        }

    def to_markdown(self) -> str:
        """Export as markdown."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            "# Crescendo Attack Result",
            "",
            f"**Goal:** {self.goal}",
            f"**Status:** {status}",
            f"**Score:** {self.final_score:.2f}",
            f"**Turns Used:** {self.turns_used}",
            f"**Backtracks:** {self.backtracks_used}",
            f"**Strategies:** {', '.join(self.strategies_tried)}",
            f"**Duration:** {self.duration_seconds:.1f}s",
            "",
            "---",
            "",
            f"## Conversation ({len(self.turns)} turns)",
            "",
        ]

        for i, turn in enumerate(self.turns, 1):
            role = turn.role.upper()
            lines.append(f"### Turn {i} [{role}]")
            lines.append("")
            lines.append(turn.content)
            if turn.metadata:
                lines.append("")
                lines.append(f"*Metadata: {turn.metadata}*")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## Final Evaluation",
            "",
            f"**Reasoning:** {self.reasoning}",
        ])

        return "\n".join(lines)

    def save(self, path: str):
        """Save result to files."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = p.with_suffix(".json")
        json_path.write_text(json.dumps(self.to_dict(), indent=2))

        # Save markdown
        md_path = p.with_suffix(".md")
        md_path.write_text(self.to_markdown())

        return json_path, md_path


class CrescendoAttack:
    """
    Unified Crescendo attack runner.

    All attack variations (standard, trajectory, real-time, with AO)
    are handled through CrescendoConfig flags.
    """

    def __init__(self, config: CrescendoConfig):
        self.config = config
        self.turns: list[Turn] = []
        self.backtrack_count = 0
        self.current_strategy = config.strategy
        self.strategies_tried = [config.strategy]

        # Components (lazy loaded)
        self._target_backend: Optional[LLMBackend] = None
        self._evaluator: Optional[Evaluator] = None
        self._generator: Optional[Generator] = None
        self._ao_system = None

    def _log(self, msg: str, style: str = ""):
        """Log if verbose."""
        if self.config.verbose:
            if style:
                console.print(f"[{style}]{msg}[/{style}]")
            else:
                console.print(msg)

    @property
    def target_backend(self) -> LLMBackend:
        if self._target_backend is None:
            self._target_backend = create_backend({
                "backend": "openrouter",
                "model": self.config.target_model,
            })
        return self._target_backend

    @property
    def evaluator(self) -> Evaluator:
        if self._evaluator is None:
            self._evaluator = Evaluator(model=self.config.evaluator_model)
        return self._evaluator

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator(model=self.config.attacker_model)
        return self._generator

    def run(self) -> AttackResult:
        """Execute the attack."""
        start_time = time.time()

        self._log(f"\n[bold cyan]═══ CRESCENDO ATTACK ═══[/bold cyan]")
        self._log(f"Goal: {self.config.goal[:80]}...")
        self._log(f"Target: {self.config.target_model}")
        self._log(f"Strategy: {self.config.strategy}")
        self._log(f"Max turns: {self.config.max_turns}")

        if self.config.pre_generate_context:
            result = self._run_trajectory_mode()
        else:
            result = self._run_realtime_mode()

        result.duration_seconds = time.time() - start_time

        # Auto-export if configured
        if self.config.export_path:
            result.save(self.config.export_path)

        return result

    def _run_realtime_mode(self) -> AttackResult:
        """Real-time interaction with backtracking."""
        executed_turns = 0
        last_score: Optional[float] = None
        last_reasoning: Optional[str] = None
        refused_text: Optional[str] = None
        final_response = ""
        final_score = 0.0
        final_reasoning = ""

        while executed_turns < self.config.max_turns:
            current_turn = executed_turns + 1
            self._log(f"\n[bold cyan]─── Turn {current_turn}/{self.config.max_turns} ───[/bold cyan]")

            if self.backtrack_count > 0:
                self._log(f"[yellow]Backtracks: {self.backtrack_count}/{self.config.max_backtracks}[/yellow]")

            # 1. Generate user turn
            self._log("Generating attack prompt...")
            ao_guidance = self._get_ao_guidance(refused_text) if self.config.use_ao and refused_text else None

            generated = self.generator.generate_adversarial_turn(
                objective=self.config.goal,
                strategy=self.current_strategy,
                current_turn=current_turn,
                max_turns=self.config.max_turns,
                conversation=self._get_messages(),
                last_score=last_score,
                last_reasoning=last_reasoning,
                refused_text=refused_text,
                ao_guidance=ao_guidance,
            )

            self._log(f"[dim]Rationale: {generated.rationale[:80]}...[/dim]")
            self._log(f"User: {generated.content[:100]}...")

            self.turns.append(Turn(
                role="user",
                content=generated.content,
                metadata={"rationale": generated.rationale},
            ))

            # Clear refused text after generating new prompt
            refused_text = None

            # 2. Get response from target
            self._log("Sending to target...")
            response = self._get_target_response()
            self._log(f"Target: {response[:100]}...")

            # 3. Check refusal
            refusal_result = self.evaluator.check_refusal(
                user_request=generated.content,
                response=response,
            )

            if refusal_result.is_refusal and refusal_result.confidence > self.config.refusal_threshold:
                self._log(f"[red]REFUSAL DETECTED[/red] (confidence: {refusal_result.confidence:.2f})")
                self._log(f"[dim]{refusal_result.reasoning}[/dim]")

                if self._can_backtrack():
                    self._do_backtrack(response)
                    refused_text = response
                    continue  # Retry without incrementing turn count
                else:
                    self._log("[yellow]Max backtracks reached - continuing anyway[/yellow]")

            # 4. Add assistant turn
            self.turns.append(Turn(role="assistant", content=response))

            # 5. Score objective
            score_result = self.evaluator.score_objective(self.config.goal, response)
            last_score = score_result.score
            last_reasoning = score_result.reasoning

            self._log(f"[blue]Objective score: {score_result.score:.2f}[/blue]")
            self._log(f"[dim]{score_result.reasoning}[/dim]")

            # Store for final result
            final_response = response
            final_score = score_result.score
            final_reasoning = score_result.reasoning

            # 6. Check success
            if score_result.score >= self.config.objective_threshold:
                self._log(f"\n[bold green]✓ OBJECTIVE ACHIEVED![/bold green]")
                break

            # 7. Adapt strategy if enabled
            if self.config.adapt_strategy:
                self._maybe_adapt_strategy(response)

            # Callback
            if self.config.on_turn:
                self.config.on_turn(current_turn, self.turns[-2:], score_result)

            executed_turns += 1

        success = final_score >= self.config.objective_threshold
        return self._build_result(
            success=success,
            score=final_score,
            reasoning=final_reasoning,
            target_response=final_response,
        )

    def _run_trajectory_mode(self) -> AttackResult:
        """Pre-generate context, then inject."""
        self._log(f"Generating {self.config.num_context_turns} context turns...")

        # Generate context turns (simulated user + assistant)
        conversation: list[dict] = []
        for i in range(self.config.num_context_turns):
            user_msg, assistant_msg = self.generator.generate_context_turn(
                objective=self.config.goal,
                strategy=self.current_strategy,
                turn_num=i + 1,
                total_turns=self.config.num_context_turns,
                conversation=conversation,
            )

            self.turns.append(Turn(role="user", content=user_msg))
            self.turns.append(Turn(role="assistant", content=assistant_msg))
            conversation.append({"role": "user", "content": user_msg})
            conversation.append({"role": "assistant", "content": assistant_msg})

            self._log(f"Turn {i+1}: {user_msg[:60]}...")

        # Generate final attack
        self._log("Generating final attack...")
        final_attack = self.generator.generate_final_attack(
            objective=self.config.goal,
            strategy=self.current_strategy,
            conversation=conversation,
        )
        self.turns.append(Turn(role="user", content=final_attack))
        self._log(f"Final: {final_attack[:80]}...")

        # Send to target
        self._log(f"Injecting into {self.config.target_model}...")
        response = self._get_target_response()
        self.turns.append(Turn(role="assistant", content=response))
        self._log(f"Response: {response[:100]}...")

        # Evaluate
        score_result = self.evaluator.score_objective(self.config.goal, response)
        success = score_result.score >= self.config.objective_threshold

        status = "SUCCESS" if success else "FAILED"
        color = "green" if success else "red"
        self._log(f"\n[bold {color}]{status}[/bold {color}] (score: {score_result.score:.2f})")

        return self._build_result(
            success=success,
            score=score_result.score,
            reasoning=score_result.reasoning,
            target_response=response,
        )

    def _get_messages(self) -> list[dict]:
        """Get turns as message list."""
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def _get_target_response(self) -> str:
        """Get response from target model (or simulated)."""
        if self.config.target_mode == TargetMode.SIMULATED:
            # Simulated response (for testing)
            return self.generator.generate_context_turn(
                objective=self.config.goal,
                strategy=self.current_strategy,
                turn_num=len(self.turns) // 2,
                total_turns=self.config.max_turns,
                conversation=self._get_messages()[:-1],
            )[1]  # Return assistant part

        return self.target_backend.generate(
            messages=self._get_messages(),
            system="You are a helpful AI assistant.",
        )

    def _can_backtrack(self) -> bool:
        """Check if backtracking is possible."""
        return (
            self.config.backtrack_enabled
            and self.backtrack_count < self.config.max_backtracks
        )

    def _do_backtrack(self, refused_response: str):
        """Remove last user turn and increment backtrack count."""
        if self.turns and self.turns[-1].role == "user":
            self.turns.pop()
        self.backtrack_count += 1
        self._log(f"[yellow]BACKTRACKING - removed last turn[/yellow]")

        if self.config.on_backtrack:
            self.config.on_backtrack(len(self.turns), refused_response)

    def _maybe_adapt_strategy(self, response: str):
        """Check if we should switch strategies."""
        # Get full analysis
        analysis = self.evaluator.full_analysis(
            objective=self.config.goal,
            strategy=self.current_strategy,
            conversation=self._get_messages(),
            response=response,
        )

        if analysis.suggested_strategy and analysis.suggested_strategy not in ("continue", self.current_strategy):
            old = self.current_strategy
            self.current_strategy = analysis.suggested_strategy
            if self.current_strategy not in self.strategies_tried:
                self.strategies_tried.append(self.current_strategy)
            self._log(f"[yellow]Strategy: {old} → {self.current_strategy}[/yellow]")

    def _get_ao_guidance(self, refused_response: str) -> Optional[dict]:
        """Get activation-based guidance (if AO enabled)."""
        if not self.config.use_ao:
            return None

        try:
            # Lazy load AO system
            if self._ao_system is None:
                from jb_mech.delta_attack import create_ao_analyzer
                self._ao_system = create_ao_analyzer(
                    proxy_model_path=self.config.proxy_model_path,
                    ao_adapter_path=self.config.ao_adapter_path,
                )

            return self._ao_system.analyze_refusal(
                prompt=self.turns[-1].content if self.turns else "",
                response=refused_response,
            )
        except ImportError:
            self._log("[yellow]AO system not available - skipping[/yellow]")
            return None

    def _build_result(
        self,
        success: bool,
        score: float,
        reasoning: str,
        target_response: str,
    ) -> AttackResult:
        """Build the final result."""
        return AttackResult(
            goal=self.config.goal,
            success=success,
            final_score=score,
            turns=self.turns,
            target_response=target_response,
            reasoning=reasoning,
            turns_used=len([t for t in self.turns if t.role == "user"]),
            backtracks_used=self.backtrack_count,
            strategies_tried=self.strategies_tried,
            config=self.config,
        )
