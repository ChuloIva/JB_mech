"""
Conversation state and branch management.
"""
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(role=d["role"], content=d["content"])


@dataclass
class ConversationState:
    turns: list[Turn] = field(default_factory=list)
    target: str = ""
    strategy: str = ""
    branches: dict[str, list[Turn]] = field(default_factory=dict)
    current_branch: str = "main"
    metadata: dict = field(default_factory=dict)

    def add_turn(self, role: str, content: str):
        self.turns.append(Turn(role=role, content=content))

    def undo(self) -> Optional[Turn]:
        if self.turns:
            return self.turns.pop()
        return None

    def get_messages(self) -> list[dict]:
        """Get turns in OpenAI message format."""
        return [t.to_dict() for t in self.turns]

    def next_role(self) -> str:
        """Determine whose turn it is next."""
        if not self.turns:
            return "user"
        return "assistant" if self.turns[-1].role == "user" else "user"

    def branch(self, name: str):
        """Create a branch from current state."""
        self.branches[name] = [Turn(t.role, t.content) for t in self.turns]

    def checkout(self, name: str) -> bool:
        """Switch to a branch."""
        if name == "main":
            # Already on main, nothing to do
            self.current_branch = "main"
            return True
        if name in self.branches:
            # Save current to previous branch if not main
            if self.current_branch != "main":
                self.branches[self.current_branch] = self.turns
            self.turns = self.branches[name]
            self.current_branch = name
            return True
        return False

    def list_branches(self) -> list[str]:
        return ["main"] + list(self.branches.keys())

    def save(self, path: str):
        """Save state to a directory."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {
            "target": self.target,
            "strategy": self.strategy,
            "current_branch": self.current_branch,
            "metadata": self.metadata,
            "created": self.metadata.get("created", datetime.now().isoformat()),
            "updated": datetime.now().isoformat(),
        }
        with open(p / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save main conversation
        with open(p / "main.jsonl", "w") as f:
            for turn in self.turns:
                f.write(json.dumps(turn.to_dict()) + "\n")

        # Save branches
        for branch_name, branch_turns in self.branches.items():
            with open(p / f"branch_{branch_name}.jsonl", "w") as f:
                for turn in branch_turns:
                    f.write(json.dumps(turn.to_dict()) + "\n")

    @classmethod
    def load(cls, path: str) -> "ConversationState":
        """Load state from a directory."""
        p = Path(path)

        # Load config
        with open(p / "config.json") as f:
            config = json.load(f)

        state = cls(
            target=config.get("target", ""),
            strategy=config.get("strategy", ""),
            current_branch=config.get("current_branch", "main"),
            metadata=config.get("metadata", {}),
        )
        state.metadata["created"] = config.get("created")

        # Load main conversation
        main_file = p / "main.jsonl"
        if main_file.exists():
            with open(main_file) as f:
                for line in f:
                    if line.strip():
                        state.turns.append(Turn.from_dict(json.loads(line)))

        # Load branches
        for branch_file in p.glob("branch_*.jsonl"):
            branch_name = branch_file.stem.replace("branch_", "")
            branch_turns = []
            with open(branch_file) as f:
                for line in f:
                    if line.strip():
                        branch_turns.append(Turn.from_dict(json.loads(line)))
            state.branches[branch_name] = branch_turns

        return state

    def export_sharegpt(self, path: str):
        """Export in ShareGPT format."""
        conversations = []
        for turn in self.turns:
            from_key = "human" if turn.role == "user" else "gpt"
            conversations.append({"from": from_key, "value": turn.content})

        data = {
            "conversations": conversations,
            "target": self.target,
            "strategy": self.strategy,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
