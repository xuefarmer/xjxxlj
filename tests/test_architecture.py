from __future__ import annotations

import ast
import inspect
from pathlib import Path

from unicvr.agents import ComparerAgent, ObserverAgent, ReasonerAgent
from unicvr.core.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_exactly_three_logical_agent_classes_exist() -> None:
    discovered: set[str] = set()
    for source_path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        discovered.update(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name.endswith("Agent")
        )
    # The three data-axis agents must exist
    assert discovered >= {"ObserverAgent", "ReasonerAgent", "ComparerAgent"}
    # Verify the classes are importable
    assert {ObserverAgent, ReasonerAgent, ComparerAgent}


def test_pipeline_contains_no_task_specific_router() -> None:
    source = inspect.getsource(Pipeline).lower()
    forbidden = ("task_type", "task_metadata", '"fsa"', '"moc"', '"pss"', "router")
    assert all(token not in source for token in forbidden)


def test_agent_public_interfaces_are_exact() -> None:
    assert "observe" in ObserverAgent.__dict__
    assert "form" in ReasonerAgent.__dict__
    assert "review" in ReasonerAgent.__dict__
    assert "compare" in ComparerAgent.__dict__
