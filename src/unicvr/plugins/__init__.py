"""Reasoning-side plugins that feed diagnostics back into agent prompts."""

from unicvr.plugins.base import TaskPlugin, collect_blocks
from unicvr.plugins.choice import ChoiceAnswerPlugin
from unicvr.plugins.moc import MocCountingPlugin
from unicvr.plugins.msr import MsrForceChoicePlugin
from unicvr.plugins.registry import PLUGIN_REGISTRY, plugins_for_task
from unicvr.plugins.timing_guard import TimingGuard

__all__ = [
    "TaskPlugin",
    "collect_blocks",
    "ChoiceAnswerPlugin",
    "MocCountingPlugin",
    "MsrForceChoicePlugin",
    "PLUGIN_REGISTRY",
    "plugins_for_task",
    "TimingGuard",
]
