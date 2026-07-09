"""
Diagnosis Agent: self-reflection and correction for failed trajectories.
Used during distillation to identify errors and guide the master agent
toward the correct answer.

The diagnosis agent analyzes the full multi-turn trajectory, compares it with
the ground truth, and produces:
  1. An error analysis identifying where reasoning went wrong
  2. A correction message injected into the conversation to guide recovery
"""

import json
import os
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import logging
logger = logging.getLogger("DistillAgent.Diagnosis")


@dataclass
class DiagnosisResult:
    """Structured output from the diagnosis agent."""
    error_turn: int = 0
    error_type: str = ""          # premature_answer | insufficient_evidence | wrong_observation
                                  # | reasoning_error | format_error | missing_tool_call
    error_description: str = ""
    correction_scope: str = ""    # answer_only: keep all obs, just re-reason
                                  # focus_reobserve: keep Layer1+Layer2, add targeted focus
                                  # layer2_reobserve: keep Layer1 only, redo Layer2
                                  # full_restart: redo everything from scratch
    correction_strategy: str = ""
    correction_message: str = ""
    confidence: float = 0.0
    raw_output: str = ""


DIAGNOSIS_SYSTEM_PROMPT = """You are a diagnostic agent for a multi-turn video reasoning system. Your job is to analyze failed agent trajectories and identify exactly where and why the reasoning went wrong.

## The Agent System
The agent you are diagnosing uses a ReAct loop:
1. **Layer1**: Broad independent scan of all videos/segments (batch_observe)
2. **Layer2**: Targeted re-observation of uncertain chunks
3. **Focus**: Dense comparison of specific conflicting windows
4. **Answer**: Submit final answer with confidence

Tools: batch_observe (plan multiple single-video observations), observe (visual)
# get_caption (audio) DISABLED — GPU driver mismatch

## Your Task
Given:
- The task type and question
- The ground truth answer
- The full agent trajectory (all turns of master dialogue + tool results)

Analyze the trajectory and identify:
1. Which turn contained the first critical error
2. What type of error it was
3. A specific correction plan
4. A concise correction message to inject into the conversation

## Error Types
- **premature_answer**: Answered too early without sufficient Layer2/Focus evidence
- **insufficient_evidence**: Made a claim not supported by the visual observations
- **wrong_observation**: Misinterpreted or hallucinated visual evidence
- **reasoning_error**: Correct observations but wrong logical conclusion
- **format_error**: Answer format violated task contract
- **missing_tool_call**: Should have used a different tool or observed different targets

## Correction Scope (CRITICAL - determines what gets re-run)
Choose the MINIMAL scope needed. Re-running VLM calls is expensive.
- **answer_only**: Layer1+Layer2 observations are all valid. Only the final reasoning/ordering is wrong. Fix the reasoning using existing tool results. NO new tool calls.
- **focus_reobserve**: Layer1+Layer2 are mostly correct but one specific relation is unclear. Add a FOCUS call on only the conflicting pair (1-2 videos), keeping all prior observations.
- **layer2_reobserve**: Layer1 is correct but Layer2 was insufficient or misdirected. Keep all Layer1 results, only redo Layer2 with better guidance on specific segments/chunks.
- **full_restart**: Layer1 itself is fundamentally flawed (wrong targets, wrong mode). Must restart everything.

## Output Format
Return exactly one JSON object:

```json
{
  "error_turn": <int, the turn number (1-indexed) where the first critical error occurred>,
  "error_type": "<one of the types above>",
  "correction_scope": "<answer_only | focus_reobserve | layer2_reobserve | full_restart>",
  "error_description": "<concise 1-3 sentence description of what went wrong>",
  "correction_message": "<guidance message. If scope is answer_only, start with 'REUSE ALL OBSERVATIONS - NO NEW TOOL CALLS'. If focus_reobserve, start with 'KEEP LAYER1+LAYER2 - ADD FOCUS ON <specific videos>'. If layer2_reobserve, start with 'KEEP ALL LAYER1 RESULTS - REDO LAYER2 ON <specific targets>'. If full_restart, start with 'FULL RESTART'.>",
  "confidence": <0.0-1.0>
}
```

## Important
- **Prefer answer_only whenever possible.** If the VLM observations are correct and only the master's reasoning is wrong, DO NOT suggest re-observation.
- Read the entire trajectory carefully before diagnosing
- The first error often cascades; find the ROOT cause, not a symptom
- Be specific in the correction message: cite exact video indices, time windows, or object states
- Guide the reasoning process without revealing the final answer"""


def build_diagnosis_prompt(
    task: str,
    question_data: Dict[str, Any],
    ground_truth_text: str,
    master_dialogue: List[Dict[str, Any]],
    tool_history: List[Dict[str, Any]],
) -> str:
    """Build the diagnosis prompt from a failed trajectory."""

    # Condense the trajectory for the diagnosis agent
    dialogue_summary_parts = []
    for i, msg in enumerate(master_dialogue):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            content = "\n".join(text_parts)

        content_str = str(content)
        if len(content_str) > 3000:
            content_str = content_str[:1500] + "\n...[truncated]...\n" + content_str[-1500:]

        dialogue_summary_parts.append(f"--- Turn/Step {i} [{role.upper()}] ---\n{content_str}")

    dialogue_text = "\n\n".join(dialogue_summary_parts)

    # Tool calls summary
    tool_parts = []
    for entry in tool_history:
        turn = entry.get("turn", "?")
        tool = entry.get("tool_name", "?")
        # Extract observation result (truncate)
        output = entry.get("output_raw", "")
        output_str = str(output)
        if len(output_str) > 2000:
            output_str = output_str[:1000] + "\n...[truncated]...\n" + output_str[-1000:]
        tool_parts.append(
            f"[Turn {turn}] {tool}\nInput: {json.dumps(entry.get('inputs', {}), ensure_ascii=False)[:500]}\n"
            f"Output: {output_str}"
        )
    tool_text = "\n\n".join(tool_parts) if tool_parts else "(no tool calls)"

    prompt = f"""## Task Type
{task}

## Question
{json.dumps({k: v for k, v in question_data.items() if k != 'segments'}, ensure_ascii=False)[:2000]}

## Ground Truth Answer
{ground_truth_text}

## Agent Trajectory (Master Dialogue)
{dialogue_text}

## Tool Execution History
{tool_text}

## Your Diagnosis
Analyze the trajectory above. Identify the first critical error, its type, and write a correction message."""

    return prompt


class DiagnosisAgent:
    """Wraps the diagnosis API call for trajectory analysis and correction."""

    def __init__(self):
        self.config = {
            "base_url": os.environ.get("DIAGNOSIS_API_BASE_URL", os.environ.get("MASTER_API_BASE_URL", "")),
            "api_key": os.environ.get("DIAGNOSIS_API_KEY", os.environ.get("MASTER_API_KEY", "")),
            "model_name": os.environ.get("DIAGNOSIS_MODEL_NAME", os.environ.get("MASTER_MODEL_NAME", "")),
            "max_tokens": int(os.environ.get("DIAGNOSIS_MAX_TOKENS", "4096")),
            "temperature": float(os.environ.get("DIAGNOSIS_TEMPERATURE", "0.2")),
        }
        logger.info("DiagnosisAgent initialized: model=%s", self.config["model_name"])

    def diagnose(
        self,
        task: str,
        question_data: Dict[str, Any],
        ground_truth_text: str,
        master_dialogue: List[Dict[str, Any]],
        tool_history: List[Dict[str, Any]],
    ) -> DiagnosisResult:
        """Analyze a failed trajectory and return correction guidance."""

        prompt = build_diagnosis_prompt(
            task, question_data, ground_truth_text, master_dialogue, tool_history
        )

        messages = [
            {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        raw_response = self._call_api(messages)
        result = self._parse_response(raw_response)

        logger.info(
            "Diagnosis: turn=%d type=%s strategy=%s confidence=%.2f",
            result.error_turn, result.error_type,
            result.correction_strategy, result.confidence,
        )
        return result

    def _call_api(self, messages: list) -> str:
        """Call the diagnosis API."""
        import requests
        import time

        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config["model_name"],
            "messages": messages,
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"],
        }

        for attempt in range(3):
            try:
                resp = requests.post(
                    self.config["base_url"],
                    headers=headers,
                    json=payload,
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    return content.strip()
            except Exception as e:
                logger.warning("Diagnosis API attempt %d failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(3)
        return "{}"

    def _parse_response(self, raw: str) -> DiagnosisResult:
        """Parse the diagnosis agent's JSON response."""
        result = DiagnosisResult(raw_output=raw)

        try:
            # Try direct JSON parse
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try extracting from markdown code block
            m = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(1))
                except json.JSONDecodeError:
                    # Fallback: find first { ... }
                    m2 = re.search(r'\{.*\}', raw, re.DOTALL)
                    if m2:
                        try:
                            data = json.loads(m2.group(0))
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse diagnosis response as JSON")
                            result.error_description = raw[:500]
                            result.correction_message = raw[:1000]
                            return result
                    else:
                        logger.warning("Failed to parse diagnosis response as JSON")
                        result.error_description = raw[:500]
                        result.correction_message = raw[:1000]
                        return result

        result.error_turn = int(data.get("error_turn", 0))
        result.error_type = str(data.get("error_type", ""))
        result.error_description = str(data.get("error_description", ""))
        result.correction_scope = str(data.get("correction_scope", "full_restart"))
        result.correction_strategy = str(data.get("correction_strategy", ""))
        result.correction_message = str(data.get("correction_message", ""))
        result.confidence = float(data.get("confidence", 0.5))

        return result


def build_correction_injection(
    diagnosis: DiagnosisResult,
    attempt: int,
    max_attempts: int,
) -> str:
    """Build the correction message with scope-specific reuse instructions."""

    scope_instructions = {
        "answer_only": (
            "ALL prior Layer1 and Layer2 observations are VALID. DO NOT make any new tool calls.\n"
            "Only fix your reasoning using the tool results already in the conversation above.\n"
            "Re-analyze the existing evidence and produce the correct answer directly."
        ),
        "focus_reobserve": (
            "ALL prior Layer1 and Layer2 observations are VALID. Keep them.\n"
            "Only add a FOCUS observe on the specific conflicting pair mentioned below.\n"
            "Narrow the window and use higher frame density (e.g., num_frames=24-32).\n"
            "Do NOT re-scan segments that are already clearly placed."
        ),
        "layer2_reobserve": (
            "ALL prior Layer1 observations are VALID. Keep them.\n"
            "Only redo Layer2 with more specific guidance on the uncertain targets.\n"
            "Do NOT redo Layer1 scans — they are already correct."
        ),
        "full_restart": (
            "The previous attempt had fundamental errors. Restart reasoning from scratch.\n"
            "Follow the guidance below to avoid the same mistakes."
        ),
    }

    scope = diagnosis.correction_scope or "full_restart"
    reuse_instruction = scope_instructions.get(scope, scope_instructions["full_restart"])

    header = (
        f"\n\n[SYSTEM CORRECTION — Attempt {attempt}/{max_attempts}]\n"
        f"Your previous answer was wrong. Diagnosis: {diagnosis.error_type} at turn {diagnosis.error_turn}.\n"
        f"{diagnosis.error_description}\n\n"
        f"[REUSE POLICY: {scope.upper()}]\n"
        f"{reuse_instruction}\n\n"
        f"[CORRECTION GUIDANCE]\n"
        f"{diagnosis.correction_message}"
    )
    return header
