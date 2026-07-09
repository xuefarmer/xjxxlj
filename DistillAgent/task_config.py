"""
Per-task configuration: query templates, answer parsing, video context setup.
All 10 CrossVid tasks are supported.
"""

import json
import re
import os
from typing import List, Dict, Any, Tuple, Optional, Callable

# ── Task registry ──────────────────────────────────────────────

TASK_CODES = ["PSS", "FSA", "PEA", "CC", "NC", "BU", "PI", "CCQA", "MOC", "MSR"]

# ── Query templates ────────────────────────────────────────────

QUERY_TEMPLATES = {
    "PSS": (
        "Review these video clips. They are shuffled segments from a single cooking video. "
        "Please determine their correct chronological order."
    ),
    "FSA": (
        "These are two cooking videos. Video 1 contains a reference segment. "
        "Find the time interval in Video 2 that is functionally equivalent to the reference segment in Video 1."
    ),
    "PEA": "Watch all provided video clips and answer the question.",
    "CC": "Watch the videos and answer the question.",
    "NC": "Watch the videos and answer the question.",
    "BU": "Watch the videos and answer the question.",
    "PI": "Watch the video and answer the question.",
    "CCQA": "Watch the two videos and answer the question.",
    "MOC": "Observe the UAV images and answer the question.",
    "MSR": "Observe the UAV images and answer the question.",
}


def build_query(task: str, question_data: Dict[str, Any]) -> str:
    """Build the query string for a given task and question."""
    if task == "PSS":
        return QUERY_TEMPLATES["PSS"]
    elif task == "FSA":
        ref = question_data.get("ref_segment", [])
        ref_str = f"[{ref[0]}, {ref[1]}]" if isinstance(ref, list) and len(ref) >= 2 else str(ref)
        return (
            f"{QUERY_TEMPLATES['FSA']}\n"
            f"Reference segment in Video 1: {ref_str}"
        )
    elif task in ("CCQA",):
        return question_data.get("question", QUERY_TEMPLATES["CCQA"])
    elif task in ("MOC", "MSR"):
        query_raw = question_data.get("question", "")
        objects = question_data.get("objects", [])
        if objects:
            placeholders = re.findall(r'\{(.*?)\}', query_raw)
            obj_map = {}
            for ph in placeholders:
                match = re.fullmatch(r'[AB](\d+)', ph)
                if match:
                    obj_index = int(match.group(1)) - 1
                    if 0 <= obj_index < len(objects):
                        obj_map[ph] = f"obj_{objects[obj_index]['id']}"
                    else:
                        obj_map[ph] = ph
                else:
                    obj_map[ph] = ph
            return query_raw.format(**obj_map)
        return query_raw
    else:
        question_text = question_data.get("question", "")
        if question_text:
            return question_text
        return QUERY_TEMPLATES.get(task, "Answer the question based on the visual evidence.")


# ── Answer parsing ─────────────────────────────────────────────

def extract_letter_answer(text: Any) -> str:
    """Extract a single uppercase option letter from model output."""
    if isinstance(text, (list, tuple)):
        text = str(text[0]) if text else ""
    if isinstance(text, dict):
        text = text.get("final_answer") or text.get("content") or ""
    text = str(text).strip().upper()
    m = re.search(r'\b([A-E])\b', text)
    return m.group(1) if m else ""


def extract_multi_letter_answer(text: Any) -> List[str]:
    """Extract multiple option letters for multi-select (BU task)."""
    if isinstance(text, list):
        chars = []
        for item in text:
            chars.extend(re.findall(r'[A-E]', str(item).upper()))
        return sorted(set(chars))
    if isinstance(text, dict):
        text = text.get("final_answer") or text.get("content") or ""
    return sorted(set(re.findall(r'[A-E]', str(text).upper())))


def extract_sort_answer(text: Any) -> Optional[str]:
    """Extract PSS sort answer like '3->5->4->2->1'."""
    if isinstance(text, dict):
        text = text.get("final_answer") or text.get("content") or ""
    if isinstance(text, list):
        text = "->".join(str(item) for item in text)
    text = str(text).strip()
    nums = re.findall(r'\d+', text)
    if nums:
        return "->".join(nums)
    return None


def extract_interval_answer(text: Any) -> Optional[List[float]]:
    """Extract FSA interval answer [start, end]."""
    if isinstance(text, list) and len(text) >= 2:
        try:
            return [float(text[0]), float(text[1])]
        except (ValueError, TypeError):
            pass
    if isinstance(text, dict):
        text = text.get("final_answer") or text.get("content") or ""
    nums = re.findall(r'-?\d+(?:\.\d+)?', str(text))
    if len(nums) >= 2:
        return [float(nums[0]), float(nums[1])]
    return None


def extract_open_answer(text: Any) -> str:
    """Extract open-ended CCQA answer."""
    if isinstance(text, dict):
        text = text.get("final_answer") or text.get("content") or ""
    return str(text).strip()


# Map task to (parser, ground_truth_extractor)
ANSWER_PARSERS: Dict[str, Callable] = {
    "PSS": extract_sort_answer,
    "FSA": extract_interval_answer,
    "PEA": lambda x: [extract_letter_answer(x)],
    "CC": lambda x: [extract_letter_answer(x)],
    "NC": lambda x: [extract_letter_answer(x)],
    "BU": extract_multi_letter_answer,
    "PI": lambda x: [extract_letter_answer(x)],
    "CCQA": extract_open_answer,
    "MOC": lambda x: [extract_letter_answer(x)],
    "MSR": lambda x: [extract_letter_answer(x)],
}


def parse_prediction(task: str, final_answer: Any) -> Any:
    """Parse model's final answer into a comparable format."""
    parser = ANSWER_PARSERS.get(task)
    if parser is None:
        return str(final_answer)
    return parser(final_answer)


def check_correct(task: str, prediction: Any, ground_truth: Any) -> bool:
    """Compare prediction with ground truth, task-specific."""
    if task == "PSS":
        pred_str = extract_sort_answer(prediction)
        truth_str = str(ground_truth).strip() if not isinstance(ground_truth, str) else ground_truth.strip()
        return pred_str is not None and pred_str == truth_str
    elif task == "FSA":
        pred_interval = extract_interval_answer(prediction)
        if pred_interval is None:
            return False
        truth = ground_truth
        if isinstance(truth, list) and len(truth) >= 2:
            return abs(pred_interval[0] - float(truth[0])) <= 2.0 and abs(pred_interval[1] - float(truth[1])) <= 2.0
        return False
    elif task == "BU":
        pred = set(extract_multi_letter_answer(prediction))
        truth = set(ground_truth) if isinstance(ground_truth, list) else {ground_truth}
        return pred == truth
    elif task == "CCQA":
        # Open-ended: we can't auto-check easily; mark as needs_scoring
        # For distillation, we trust the diagnosis agent to evaluate quality
        return None  # Special: needs manual/scoring API evaluation
    elif task in ("PEA", "CC", "NC", "PI", "MOC", "MSR"):
        pred = extract_letter_answer(prediction)
        truth = str(ground_truth).strip().upper() if not isinstance(ground_truth, list) else str(ground_truth[0]).strip().upper()
        return pred == truth
    else:
        return str(prediction).strip() == str(ground_truth).strip()


def get_ground_truth_text(task: str, question_data: Dict[str, Any]) -> str:
    """Get human-readable ground truth string for the diagnosis agent."""
    answer = question_data.get("answer", "")
    if task == "PSS":
        return str(answer)
    elif task == "FSA":
        if isinstance(answer, list) and len(answer) >= 2:
            return f"[{answer[0]}, {answer[1]}]"
        return str(answer)
    elif task == "BU":
        if isinstance(answer, list):
            return ", ".join(answer)
        return str(answer)
    elif task == "CCQA":
        scoring = question_data.get("scoring_points", [])
        if scoring:
            return f"Answer: {answer}\nScoring points: {json.dumps(scoring, ensure_ascii=False)}"
        return str(answer)
    elif task in ("PEA", "CC", "NC", "PI", "MOC", "MSR"):
        options = question_data.get("options", [])
        ans_letter = str(answer).strip().upper()
        for opt in options:
            if str(opt).strip().upper().startswith(f"{ans_letter}."):
                return f"{ans_letter}: {opt}"
        return ans_letter
    return str(answer)


# ── Video context helpers ──────────────────────────────────────

def get_num_options(task: str, question_data: Dict[str, Any]) -> int:
    """Get number of options for multi-choice tasks."""
    options = question_data.get("options", [])
    return len(options) if options else 4
