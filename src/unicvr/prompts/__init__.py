from pathlib import Path

_PROMPT_DIR = Path(__file__).parent


def prompt(name: str) -> str:
    path = _PROMPT_DIR / f"{name}.v1.txt"
    return path.read_text(encoding="utf-8").strip()
