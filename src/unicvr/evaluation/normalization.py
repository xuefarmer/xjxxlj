from __future__ import annotations

import re

_LETTER = re.compile(r"\b([A-Z])\b", re.IGNORECASE)


def normalize_answer(value: str | list[str], options: list[str]) -> str:
    raw = ",".join(value) if isinstance(value, list) else value
    stripped = " ".join(raw.strip().split())
    if options:
        valid = {chr(ord("A") + index) for index in range(len(options))}
        letters = [match.upper() for match in _LETTER.findall(stripped) if match.upper() in valid]
        if letters:
            return ",".join(dict.fromkeys(letters))
        for index, option in enumerate(options):
            option_text = re.sub(r"^[A-Z][.)]\s*", "", option.strip())
            if stripped.casefold() == option_text.casefold():
                return chr(ord("A") + index)
    return stripped.casefold()
