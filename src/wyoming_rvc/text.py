"""Sentence splitting for complete texts (used by the upstream Wyoming source)."""

from __future__ import annotations

import re

# Terminal punctuation, optional closing quotes/brackets, whitespace, then the
# start of a new sentence (Spanish inverted marks, quotes, uppercase or digit).
_BOUNDARY = re.compile(r"[.!?…]+[\"'»”)\]]*\s+(?=[¿¡\"'«“(\[]?[A-ZÁÉÍÓÚÑÜ0-9])")
_ABBREVIATIONS = frozenset(
    "sr sra srta dr dra ud uds lic ing prof etc vs av nº núm pág ej aprox mr mrs ms st jr".split()
)
# Also a word ("I said no."): an abbreviation only before a number ("No. 5").
_NUMBER_ABBREVIATIONS = frozenset({"no"})


def split_sentences(text: str) -> list[str]:
    """Split complete text into sentences, keeping abbreviations and initials intact."""
    parts: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        words = text[start : match.start()].split()
        last = words[-1].lower().strip("\"'«“(¿¡") if words else ""
        before_number = text[match.end() : match.end() + 1].isdigit()
        if (
            last in _ABBREVIATIONS
            or (last in _NUMBER_ABBREVIATIONS and before_number)
            or (len(last) == 1 and last.isalpha())
        ):
            continue
        sentence = text[start : match.end()].strip()
        if sentence:
            parts.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts
