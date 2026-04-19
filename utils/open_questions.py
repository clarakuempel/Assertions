"""Open-ended (wh-) questions for PopQA-style facts and answer classification."""

from __future__ import annotations

import re
from typing import Any

WHO_RELATIONS = frozenset(
    {"author", "composer", "director", "screenwriter", "producer", "father", "mother"}
)


def build_open_ended_question(fact: dict[str, Any]) -> str:
    """
    Build a single open-ended question whose truthful answer is ``object_pri``.

    Chooses who / what / where from ``relation`` and PopQA ``subject`` / ``subject_relation`` fields.
    """
    rel = (fact.get("relation") or "").strip().lower()
    subject = (fact.get("subject") or "").strip()
    sr = (fact.get("subject_relation") or "").strip()

    if rel == "place of birth" and subject:
        return f"Where was {subject} born?"
    if rel in WHO_RELATIONS and sr:
        return f"Who is {sr}?"
    if sr:
        return f"What is {sr}?"
    if subject:
        return f"What is the correct answer regarding {subject}?"
    return "What is the answer?"


def _dedupe_longest_first(strings: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in sorted((x.strip() for x in strings if x and str(x).strip()), key=len, reverse=True):
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def _alias_lists(fact: dict[str, Any]) -> tuple[list[str], list[str]]:
    pri = [str(fact.get("object_pri") or "").strip()]
    pri.extend(str(x).strip() for x in (fact.get("possible_prior_answers") or []) if x)
    ctx = [str(fact.get("object_ctx") or "").strip()]
    ctx.extend(str(x).strip() for x in (fact.get("object_ctx_aliases") or []) if x)
    return _dedupe_longest_first(pri), _dedupe_longest_first(ctx)


def _answer_mentions_phrase(answer_norm: str, phrase: str) -> bool:
    phrase = phrase.strip().lower()
    if not phrase:
        return False
    if len(phrase) <= 3:
        return re.search(rf"\b{re.escape(phrase)}\b", answer_norm) is not None
    return phrase in answer_norm


def classify_open_ended(answer: str, query_type: str, fact: dict[str, Any]) -> str:
    """
    Label free-form ``answer`` as memory (aligned with ``object_pri``), context (``object_ctx``), or other.

    ``query_type`` is ``prior_yes`` or ``ctx_yes`` (same as yes/no pipeline); used only to break ties
    when both true and false aliases appear in the answer.
    """
    if answer is None or answer in ("ERROR", -1) or (isinstance(answer, int) and answer == -1):
        return "error"
    a = str(answer).lower()
    pri_aliases, ctx_aliases = _alias_lists(fact)
    pri_hit = any(_answer_mentions_phrase(a, p) for p in pri_aliases)
    ctx_hit = any(_answer_mentions_phrase(a, p) for p in ctx_aliases)
    if pri_hit and ctx_hit:
        return "context" if query_type == "ctx_yes" else "memory"
    if pri_hit:
        return "memory"
    if ctx_hit:
        return "context"
    return "other"
