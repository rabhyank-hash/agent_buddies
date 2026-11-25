"""A tiny offline world for tests and `--dry-run`: a toy corpus with gold docs,
evidence and hard negatives, plus a scripted policy that drives every arm's
tools the way a model would. Not a model of agent behavior: it exists to
exercise the harness end to end without a GPU."""

from __future__ import annotations

import itertools
import json
import re
from typing import Any

from crossover.types import LLMResponse, Query, ToolCall

TOY_CORPUS = {
    "d1": "The Velloran bridge was designed by engineer Mara Quist in 1931. ANSWER: quist",
    "d2": "Mara Quist studied at the Ostrava polytechnic before designing bridges.",
    "d3": "Bridges in Velloran include several stone arches from the 1800s.",
    "d4": "A bridge engineer named Mara Quintana worked on dams, not the Velloran bridge.",
    "d5": "The Velloran bridge festival is held every spring near the river.",
    "d6": "Engineer Mara Quist's 1931 bridge rumors: a different designer is often credited.",
    "d7": "Velloran tourism guide: bridges, museums and the old town.",
    "d8": "The comet Tersh was first observed by astronomer Ilia Venn in 1888. ANSWER: venn",
    "d9": "Ilia Venn kept detailed notebooks of comet observations.",
    "d10": "Comet sightings in 1888 were reported by several observatories.",
    "d11": "Astronomer Ilia Vennard catalogued variable stars, not comets.",
    "d12": "The comet Tersh returns every 76 years according to some sources.",
    "d13": "Observatory history: telescopes and astronomers of the 19th century.",
    "d14": "A 1888 comet was misattributed to astronomer Ilia Venn in a newspaper.",
}

TOY_QUERIES = [
    Query("q1", "Who designed the Velloran bridge?", "Quist", ["d1"], ["d1", "d2"], ["d3", "d4", "d5", "d6", "d7"]),
    Query("q2", "Who first observed the comet Tersh?", "Venn", ["d8"], ["d8", "d9"], ["d10", "d11", "d12", "d13", "d14"]),
]

_ids = itertools.count()
_DOCID = re.compile(r"docid: (\S+)")
_ANSWER = re.compile(r"ANSWER: (\w+)")


def _call(name: str, **args) -> ToolCall:
    raw = json.dumps(args)
    return ToolCall(id=f"call_{next(_ids)}", name=name, arguments=args, raw_arguments=raw)


def _resp(calls: list[ToolCall], messages, content: str | None = None) -> LLMResponse:
    prompt = sum(len(str(m.get("content") or "").split()) for m in messages)
    completion = 20 + sum(len(c.raw_arguments.split()) for c in calls)
    return LLMResponse(content=content, reasoning="thinking about the docs", tool_calls=calls,
                       prompt_tokens=prompt, completion_tokens=completion)


def toy_policy(messages: list[dict[str, Any]], tools: list[dict] | None, tool_choice: Any) -> LLMResponse:
    names = [t["function"]["name"] for t in tools or []]
    called = [tc["function"]["name"] for m in messages if m["role"] == "assistant" for tc in m.get("tool_calls") or []]
    text = "\n".join(str(m.get("content") or "") for m in messages)
    brief = str(messages[1]["content"]).split("\n")[0]
    docids = _DOCID.findall(text)
    found = _ANSWER.findall(text)

    if not tools:  # compaction request
        return _resp([], messages, content=f"Progress: saw {', '.join(dict.fromkeys(docids))}. {' '.join(found)}")
    if len(names) == 1:
        forced = names[0]
    elif "delegate" in names and "delegate" not in called:
        forced = "delegate"
    elif "start_subtask" in names and "start_subtask" not in called:
        forced = "start_subtask"
    elif "search" in names and "search" not in called:
        forced = "search"
    elif "finish_subtask" in names:
        forced = "finish_subtask"
    elif "report" in names:
        forced = "report"
    else:
        forced = "answer"

    summary = f"Key facts: {'; '.join('ANSWER: ' + a for a in found) or 'none'}. Supporting docids: {', '.join(dict.fromkeys(docids))}"
    if forced == "delegate":
        return _resp([_call("delegate", task=brief)], messages)
    if forced == "start_subtask":
        return _resp([_call("start_subtask", goal=brief)], messages)
    if forced == "search":
        return _resp([_call("search", query=brief.replace("Subtask goal: ", ""))], messages)
    if forced in ("finish_subtask", "report"):
        return _resp([_call(forced, summary=summary)], messages)
    if forced == "answer":
        guess = found[0] if found else "unknown"
        return _resp([_call("answer", explanation=summary, exact_answer=guess, confidence=80)], messages)
    return _resp([_call(forced)], messages)
