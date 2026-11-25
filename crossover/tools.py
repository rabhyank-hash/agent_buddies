"""OpenAI-format tool specs shared by all arms."""

from __future__ import annotations


def _fn(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": required or []},
        },
    }


def _relay_note(cap: int | None) -> str:
    return f" Keep it under {cap} tokens; anything longer is cut off." if cap else ""


SEARCH = _fn(
    "search",
    "Search the document corpus. Returns up to 5 documents, each truncated, labelled with its docid.",
    {"query": {"type": "string", "description": "Search query."}},
    ["query"],
)

ANSWER = _fn(
    "answer",
    "Give the final answer to the user's question. Ends the episode.",
    {
        "explanation": {"type": "string", "description": "Short justification citing docids."},
        "exact_answer": {"type": "string", "description": "The succinct final answer."},
        "confidence": {"type": "number", "description": "Confidence from 0 to 100."},
    },
    ["exact_answer"],
)

UPDATE_PLAN = _fn(
    "update_plan",
    "Replace the global plan (objective, steps, status). Doc IDs found are tracked for you.",
    {"plan": {"type": "string"}},
    ["plan"],
)

READ_PLAN = _fn("read_plan", "Return the current global plan, including doc IDs found so far.")

START_SUBTASK = _fn(
    "start_subtask",
    "Start a subtask with a short, self-contained goal. One subtask at a time.",
    {"goal": {"type": "string", "description": "What the subtask must find."}},
    ["goal"],
)


def finish_subtask(cap: int | None) -> dict:
    return _fn(
        "finish_subtask",
        "Finish the current subtask. The summary must list the key facts found and the docid supporting each."
        + _relay_note(cap),
        {"summary": {"type": "string"}},
        ["summary"],
    )


DELEGATE = _fn(
    "delegate",
    "Hand a short, self-contained task to a worker agent, which searches and reports back.",
    {"task": {"type": "string", "description": "What the worker must find."}},
    ["task"],
)


def report(cap: int | None) -> dict:
    return _fn(
        "report",
        "Report back to the orchestrator and stop. List the key facts found and the docid supporting each."
        + _relay_note(cap),
        {"summary": {"type": "string"}},
        ["summary"],
    )


def name_of(spec: dict) -> str:
    return spec["function"]["name"]
