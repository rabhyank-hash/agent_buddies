"""Evidence attribution for every failure, defined identically for all systems.

  (a) no gold document ever appeared in search results
  (b) a gold document appeared, but the agent never used it
  (c) used, but its key fact is absent from the final answering context
  (d) key fact present in the final answering context, answer still wrong

"Used" is operationalized as: some agent's generated text (content, reasoning
or tool arguments) cites the gold docid after it was shown. Settle this
definition with the hand-labelled pilot trajectories before the full runs.

"Key fact present" is judged by an LLM check against the gold document. A doc
ID alone does not count. A cheap pre-check skips the LLM when the gold
document's search snippet appears verbatim in the final context (L0 without
compaction is usually in this case).
"""

from __future__ import annotations

import gzip
from typing import Callable, Mapping

KEYFACT_TEMPLATE = """You are checking whether an agent still had a crucial piece of evidence when it answered.

[question]: {question}

[correct_answer]: {answer}

[gold_document] (supports the correct answer):
{gold_doc}

[agent_final_context] (everything the agent could see when it answered):
{context}

Does [agent_final_context] contain the specific fact(s) from [gold_document] that are needed to reach [correct_answer]? A document ID alone, without the fact itself, does not count.

Reply with exactly one line: present: yes  OR  present: no"""


def load_context(record: dict) -> str:
    path = record.get("final_context_path")
    if not path:
        return ""
    with gzip.open(path, "rt") as f:
        return f.read()


def snippet_in_context(doc_text: str, context: str, probe_chars: int = 300) -> bool:
    probe = " ".join(doc_text.split())[:probe_chars]
    return bool(probe) and probe in " ".join(context.split())


def attribute(
    record: dict,
    question: str,
    answer: str,
    corpus: Mapping[str, str],
    keyfact_check: Callable[[str], str] | None,
    max_context_chars: int = 400_000,
    max_doc_chars: int = 12_000,
) -> dict:
    """Return {"category": a|b|c|d|None, ...}. None means the answer was correct."""
    if record.get("correct"):
        return {"category": None}
    if not record["gold_seen"]:
        return {"category": "a"}
    used = record["gold_cited"]
    if not used:
        return {"category": "b", "gold_seen": record["gold_seen"]}

    context = load_context(record)
    present, method = [], {}
    for docid in used:
        doc = corpus.get(docid, "")
        if snippet_in_context(doc, context):
            present.append(docid)
            method[docid] = "verbatim"
            continue
        if keyfact_check is None:
            method[docid] = "unchecked"
            continue
        prompt = KEYFACT_TEMPLATE.format(question=question, answer=answer, gold_doc=doc[:max_doc_chars],
                                         context=context[-max_context_chars:])
        verdict = keyfact_check(prompt).lower()
        method[docid] = "llm"
        if "present: yes" in verdict:
            present.append(docid)
    if keyfact_check is None and not present:
        return {"category": "c?", "used": used, "method": method}
    return {"category": "d" if present else "c", "used": used, "present": present, "method": method}
