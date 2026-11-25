"""Shared episode runtime: threads, the tool loop, budget enforcement, logging.

Every arm is built from the same pieces:
  * Thread: one context window (a message list) owned by one agent identity.
  * Episode: per-trajectory state shared by all threads: LLM, the pressure
    search tool, the token ledger, citation tracking and relay logging.
  * run_loop: call the model, run tools, stop on the thread's finish tool.
    It also enforces the budget cap: at 0.9B the thread is told to finish
    and is forced to that tool.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any, Callable, Iterable

from crossover import prompts
from crossover.ledger import COORDINATION_TOOLS, TokenLedger
from crossover.llm import LLM, ContextOverflow
from crossover.pressure import PressureSearch
from crossover.tools import name_of
from crossover.types import LLMResponse, Query, Stop, ToolCall


class Thread:
    def __init__(self, agent_id: str, thread_id: str, system: str, user: str, depth: int = 0):
        self.agent_id = agent_id
        self.thread_id = thread_id
        self.depth = depth
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        self.prompt_trace: list[int] = []
        self.searches_since_branch = 0
        self.wrap_notice = False
        self.last_step: int | None = None

    def add_assistant(self, resp: LLMResponse, carry_reasoning: bool, reasoning_field: str) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": resp.content or (None if resp.tool_calls else "")}
        if resp.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.raw_arguments or json.dumps(tc.arguments)},
                }
                for tc in resp.tool_calls
            ]
        if carry_reasoning and resp.reasoning:
            msg[reasoning_field] = resp.reasoning
        self.messages.append(msg)

    def add_tool_result(self, tc: ToolCall, content: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def last_assistant_text(self) -> str:
        for m in reversed(self.messages):
            if m["role"] == "assistant" and m.get("content"):
                return m["content"]
        return ""

    def render_text(self, include_system: bool = False) -> str:
        """Plain-text transcript: what a M-shared worker sees, and the final
        answering context used for evidence attribution."""
        out = []
        for m in self.messages:
            role = m["role"]
            if role == "system" and not include_system:
                continue
            for key in ("reasoning_content", "reasoning"):
                if m.get(key):
                    out.append(f"[{role} reasoning]\n{m[key]}")
            if m.get("content"):
                out.append(f"[{role}]\n{m['content']}")
            for tc in m.get("tool_calls") or []:
                out.append(f"[{role} -> {tc['function']['name']}] {tc['function']['arguments']}")
        return "\n\n".join(out)


class Plan:
    """The global plan file of the decomposition protocol."""

    def __init__(self, question: str):
        self.text = f"Objective: {question}\nSteps: (none yet)"
        self.docs: list[str] = []
        self._lock = threading.Lock()

    def update(self, text: str) -> None:
        with self._lock:
            self.text = text.strip() or self.text

    def add_docs(self, docids: Iterable[str]) -> None:
        with self._lock:
            for d in docids:
                if d not in self.docs:
                    self.docs.append(d)

    def render(self) -> str:
        with self._lock:
            docs = ", ".join(self.docs) if self.docs else "(none yet)"
            return f"{self.text}\nDoc IDs found so far: {docs}"


class Episode:
    def __init__(self, cfg, llm: LLM, search: PressureSearch, ledger: TokenLedger, tokenizer, query: Query, sample: int = 0):
        self.cfg, self.llm, self.search, self.ledger = cfg, llm, search, ledger
        self.tokenizer, self.query, self.sample = tokenizer, query, sample
        self.lock = threading.RLock()
        self.threads: list[Thread] = []
        self.seen_docids: set[str] = set()
        self.cited_docids: set[str] = set()
        self.relays: list[dict[str, Any]] = []
        self.branches = 0
        self.rejected_branches = 0
        self.compactions = 0
        self.cap_hit = False
        self.overflow = False
        self.peak_context = 0
        self.final: dict[str, Any] | None = None
        self.final_context: str = ""
        self.final_context_docids: list[str] = []
        self.answer_step: int | None = None
        self.stop_reason: str | None = None
        self.extra: dict[str, Any] = {}

    # -- threads ----------------------------------------------------------------

    def new_thread(self, agent_id: str, thread_id: str, system: str, user: str, depth: int = 0) -> Thread:
        th = Thread(agent_id, thread_id, system, user, depth)
        with self.lock:
            self.threads.append(th)
        return th

    # -- model calls --------------------------------------------------------------

    def call(self, th: Thread, specs: list[dict], force: str | None = None, kind: str | None = None) -> LLMResponse:
        max_tokens = self.ledger.max_tokens_for_next_call(self.cfg.model.max_tokens_per_call)
        tool_choice = {"type": "function", "function": {"name": force}} if force else None
        resp = self.llm.chat(th.messages, specs or None, max_tokens, tool_choice)
        if kind is None:
            kind = "coordination" if {tc.name for tc in resp.tool_calls} & COORDINATION_TOOLS else "work"
        step = self.ledger.record(th.agent_id, th.thread_id, resp.prompt_tokens, resp.completion_tokens, kind, self.sample)
        th.last_step = step.idx
        th.prompt_trace.append(resp.prompt_tokens)
        with self.lock:
            self.peak_context = max(self.peak_context, resp.prompt_tokens)
        self._scan_citations(resp)
        return resp

    def _scan_citations(self, resp: LLMResponse) -> None:
        text = " ".join(filter(None, [resp.content, resp.reasoning, *(tc.raw_arguments or json.dumps(tc.arguments) for tc in resp.tool_calls)]))
        with self.lock:
            self.cited_docids |= docids_in(text, self.seen_docids)

    # -- tools --------------------------------------------------------------------

    def do_search(self, th: Thread, query: str) -> str:
        shown, text = self.search.search(query, th.agent_id, th.thread_id)
        th.searches_since_branch += 1
        with self.lock:
            self.seen_docids.update(shown)
        return text

    def relay(self, text: str, kind: str, thread: Thread) -> str:
        """Apply the relay-bandwidth cap to a summary/report crossing contexts."""
        cap = self.cfg.agents.relay_cap_tokens
        raw_tokens = self.tokenizer.count(text)
        kept = self.tokenizer.truncate(text, cap) if cap else text
        with self.lock:
            self.relays.append({
                "kind": kind,
                "thread_id": thread.thread_id,
                "raw_tokens": raw_tokens,
                "kept_tokens": self.tokenizer.count(kept),
                "truncated": kept != text,
                "text": kept,
            })
        return kept

    def thread_docids(self, th: Thread) -> set[str]:
        return docids_in(th.render_text(), self.seen_docids)

    def docids_cited_in(self, text: str) -> set[str]:
        return docids_in(text, self.seen_docids)

    # -- ending -------------------------------------------------------------------

    def finish(self, th: Thread, stop: Stop) -> None:
        self.stop_reason = stop.kind
        answered = stop.kind == "finish"
        if answered and "_raw" in stop.args:
            raw = stop.args["_raw"]
            final = {"explanation": "", "exact_answer": extract_exact_answer(raw), "confidence": None, "raw": raw}
        elif answered:
            final = {
                "explanation": str(stop.args.get("explanation", "")),
                "exact_answer": str(stop.args.get("exact_answer", "")),
                "confidence": stop.args.get("confidence"),
                "raw": None,
            }
        else:
            raw = th.last_assistant_text()
            final = {"explanation": "", "exact_answer": extract_exact_answer(raw) if raw else "", "confidence": None, "raw": raw}
        final["answered"] = answered
        self.final = final
        self.answer_step = th.last_step if answered else None
        self.final_context = th.render_text()
        self.final_context_docids = sorted(docids_in(self.final_context, self.seen_docids))


def response_text(final: dict[str, Any]) -> str:
    """The response string the judge sees (BrowseComp answer format)."""
    if final.get("raw") and not final.get("exact_answer"):
        return final["raw"]
    conf = final.get("confidence")
    return (
        f"Explanation: {final.get('explanation', '')}\n"
        f"Exact Answer: {final.get('exact_answer', '')}\n"
        f"Confidence: {conf if conf is not None else 100}%"
    )


_EXACT = re.compile(r"exact answer\s*:\s*(.+)", re.IGNORECASE)


def extract_exact_answer(text: str) -> str:
    m = _EXACT.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()


def docids_in(text: str, candidates: Iterable[str]) -> set[str]:
    found = set()
    for d in candidates:
        if d and re.search(rf"(?<![\w-]){re.escape(d)}(?![\w-])", text):
            found.add(d)
    return found


HandleBatch = Callable[[list[ToolCall]], list[str]]


def sequential(handle: Callable[[ToolCall], str]) -> HandleBatch:
    return lambda calls: [handle(c) for c in calls]


def run_loop(
    ep: Episode,
    th: Thread,
    tools: list[dict] | Callable[[], list[dict]],
    handle_batch: HandleBatch,
    finish_tool: str,
    max_turns: int,
    branch_tool: str | None = None,
    before_call: Callable[[Thread], None] | None = None,
) -> Stop:
    cfg = ep.cfg
    force_after = cfg.agents.force_branch_after_searches
    for _ in range(max_turns):
        if ep.ledger.exhausted():
            return Stop("budget")
        specs = tools() if callable(tools) else tools
        available = {name_of(s) for s in specs}
        force = None
        if ep.ledger.wrap_up():
            ep.cap_hit = True
            if not th.wrap_notice:
                th.add_user(prompts.WRAP_UP[finish_tool])
                th.wrap_notice = True
            force = finish_tool
        elif branch_tool in available and force_after and th.searches_since_branch >= force_after:
            th.add_user(prompts.FORCE_BRANCH[branch_tool])
            th.searches_since_branch = 0
            force = branch_tool
        if force:
            specs = [s for s in specs if name_of(s) == force]
        if before_call:
            before_call(th)
        try:
            resp = ep.call(th, specs, force)
        except ContextOverflow:
            ep.overflow = True
            return Stop("overflow")
        th.add_assistant(resp, cfg.model.carry_reasoning, cfg.model.reasoning_field)

        if not resp.tool_calls:
            text = (resp.content or "").strip()
            if text:
                return Stop("finish", {"_raw": text})
            th.add_user(prompts.NUDGE.format(finish=finish_tool))
            continue

        allowed = {name_of(s) for s in specs}
        pending: list[ToolCall] = []
        for tc in resp.tool_calls:
            if tc.name == finish_tool and finish_tool in allowed:
                _run(th, pending, handle_batch, allowed)
                return Stop("finish", tc.arguments)
            pending.append(tc)
        _run(th, pending, handle_batch, allowed)
    return Stop("max_turns")


def _run(th: Thread, calls: list[ToolCall], handle_batch: HandleBatch, allowed: set[str]) -> None:
    valid = [c for c in calls if c.name in allowed]
    results = dict(zip((c.id for c in valid), handle_batch(valid))) if valid else {}
    for c in calls:
        th.add_tool_result(c, results.get(c.id, f"Error: tool {c.name!r} is not available here."))
