"""L0 ReAct and its post-hoc summarization variant (Context-Folding's summary-agent baseline)."""

from __future__ import annotations

from crossover import prompts, tools
from crossover.agents.base import Episode, Thread, run_loop, sequential
from crossover.llm import ContextOverflow


class ReactAgent:
    def __init__(self, summarize: bool = False):
        self.summarize = summarize
        self.name = "L0-sum" if summarize else "L0"

    def run(self, ep: Episode) -> None:
        q = ep.query
        th = ep.new_thread("agent", "root", prompts.REACT_SYSTEM, q.question)

        def handle(tc):
            if tc.name == "search":
                return ep.do_search(th, str(tc.arguments.get("query", "")))
            return f"Error: unknown tool {tc.name!r}"

        before = self._compact_if_long(ep) if self.summarize else None
        stop = run_loop(ep, th, [tools.SEARCH, tools.ANSWER], sequential(handle), "answer",
                        ep.cfg.agents.max_turns, before_call=before)
        ep.finish(th, stop)

    def _compact_if_long(self, ep: Episode):
        trigger = ep.cfg.agents.summary_trigger_tokens
        state = {"calls_at_last_compaction": -1}

        def compact(th: Thread) -> None:
            # prompt_trace[-1] is the size of the last request; only act on a
            # request made since the previous compaction.
            if len(th.prompt_trace) - 1 <= state["calls_at_last_compaction"]:
                return
            if not th.prompt_trace or th.prompt_trace[-1] < trigger or ep.ledger.wrap_up():
                return
            th.add_user(prompts.COMPACTION_REQUEST)
            try:
                resp = ep.call(th, [], kind="compaction")
            except ContextOverflow:
                th.messages.pop()
                return
            summary = (resp.content or resp.reasoning or "").strip()
            th.messages = [
                th.messages[0],
                {"role": "user", "content": prompts.COMPACTED_USER.format(question=ep.query.question, summary=summary)},
            ]
            state["calls_at_last_compaction"] = len(th.prompt_trace)
            ep.compactions += 1

        return compact
