"""The one-agent row of the 2x2: S-shared and S-iso (the hierarchical plan agent).

Both use the same decomposition protocol and tools. The only difference:

  S-shared  start_subtask/finish_subtask are markers. The subtask runs in the
            same thread, so its full sub-trajectory stays visible to the parent
            and to later subtasks. (Gap-check design issue 1: if S-shared
            dropped the sub-trajectory it would be an isolation arm by
            accident, and the isolation effect would be ~0 by construction.)

  S-iso     start_subtask(goal) swaps the global plan and the parent's history
            out for a fresh context holding only the goal; read_plan() stays
            available. finish_subtask(summary) returns only the (relay-capped)
            summary to the parent, and the harness re-injects the plan.

The harness enforces every context change, so the result does not depend on
the model following a convention. Both cells allow one subtask per turn.
"""

from __future__ import annotations

from crossover import prompts, tools
from crossover.agents.base import Episode, Plan, Thread, run_loop
from crossover.types import ToolCall

REJECT_EXTRA_BRANCH = "Rejected: only one subtask can be started per turn. Re-issue it after this one finishes."


class SingleAgent:
    def __init__(self, isolated: bool):
        self.isolated = isolated
        self.name = "S-iso" if isolated else "S-shared"

    def run(self, ep: Episode) -> None:
        cfg = ep.cfg.agents
        q = ep.query
        plan = Plan(q.question)
        system = prompts.SINGLE_ISO_SYSTEM if self.isolated else prompts.SINGLE_SHARED_SYSTEM
        root = ep.new_thread("agent", "root", system, f"{q.question}\n\nCurrent plan:\n{plan.render()}")
        state = {"in_subtask": False, "n": 0}
        finish_spec = tools.finish_subtask(cfg.relay_cap_tokens if self.isolated else None)

        def root_tools() -> list[dict]:
            specs = [tools.SEARCH] if cfg.parent_can_search or state["in_subtask"] else []
            if state["in_subtask"]:  # S-shared only
                specs += [finish_spec, tools.READ_PLAN, tools.UPDATE_PLAN, tools.ANSWER]
            else:
                specs += [tools.START_SUBTASK, tools.UPDATE_PLAN, tools.READ_PLAN, tools.ANSWER]
            return specs

        def handle_root(calls: list[ToolCall]) -> list[str]:
            out, branched = [], False
            for tc in calls:
                if tc.name == "start_subtask":
                    if branched:
                        ep.rejected_branches += 1
                        out.append(REJECT_EXTRA_BRANCH)
                        continue
                    branched = True
                    out.append(self._start(ep, root, plan, state, str(tc.arguments.get("goal", "")), finish_spec))
                elif tc.name == "finish_subtask":
                    out.append(self._finish_shared(ep, root, plan, state, str(tc.arguments.get("summary", ""))))
                else:
                    out.append(common_tool(ep, root, plan, tc))
            return out

        stop = run_loop(ep, root, root_tools, handle_root, "answer", cfg.max_turns, branch_tool="start_subtask")
        ep.finish(root, stop)

    # -- S-shared ---------------------------------------------------------------

    def _start(self, ep, root, plan, state, goal, finish_spec) -> str:
        if state["in_subtask"]:
            return "Rejected: finish the current subtask before starting another."
        ep.branches += 1
        root.searches_since_branch = 0
        state["n"] += 1
        if self.isolated:
            return self._run_isolated(ep, plan, state["n"], goal, finish_spec)
        state["in_subtask"] = True
        return f"Subtask {state['n']} started: {goal}\nSearch for it, then call `finish_subtask(summary)`."

    def _finish_shared(self, ep, root, plan, state, summary) -> str:
        if self.isolated or not state["in_subtask"]:
            return "Error: no subtask is open."
        state["in_subtask"] = False
        ep.relay(summary, "subtask_summary", root)  # logged for summary-length stats; never capped here
        plan.add_docs(ep.docids_cited_in(summary))
        return f"Subtask closed.\n\nCurrent plan:\n{plan.render()}"

    # -- S-iso ------------------------------------------------------------------

    def _run_isolated(self, ep: Episode, plan: Plan, n: int, goal: str, finish_spec: dict) -> str:
        child = ep.new_thread("agent", f"sub-{n}", prompts.SINGLE_ISO_SYSTEM, prompts.SUBTASK_BRIEF.format(goal=goal), depth=1)

        def handle(calls: list[ToolCall]) -> list[str]:
            return [common_tool(ep, child, plan, tc) for tc in calls]

        stop = run_loop(ep, child, [tools.SEARCH, tools.READ_PLAN, finish_spec], handle, "finish_subtask",
                        ep.cfg.agents.max_subtask_turns)
        summary = subtask_summary(stop, child)
        kept = ep.relay(summary, "subtask_summary", child)
        plan.add_docs(ep.docids_cited_in(kept) & ep.thread_docids(child))
        return f"Subtask finished. Summary:\n{kept}\n\nCurrent plan:\n{plan.render()}"


def subtask_summary(stop, th: Thread) -> str:
    if stop.kind == "finish":
        return str(stop.args.get("summary") or stop.args.get("_raw") or "").strip() or "(empty summary)"
    last = th.last_assistant_text()
    return f"(subtask ended without a summary: {stop.kind}) {last}".strip()


def common_tool(ep: Episode, th: Thread, plan: Plan, tc: ToolCall) -> str:
    if tc.name == "search":
        return ep.do_search(th, str(tc.arguments.get("query", "")))
    if tc.name == "read_plan":
        return plan.render()
    if tc.name == "update_plan":
        plan.update(str(tc.arguments.get("plan", "")))
        return "Plan updated.\n\n" + plan.render()
    return f"Error: unknown tool {tc.name!r}"
