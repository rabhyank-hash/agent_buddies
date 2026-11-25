"""The separate-agents row of the 2x2, plus the parallelism control.

  M-iso      Standard orchestrator-worker. Each worker is a fresh agent
             (cold start, its own role prompt, no plan access) that sees only
             its task. Several delegations in one turn run concurrently.
  M-iso-seq  Same orchestrator and workers, but one delegation per turn, so
             the orchestrator reads each report before issuing the next task,
             exactly like S-iso's one-subtask-at-a-time. This splits
             S-iso vs. M-iso into agent separation (M-iso-seq - S-iso) and
             parallelism (M-iso - M-iso-seq).
  M-shared   Like M-iso, but each worker's brief is the orchestrator's full
             transcript at delegation time, plus its task.

Reports cross contexts, so they go through the relay cap.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from crossover import prompts, tools
from crossover.agents.base import Episode, Plan, Thread, run_loop
from crossover.agents.single import common_tool, subtask_summary
from crossover.types import ToolCall

REJECT_SEQ = "Rejected: only one delegation per turn in this configuration. Re-issue it after reading this report."
REJECT_PAR = "Rejected: at most {n} delegations per turn. Re-issue it next turn."


class MultiAgent:
    def __init__(self, shared: bool, parallel: bool):
        self.shared, self.parallel = shared, parallel
        self.name = "M-shared" if shared else ("M-iso" if parallel else "M-iso-seq")

    def run(self, ep: Episode) -> None:
        cfg = ep.cfg.agents
        q = ep.query
        plan = Plan(q.question)
        orch = ep.new_thread("orchestrator", "root", prompts.ORCHESTRATOR_SYSTEM,
                             f"{q.question}\n\nCurrent plan:\n{plan.render()}")
        specs = ([tools.SEARCH] if cfg.parent_can_search else []) + [
            tools.DELEGATE, tools.UPDATE_PLAN, tools.READ_PLAN, tools.ANSWER]
        max_per_turn = cfg.max_parallel_workers if self.parallel else 1
        counter = {"n": 0}

        def handle(calls: list[ToolCall]) -> list[str]:
            results: dict[str, str] = {}
            delegations = [c for c in calls if c.name == "delegate"]
            accepted, rejected = delegations[:max_per_turn], delegations[max_per_turn:]
            for c in rejected:
                ep.rejected_branches += 1
                results[c.id] = REJECT_SEQ if not self.parallel else REJECT_PAR.format(n=max_per_turn)
            if accepted:
                orch.searches_since_branch = 0
                transcript = orch.render_text() if self.shared else None  # snapshot at delegation time
                jobs = []
                for c in accepted:
                    counter["n"] += 1
                    ep.branches += 1
                    jobs.append((c, counter["n"]))
                if len(jobs) == 1:
                    outs = [self._worker(ep, plan, jobs[0][1], str(jobs[0][0].arguments.get("task", "")), transcript)]
                else:
                    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                        outs = list(pool.map(
                            lambda j: self._worker(ep, plan, j[1], str(j[0].arguments.get("task", "")), transcript), jobs))
                for (c, _), out in zip(jobs, outs):
                    results[c.id] = out
                # Re-inject the plan once, after the last report of this turn.
                last = accepted[-1].id
                results[last] += f"\n\nCurrent plan:\n{plan.render()}"
            for c in calls:
                if c.name != "delegate":
                    results[c.id] = common_tool(ep, orch, plan, c)
            return [results[c.id] for c in calls]

        stop = run_loop(ep, orch, specs, handle, "answer", cfg.max_turns, branch_tool="delegate")
        ep.finish(orch, stop)

    def _worker(self, ep: Episode, plan: Plan, k: int, task: str, transcript: str | None) -> str:
        brief = prompts.WORKER_SHARED_BRIEF.format(transcript=transcript, task=task) if self.shared else task
        w: Thread = ep.new_thread(f"worker-{k}", f"worker-{k}", prompts.WORKER_SYSTEM, brief, depth=1)

        def handle(calls: list[ToolCall]) -> list[str]:
            return [ep.do_search(w, str(c.arguments.get("query", ""))) if c.name == "search"
                    else f"Error: unknown tool {c.name!r}" for c in calls]

        stop = run_loop(ep, w, [tools.SEARCH, tools.report(ep.cfg.agents.relay_cap_tokens)], handle, "report",
                        ep.cfg.agents.max_subtask_turns)
        kept = ep.relay(subtask_summary(stop, w), "worker_report", w)
        plan.add_docs(ep.docids_cited_in(kept) & ep.thread_docids(w))
        return f"Report from worker-{k}:\n{kept}"
