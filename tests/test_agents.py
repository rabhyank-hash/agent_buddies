import pytest

from crossover.agents import SYSTEMS
from crossover.config import load_config
from crossover.harness import run_episode
from crossover.llm import ScriptedLLM
from crossover.retrieval import BM25Retriever
from crossover.tokenizer import get_tokenizer
from crossover.toy import TOY_CORPUS, TOY_QUERIES, _call, _resp, toy_policy

TOK = get_tokenizer("whitespace")


def cfg(**overrides):
    sets = ["tokenizer=whitespace", "budget.B=5000", "output.save_final_context=false"]
    return load_config(None, sets + [f"{k}={v}" for k, v in overrides.items()])


def episode(system, level=0.0, c=None, policy=toy_policy, query=TOY_QUERIES[0]):
    llm = ScriptedLLM(policy)
    rec = run_episode(c or cfg(), llm, BM25Retriever(TOY_CORPUS), TOY_CORPUS, query, system, level, TOK)
    return rec, llm


@pytest.mark.parametrize("system", sorted(SYSTEMS))
@pytest.mark.parametrize("level", [0.0, 1.0])
def test_every_arm_runs_end_to_end(system, level):
    rec, _ = episode(system, level)
    assert rec["answered"], rec["stop_reason"]
    assert rec["final"]["exact_answer"].lower() == "quist"
    assert rec["gold_seen"] == ["d1"]
    assert rec["ledger"]["generated"] > 0
    if system in ("S-iso", "S-shared", "M-iso", "M-iso-seq", "M-shared"):
        assert rec["branches"] >= 1 and rec["relays"]


def _first_request(llm, thread_marker):
    return next(r for r in llm.requests if thread_marker in r["messages"][1]["content"])


def test_s_iso_subtask_sees_only_its_brief():
    _, llm = episode("S-iso")
    sub = _first_request(llm, "Subtask goal:")
    assert len(sub["messages"]) == 2  # system + brief, no parent history


def test_s_shared_keeps_full_history_in_one_thread():
    rec, llm = episode("S-shared")
    assert len(rec["threads"]) == 1
    sizes = [len(r["messages"]) for r in llm.requests]
    assert sizes == sorted(sizes)  # one ever-growing context


def test_m_shared_worker_gets_orchestrator_transcript_but_m_iso_does_not():
    _, llm = episode("M-shared")
    worker = [r for r in llm.requests if r["messages"][0]["content"].startswith("You are a research worker")][0]
    assert "Lead agent's transcript so far" in worker["messages"][1]["content"]
    _, llm = episode("M-iso")
    worker = [r for r in llm.requests if r["messages"][0]["content"].startswith("You are a research worker")][0]
    assert "transcript" not in worker["messages"][1]["content"]


def two_delegations(messages, tools, tool_choice):
    names = [t["function"]["name"] for t in tools or []]
    called = [tc["function"]["name"] for m in messages if m["role"] == "assistant" for tc in m.get("tool_calls") or []]
    if "delegate" in names and "delegate" not in called and len(names) > 1:
        return _resp([_call("delegate", task="Who designed the Velloran bridge?"),
                      _call("delegate", task="Mara Quist engineer")], messages)
    return toy_policy(messages, tools, tool_choice)


def test_m_iso_runs_parallel_but_m_iso_seq_rejects_second_delegation():
    par, _ = episode("M-iso", policy=two_delegations)
    seq, _ = episode("M-iso-seq", policy=two_delegations)
    assert par["branches"] == 2 and par["rejected_branches"] == 0
    assert seq["branches"] == 1 and seq["rejected_branches"] == 1


def test_relay_cap_truncates_reports():
    rec, _ = episode("M-iso", c=cfg(**{"agents.relay_cap_tokens": 3}))
    assert all(r["kept_tokens"] <= 3 for r in rec["relays"])
    assert any(r["truncated"] for r in rec["relays"])


def endless_search(messages, tools, tool_choice):
    names = [t["function"]["name"] for t in tools or []]
    if names == ["answer"]:
        return toy_policy(messages, tools, tool_choice)
    resp = _resp([_call("search", query="Velloran bridge")], messages)
    resp.completion_tokens = 400
    return resp


def test_budget_wrap_up_forces_answer():
    rec, llm = episode("L0", c=cfg(**{"budget.B": 2000}), policy=endless_search)
    assert rec["cap_hit"] and rec["answered"]
    assert llm.requests[-1]["tool_choice"] == {"type": "function", "function": {"name": "answer"}}
    assert rec["ledger"]["generated"] <= 2000 + 400


def test_forced_branch_after_n_searches():
    def never_branch(messages, tools, tool_choice):
        """The root keeps searching unless forced; once it has branched, it answers."""
        names = [t["function"]["name"] for t in tools or []]
        if "start_subtask" in names and len(names) > 1:
            called = [tc["function"]["name"] for m in messages if m["role"] == "assistant" for tc in m.get("tool_calls") or []]
            if "start_subtask" not in called:
                return _resp([_call("search", query="Velloran bridge")], messages)
            return toy_policy(messages, [t for t in tools if t["function"]["name"] == "answer"], None)
        return toy_policy(messages, tools, tool_choice)

    rec, llm = episode("S-iso", c=cfg(**{"agents.force_branch_after_searches": 2}), policy=never_branch)
    assert any(r["tool_choice"] == {"type": "function", "function": {"name": "start_subtask"}} for r in llm.requests)
    assert rec["branches"] >= 1
