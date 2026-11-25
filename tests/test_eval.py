import json
import sys
from pathlib import Path
from types import SimpleNamespace

from crossover.config import DataConfig
from crossover.data import split_queries
from crossover.eval.attribution import categorize
from crossover.eval.judge import grader_prompt, parse_grade
from crossover.eval.sail_batch import SailBatchJudge, custom_id, response_text
from crossover.ledger import TokenLedger, estimate_cost
from crossover.types import Query

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_retriever import ndcg_at, recall_at  # noqa: E402


def test_splits_are_disjoint_nested_and_deterministic():
    qs = [Query(str(i), f"q{i}", "a") for i in range(830)]
    s = split_queries(qs, DataConfig())
    ids = {k: [q.query_id for q in v] for k, v in s.items()}
    assert (len(ids["pilot"]), len(ids["core"]), len(ids["ablation"]), len(ids["rest"])) == (50, 500, 150, 280)
    assert not set(ids["pilot"]) & set(ids["nonpilot"])
    assert ids["ablation"] == ids["core"][:150]
    assert ids["nonpilot"] == ids["core"] + ids["rest"]
    assert split_queries(list(reversed(qs)), DataConfig())["core"] == s["core"]


def test_official_grade_parser_formats():
    assert parse_grade("extracted_final_answer: Quist\nreasoning: same\ncorrect: yes\nconfidence: 80%")["correct"]
    bold = parse_grade("**extracted_final_answer:** Venn\n**correct:** no\n**confidence:** 140")
    assert not bold["correct"] and not bold["parse_error"] and bold["judge_confidence"] == 100
    assert parse_grade("no verdict here")["parse_error"]
    assert "[correct_answer]: Quist" in grader_prompt("Who?", "Exact Answer: Quist", "Quist")


def test_batch_result_text_and_ids():
    body = {"output": [{"type": "reasoning", "content": []},
                       {"type": "message", "content": [{"type": "output_text", "text": "correct: "},
                                                       {"type": "output_text", "text": "yes"}]}]}
    assert response_text(body) == "correct: yes"
    cid = custom_id("grade:S-iso|P100|rnone|769")
    assert cid[0].isalnum() and cid.islower() and len(cid) <= 64


def test_batch_judge_resumes_from_saved_results(tmp_path):
    judge = SailBatchJudge(SimpleNamespace(base_url=None, api_key_env="X", model="m", max_output_tokens=10), tmp_path)
    with open(judge.results_path, "w") as f:
        f.write(json.dumps({"custom_id": custom_id("grade:a"), "text": "correct: yes"}) + "\n")
    # Already-graded prompts are served from disk without any HTTP call.
    assert judge.complete_many({"grade:a": "prompt"}) == {"grade:a": "correct: yes"}


def test_categorize():
    base = {"correct": False, "gold_seen": ["d1"], "gold_cited": ["d1"]}
    assert categorize({**base, "correct": True}) == {"category": None}
    assert categorize({**base, "gold_seen": []})["category"] == "a"
    assert categorize({**base, "gold_cited": []})["category"] == "b"
    assert categorize(base, [], {"d1": "present: no"})["category"] == "c"
    assert categorize(base, [], {"d1": "present: yes"})["category"] == "d"
    assert categorize(base, ["d1"], {})["category"] == "d"


def test_cost_uses_cached_rate():
    led = TokenLedger(10_000)
    led.record("a", "root", 1_000_000, 100_000, "work", cached_tokens=900_000, reasoning_tokens=60_000)
    s = led.summary()
    assert s["cached"] == 900_000 and s["reasoning"] == 60_000
    cost = estimate_cost(s, {"input": 0.07, "cached": 0.005, "output": 0.22})
    assert abs(cost - (0.1 * 0.07 + 0.9 * 0.005 + 0.1 * 0.22)) < 1e-9
    assert estimate_cost(TokenLedger(1).summary(), None) is None


def test_retrieval_metrics():
    assert ndcg_at(["x", "d1"], {"d1"}) < ndcg_at(["d1", "x"], {"d1"}) == 1.0
    assert recall_at(["d1", "x", "d2"], {"d1", "d2"}, 2) == 0.5


def test_chat_request_uses_max_completion_tokens_and_service_tier():
    from crossover.config import ModelConfig
    from crossover.llm import OpenAIChatLLM

    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        msg = SimpleNamespace(content="Exact Answer: x", tool_calls=None, reasoning_content="r", model_extra={})
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20,
                                prompt_tokens_details={"cached_tokens": 80},
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=15))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)

    llm = OpenAIChatLLM.__new__(OpenAIChatLLM)
    llm.cfg = ModelConfig(service_tier="balanced")
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    resp = llm.chat([{"role": "user", "content": "hi"}], None, 512)
    assert sent["max_completion_tokens"] == 512 and "max_tokens" not in sent
    assert sent["extra_body"] == {"service_tier": "balanced"}
    assert (resp.cached_tokens, resp.reasoning_tokens, resp.reasoning) == (80, 15, "r")
