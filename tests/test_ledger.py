from crossover.analysis.budget import correct_at_budget, spend_at_answer
from crossover.ledger import TokenLedger


def test_wrap_up_and_reservation():
    led = TokenLedger(1000, wrap_up_frac=0.9)
    assert led.max_tokens_for_next_call(10_000) == 900  # can't eat the reserved 10%
    led.record("a", "root", prompt_tokens=500, completion_tokens=850, kind="work")
    assert not led.wrap_up()
    led.record("a", "root", 500, 60, "work")
    assert led.wrap_up() and not led.exhausted()
    assert led.max_tokens_for_next_call(10_000) == 256  # min floor; remaining is 90
    led.record("w", "worker-1", 100, 100, "coordination")
    assert led.exhausted()
    s = led.summary()
    assert s["generated"] == 1010 and s["prefill"] == 1100 and s["calls"] == 3
    assert set(s["per_agent"]) == {"a", "w"}


def test_total_definition_counts_prefill():
    led = TokenLedger(1000, definition="total")
    led.record("a", "root", 600, 350, "work")
    assert led.wrap_up()


def test_budget_rescoring():
    rec = {"correct": True, "answer_step": 1, "steps": [
        {"completion_tokens": 100, "prompt_tokens": 1000},
        {"completion_tokens": 50, "prompt_tokens": 2000},
        {"completion_tokens": 999, "prompt_tokens": 999},  # after the answer: ignored
    ]}
    assert spend_at_answer(rec, "generated") == 150
    assert spend_at_answer(rec, "total") == 3150
    assert correct_at_budget(rec, 200, "generated")
    assert not correct_at_budget(rec, 200, "total")
    assert not correct_at_budget({**rec, "answer_step": None}, 1e9, "generated")
