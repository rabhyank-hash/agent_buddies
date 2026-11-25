from crossover.pressure import PressureSearch
from crossover.retrieval import BM25Retriever
from crossover.tokenizer import get_tokenizer
from crossover.toy import TOY_CORPUS, TOY_QUERIES

TOK = get_tokenizer("whitespace")
QUERIES = ["Velloran bridge designer", "Mara Quist engineer", "bridge 1931", "Velloran tourism"]


def run(level, query=TOY_QUERIES[0], searches=QUERIES):
    ps = PressureSearch(BM25Retriever(TOY_CORPUS), TOY_CORPUS, query, level, TOK, k=5, seed=1)
    for s in searches:
        ps.search(s)
    return ps


def test_gold_recall_identical_across_levels():
    """Kickoff step 2: for a fixed query sequence, gold docs shown (and their ranks) never change."""
    gold = set(TOY_QUERIES[0].gold_docids)
    ref = [[(i, d) for i, d in enumerate(r.shown) if d in gold] for r in run(0.0).log]
    assert any(ref), "test needs at least one search that retrieves gold"
    for level in (0.25, 0.5, 0.75, 1.0):
        got = [[(i, d) for i, d in enumerate(r.shown) if d in gold] for r in run(level).log]
        assert got == ref


def test_p0_is_untouched_and_p100_replaces_every_nongold_slot():
    for r in run(0.0).log:
        assert r.shown == r.base and not any(r.replaced)
    ps = run(1.0)
    gold = set(TOY_QUERIES[0].gold_docids)
    for r in ps.log:
        for base, shown, rep in zip(r.base, r.shown, r.replaced):
            if base in gold:
                assert shown == base and not rep
            elif rep:
                assert shown in TOY_QUERIES[0].negative_docids
    assert ps.stats()["replaced"] + ps.unfilled == sum(d not in gold for r in ps.log for d in r.base)


def test_no_duplicates_and_nested_replacements():
    low, high = run(0.25), run(0.75)
    for a, b in zip(low.log, high.log):
        assert len(set(b.shown)) == len(b.shown)
        # same coins: any slot replaced at P25 is also replaced at P75
        assert all(rb for ra, rb in zip(a.replaced, b.replaced) if ra)


def test_negatives_repeat_only_after_pool_runs_out():
    ps = run(1.0, searches=QUERIES * 5)
    assert ps.draws > len(TOY_QUERIES[0].negative_docids)
    assert ps.repeats > 0
    assert ps.repeats == ps.draws - len(ps._drawn)  # every first-time draw is unique


def test_length_neutral_and_truncated():
    ps = PressureSearch(BM25Retriever(TOY_CORPUS), TOY_CORPUS, TOY_QUERIES[0], 1.0, TOK, k=5, snippet_tokens=3)
    shown, text = ps.search("Velloran bridge")
    assert len(shown) == len(ps.log[0].base)
    for block in text.split("\n\n")[1:]:
        assert len(block.split("\n", 1)[1].split()) <= 3
