import numpy as np

from crossover.analysis.stats import bootstrap_crossover, crossover_point, holm, tost_paired


def synthetic_rows(p_star=0.5, n_q=400, seed=0):
    """L0 falls with pressure, M-iso is flat; they cross at p_star."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_q):
        for lv in (0.0, 0.25, 0.5, 0.75, 1.0):
            l0 = 0.45 - 0.3 * lv
            m = 0.45 - 0.3 * p_star
            rows.append({"query_id": str(q), "system": "L0", "level": lv, "correct": rng.random() < l0})
            rows.append({"query_id": str(q), "system": "M-iso", "level": lv, "correct": rng.random() < m})
    return rows


def test_crossover_recovered():
    est = crossover_point(synthetic_rows(0.5))["p_star"]
    assert 0.3 < est < 0.7
    boot = bootstrap_crossover(synthetic_rows(0.5), n_boot=100)
    assert boot["ci"][0] < 0.5 < boot["ci"][1]


def test_tost():
    rng = np.random.default_rng(1)
    a = (rng.random(830) < 0.3).astype(float)
    assert tost_paired(a, a.copy())["equivalent"]
    b = np.ones(830)
    assert not tost_paired(a, b)["equivalent"]


def test_holm():
    out = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert out["x"]["p_holm"] == 0.03 and out["x"]["reject"]
    assert out["y"]["p_holm"] == 0.06 and not out["y"]["reject"]
