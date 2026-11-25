"""Pre-registered statistics.

  H2  p*: logistic regression of correctness on pressure, system and their
      interaction; p* is where predicted L0 and M-iso accuracy are equal.
      95% CI from a bootstrap over queries.
  H3  S-iso vs. M-iso at P100: paired TOST with a +/-5-point margin.
  H4+ contrasts: paired bootstrap intervals, Holm correction for the
      exploratory family.

The proposal specifies a random effect per query in the p* model. This file
fits fixed effects only; the query bootstrap handles within-query
correlation for the interval. Swap in a mixed model (e.g. statsmodels
BinomialBayesMixedGLM) for the final analysis if you want the point
estimate from the random-effects fit.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-6, iters: int = 100) -> np.ndarray:
    """Newton-Raphson / IRLS with a tiny ridge for separation."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        w = p * (1 - p)
        grad = X.T @ (y - p) - l2 * beta
        hess = (X * w[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def _design(rows: Sequence[dict], sys_a: str, sys_b: str):
    X, y = [], []
    for r in rows:
        if r["system"] not in (sys_a, sys_b):
            continue
        b = 1.0 if r["system"] == sys_b else 0.0
        p = float(r["level"])
        X.append([1.0, p, b, p * b])
        y.append(1.0 if r["correct"] else 0.0)
    return np.array(X), np.array(y)


def crossover_point(rows: Sequence[dict], sys_a: str = "L0", sys_b: str = "M-iso") -> dict:
    """p* where logit_a(p) == logit_b(p): b0 + b1 p = (b0 + b2) + (b1 + b3) p  =>  p* = -b2 / b3."""
    X, y = _design(rows, sys_a, sys_b)
    if len(y) == 0:
        return {"p_star": None, "beta": None}
    beta = fit_logistic(X, y)
    p_star = -beta[2] / beta[3] if abs(beta[3]) > 1e-9 else math.inf
    return {"p_star": float(p_star), "beta": beta.tolist(), "n": int(len(y))}


def bootstrap_crossover(rows: Sequence[dict], sys_a: str = "L0", sys_b: str = "M-iso",
                        n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["system"] in (sys_a, sys_b):
            by_q[r["query_id"]].append(r)
    qids = sorted(by_q)
    point = crossover_point(rows, sys_a, sys_b)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sample = [r for q in rng.choice(qids, size=len(qids), replace=True) for r in by_q[q]]
        draws.append(crossover_point(sample, sys_a, sys_b)["p_star"])
    arr = np.array([d for d in draws if d is not None])
    finite = arr[np.isfinite(arr)]
    lo, hi = (np.quantile(finite, [alpha / 2, 1 - alpha / 2]).tolist() if len(finite) else (None, None))
    inside = float(np.mean((arr >= 0) & (arr <= 1))) if len(arr) else 0.0
    return {**point, "ci": [lo, hi], "share_of_draws_inside_sweep": inside, "n_boot": n_boot}


def paired(rows: Sequence[dict], sys_a: str, sys_b: str, level: float) -> tuple[np.ndarray, np.ndarray]:
    a = {r["query_id"]: float(r["correct"]) for r in rows if r["system"] == sys_a and r["level"] == level}
    b = {r["query_id"]: float(r["correct"]) for r in rows if r["system"] == sys_b and r["level"] == level}
    common = sorted(set(a) & set(b))
    return np.array([a[q] for q in common]), np.array([b[q] for q in common])


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def tost_paired(a: np.ndarray, b: np.ndarray, margin: float = 0.05, alpha: float = 0.05) -> dict:
    """Two one-sided tests for |mean(a - b)| < margin, normal approximation."""
    d = a - b
    n = len(d)
    if n < 2:
        return {"n": n, "equivalent": False}
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n)) or 1e-12
    p_lower = 1 - _norm_cdf((mean + margin) / se)  # H0: diff <= -margin
    p_upper = _norm_cdf((mean - margin) / se)  # H0: diff >= +margin
    z = _z(1 - alpha)
    return {
        "n": n,
        "diff": mean,
        "se": se,
        "ci90": [mean - z * se, mean + z * se],
        "p": max(p_lower, p_upper),
        "equivalent": max(p_lower, p_upper) < alpha,
        "margin": margin,
    }


def _z(q: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _norm_cdf(mid) < q else (lo, mid)
    return (lo + hi) / 2


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05) -> dict:
    d = a - b
    if len(d) == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    p = 2 * min(np.mean(means <= 0), np.mean(means >= 0))
    return {"n": int(len(d)), "diff": float(d.mean()), "ci": [float(lo), float(hi)], "p": float(min(1.0, p))}


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    order = sorted(pvals, key=pvals.get)
    m = len(order)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        adj = min(1.0, max(running, (m - i) * pvals[k]))
        running = adj
        out[k] = {"p": pvals[k], "p_holm": adj, "reject": adj < alpha}
    return out


CONTRASTS = {
    # name: (minuend, subtrahend)
    "isolation_single": ("S-iso", "S-shared"),
    "isolation_multi": ("M-iso", "M-shared"),
    "agent_separation": ("M-iso-seq", "S-iso"),
    "parallelism": ("M-iso", "M-iso-seq"),
    "siso_vs_miso": ("S-iso", "M-iso"),
    "miso_vs_l0": ("M-iso", "L0"),
}
