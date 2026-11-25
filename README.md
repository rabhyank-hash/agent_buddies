# When Does a Single Agent Need Teammates?

Code for the 11-768 final project (Abhyankar & Ramnath). It locates the
context-pressure crossover on BrowseComp-Plus. We raise context pressure in
controlled steps and find where an orchestrator–worker system overtakes a
single ReAct agent. We then test whether the gain comes from **context
isolation**, from **agent separation** or from **parallelism**.

Design sources: the proposal doc and the "Context-Pressure Crossover —
References, Gap Check & Experiment Kickoff" doc on claude.ai.

## Setup

```bash
uv sync                                    # core deps + pytest
uv run pytest                              # offline tests, no model needed
uv run python scripts/run.py --dry-run     # every arm end to end on a toy corpus
```

Extras: `uv sync --extra dense` gives Qwen3-Embedding retrieval, and
`uv sync --extra hf` downloads BrowseComp-Plus from the Hugging Face hub.

## Serving (any platform)

The code needs two HTTP endpoints, and they can run anywhere:

- **Model server:** any OpenAI-compatible API with tool calling, set in
  `model.base_url`. One example is vLLM serving gpt-oss-20b with
  `--enable-auto-tool-choice --tool-call-parser openai`. Turn on prefix caching
  if your server supports it, since L0 and the shared-context arms re-read long
  contexts on every call.
- **Retrieval server:** `scripts/serve_retrieval.py`, set in
  `retrieval.http_url`. You can skip it by setting `retrieval.backend: dense`,
  which loads the index in the same process instead.

Sweeps resume from `trajectories.jsonl`, so an interrupted run can be restarted
with the same command.

## Workflow (the kickoff steps, in order)

1. **Testbed.** Decrypt the BrowseComp-Plus queries with the script in their
   repo and save them to `data/browsecomp_plus_decrypted.jsonl`. Export the
   corpus to `data/corpus.jsonl` with `crossover.data.export_corpus_from_hf`.
   Serve gpt-oss-20b with vLLM, which gives an OpenAI-compatible API at
   `model.base_url`. Run a search backend, either `retrieval.backend: dense`
   or `http`. Then run L0 on 100 pilot queries and compare with the paper:
   `uv run python scripts/run.py --pilot --systems L0 --levels 0 --n 100`
2. **Pressure dial.** This is already built and unit-tested in
   [crossover/pressure.py](crossover/pressure.py). Gold docs, and their
   ranks, are identical at every level.
3. **Harness and ledger.** Also built: [crossover/harness.py](crossover/harness.py)
   and [crossover/ledger.py](crossover/ledger.py).
4. **RQ1 kill test.** Compare L0 and M-iso at P0 and P100 on 100 queries. If
   there is no gap at P100, rethink the dial before going further:
   `uv run python scripts/run.py --systems L0 M-iso --levels 0 1 --n 100`
5. **Remaining arms.** These are built too: S-iso, S-shared, M-shared,
   M-iso-seq, L0-sum and SC. Pass them with `--systems`.
6. **Grade and attribute.** `uv run python scripts/grade.py runs/sweep`. Before
   trusting the attribution judge, check it against 50 trajectories you label
   by hand.
7. **Analyze.** `uv run python scripts/analyze.py runs/sweep`. Freeze the
   analysis plan before the full runs.

Any config key can be overridden from the command line, for example
`--set budget.B=80000 --set agents.relay_cap_tokens=256 --set output.run_dir=runs/relay256`.

## Arms

| Name | Context | Agents | What it isolates |
|---|---|---|---|
| `L0` | one growing context | one | ReAct baseline |
| `L0-sum` | compacted when long | one | does simple compaction move p*? |
| `S-shared` | subtask sees full history | one | 2×2 cell |
| `S-iso` | subtask sees only its goal | one | ours: hierarchical plan agent |
| `M-shared` | worker gets the lead's transcript | many | 2×2 cell |
| `M-iso` | worker sees only its task, runs in parallel | many | standard orchestrator–worker |
| `M-iso-seq` | like M-iso, one delegation per turn | many | parallelism control |
| `SC` | 4 × L0 at B/4, majority vote | one | strongest simple baseline |

From these, the contrasts are:

- isolation = `S-iso − S-shared`
- agent separation = `M-iso-seq − S-iso`
- parallelism = `M-iso − M-iso-seq`

## Repo structure

```
final_project/
├── pyproject.toml            package + dependencies (uv)
├── configs/
│   └── default.yaml          all experiment settings: model, data, pressure levels, budget B, agent limits, judge
├── crossover/                the library
│   ├── config.py             YAML → dataclasses; `--set a.b=c` overrides
│   ├── types.py              Query, ToolCall, LLMResponse, Stop
│   ├── data.py               load decrypted BrowseComp-Plus queries + corpus; fixed-seed pilot/eval split (300 ⊂ 500 ⊂ all)
│   ├── tokenizer.py          token counting/truncation (o200k_harmony; whitespace stand-in for tests)
│   ├── retrieval.py          base retrievers: BM25 (toy/dev), dense Qwen3-Embedding index, HTTP retrieval server
│   ├── pressure.py           the pressure dial: hard-negative substitution of non-gold slots, k=5, 512-token snippets, search log
│   ├── ledger.py             per-episode token ledger over all agents; 0.9B wrap-up, per-call max_tokens, coordination share
│   ├── llm.py                OpenAI-compatible client (vLLM/gpt-oss) + ScriptedLLM for tests; context-overflow detection
│   ├── tools.py              tool schemas: search, answer, start/finish_subtask, delegate, report, read/update_plan
│   ├── prompts.py            system prompts; all 2×2 cells share one decomposition protocol
│   ├── agents/
│   │   ├── base.py           Thread (one context), Plan, Episode (shared state), run_loop (tool loop + budget forcing)
│   │   ├── react.py          L0 and L0-sum
│   │   ├── single.py         S-shared and S-iso
│   │   ├── multi.py          M-iso, M-iso-seq, M-shared
│   │   ├── self_consistency.py  SC
│   │   └── __init__.py       registry: name → arm
│   ├── harness.py            run_episode (one trajectory → one JSON record) and resumable run_sweep
│   ├── eval/
│   │   ├── judge.py          gpt-4.1 accuracy judge (BrowseComp grading template)
│   │   └── attribution.py    failure categories (a)–(d) with the key-fact check
│   ├── analysis/
│   │   ├── stats.py          logistic p* + query bootstrap (H2), paired TOST (H3), paired bootstrap, Holm
│   │   └── budget.py         re-scoring under "generated" vs. "total" token budgets
│   └── toy.py                toy corpus + scripted policy for tests and --dry-run
├── scripts/
│   ├── run.py                run a sweep (systems × levels × queries); --pilot, --n, --dry-run
│   ├── serve_retrieval.py    HTTP search server over the dense index
│   ├── grade.py              judge + attribution → graded.jsonl
│   └── analyze.py            pilot diagnostics, accuracy table, p*, TOST, contrasts, failure modes → analysis.json
└── tests/
    ├── test_pressure.py      gold recall identical across levels, nested replacements, no duplicates, repeat rule
    ├── test_ledger.py        wrap-up/reservation, both budget definitions, budget re-scoring
    ├── test_agents.py        every arm end to end; isolation really isolates; M-iso-seq; relay cap; forced answer/branch
    └── test_stats.py         p* recovered on synthetic data, TOST, Holm
```

Each sweep writes to `runs/<name>/`:

- `trajectories.jsonl`: one record per (system, level, query). It holds tokens
  per agent and per call, context length per call, every search (base vs.
  shown docs), doc IDs cited, gold docs seen, cited and in the final context,
  relay summaries, branch counts, and cap-hit and overflow flags.
- `contexts/`: the final answering context of each trajectory, gzipped.
- `errors.jsonl`: any failed episodes. They are retried when you re-run.
- `graded.jsonl` and `analysis.json`: written by the grade and analyze scripts.

## Decisions built into the code (check these in the pilot)

- **S-shared keeps the full sub-trajectory visible.** This is gap-check design
  issue 1. Otherwise it would be an isolation arm by accident.
- **One subtask at a time** for S-shared, S-iso and M-iso-seq. M-iso allows up
  to `agents.max_parallel_workers` delegations per turn. This interpretation
  is what makes the parallelism contrast meaningful.
- **All parent threads can search** (`agents.parent_can_search`), and every
  cell gets the same plan tools. The harness re-injects the plan after every
  subtask or report, and tracks the doc IDs that summaries cite.
- **Pressure coins** are fixed per (query, call index, slot), so P25's
  replacements are a subset of P50's. Gold docs are protected, and so are
  evidence docs with `pressure.protect: gold+evidence`.
- **"Used" in attribution** means the docid is cited in generated text after
  it was shown. Settle this definition with the hand-labelled trajectories.
- **Budget cap:** generated tokens count against the cap. Prefill is logged
  so you can re-score under total tokens. At 0.9B every thread is forced to
  its finish tool.

## Still to do or verify

- Swap in the **official BrowseComp-Plus grader** template in `eval/judge.py`
  so your accuracy is comparable with the paper.
- **Data field names.** Check `data.FIELD_ALIASES` against your decrypted file.
- **Dense query prefix** (`retrieval.dense_query_prefix`). It must match how
  the index was built.
- **Carried reasoning.** Confirm that the server reads the reasoning field
  back (`model.reasoning_field`): prompt tokens should grow when
  `carry_reasoning` is on. If L0's p95 peak context is above 110K, set it to
  false for all systems.
- **p\* model.** `analysis/stats.py` fits fixed effects with a query
  bootstrap. Swap in a mixed model if you want the per-query random effect
  from the proposal.
- **Pilot queries and the 830-query P100 run.** Pilot queries are excluded
  from eval subsets, but the plan's 830-query run uses all queries. Decide
  which wins: `--set data.pilot_size=0 --set data.subset_size=null` gives the
  full set.
