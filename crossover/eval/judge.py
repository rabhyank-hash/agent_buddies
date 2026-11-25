"""Accuracy judge. Template and parsing are copied from the official
BrowseComp-Plus evaluator (scripts_evaluation/evaluate_run.py), so grades are
comparable with the benchmark's own runs. The judge model is set in config
(gpt-4.1 in the paper; the official leaderboard uses Qwen3-32B)."""

from __future__ import annotations

import os
import re

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def grader_prompt(question: str, response: str, correct_answer: str) -> str:
    return GRADER_TEMPLATE.format(question=question, response=response, correct_answer=correct_answer)


def _field(text: str, name: str, pattern: str) -> str | None:
    # Same precedence as the official parser: **name:**, then **name**:, then name:
    for prefix in (rf"\*\*{name}:\*\*\s*", rf"\*\*{name}\*\*:\s*", rf"{name}:\s*"):
        m = re.search(prefix + pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def parse_grade(text: str) -> dict:
    text = text or ""
    correct = _field(text, "correct", r"(yes|no)")
    conf = _field(text, "confidence", r"(\d+(?:\.\d+)?)\s*%?")
    return {
        "correct": correct is not None and correct.lower() == "yes",
        "parse_error": correct is None,
        "extracted_final_answer": _field(text, "extracted_final_answer", r"(.*?)(?=\n|$)"),
        "judge_confidence": min(100.0, float(conf)) if conf else None,
    }


class ChatJudge:
    """Synchronous chat-completions judge (vLLM, OpenAI, Sail ASAP/Balanced models)."""

    def __init__(self, judge_cfg, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=judge_cfg.base_url, api_key=os.environ.get(judge_cfg.api_key_env))
        self.client, self.cfg = client, judge_cfg

    def complete(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=self.cfg.max_output_tokens,
        )
        return resp.choices[0].message.content or ""

    def complete_many(self, prompts: dict[str, str], workers: int = 16) -> dict[str, str]:
        from concurrent.futures import ThreadPoolExecutor

        keys = list(prompts)
        with ThreadPoolExecutor(workers) as pool:
            return dict(zip(keys, pool.map(lambda k: self.complete(prompts[k]), keys)))


def make_judge(judge_cfg, state_dir=None):
    if judge_cfg.mode == "chat":
        return ChatJudge(judge_cfg)
    if judge_cfg.mode == "sail_batch":
        from crossover.eval.sail_batch import SailBatchJudge

        return SailBatchJudge(judge_cfg, state_dir)
    raise ValueError(f"Unknown judge mode {judge_cfg.mode!r}")
