"""Accuracy judge (gpt-4.1), BrowseComp grading template.

For numbers comparable to the paper, swap GRADER_TEMPLATE for the exact
template and parsing in the BrowseComp-Plus repo's evaluation script before
the full runs (gap-check design issue 8). This copy follows the public
BrowseComp grader and is enough for the pilot.
"""

from __future__ import annotations

import os
import re

GRADER_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""

_CORRECT = re.compile(r"correct\s*:\s*\**\s*(yes|no)", re.IGNORECASE)
_EXTRACTED = re.compile(r"extracted_final_answer\s*:\s*(.+)", re.IGNORECASE)


def parse_grade(text: str) -> dict:
    m = _CORRECT.search(text)
    e = _EXTRACTED.search(text)
    return {
        "correct": bool(m and m.group(1).lower() == "yes"),
        "parsed": m is not None,
        "extracted_final_answer": e.group(1).strip() if e else None,
    }


class Judge:
    def __init__(self, judge_cfg, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=judge_cfg.base_url, api_key=os.environ.get(judge_cfg.api_key_env))
        self.client = client
        self.model = judge_cfg.model

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def grade(self, question: str, response: str, correct_answer: str) -> dict:
        if not response.strip():
            return {"correct": False, "parsed": True, "extracted_final_answer": None, "judge_raw": ""}
        raw = self.complete(GRADER_TEMPLATE.format(question=question, response=response, correct_answer=correct_answer))
        return {**parse_grade(raw), "judge_raw": raw}
