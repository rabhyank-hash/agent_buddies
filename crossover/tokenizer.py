"""Token counting and truncation (search snippets, relay caps)."""

from __future__ import annotations

from functools import lru_cache


class WhitespaceTokenizer:
    """Offline stand-in for tests and dry runs: one token per whitespace word."""

    name = "whitespace"

    def count(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, n: int) -> str:
        words = text.split()
        return text if len(words) <= n else " ".join(words[:n])


class TiktokenTokenizer:
    def __init__(self, encoding_name: str):
        import tiktoken

        try:
            self._enc = tiktoken.get_encoding(encoding_name)
        except (KeyError, ValueError):
            # Older tiktoken has no o200k_harmony; o200k_base shares its vocabulary.
            self._enc = tiktoken.get_encoding("o200k_base")
        self.name = self._enc.name

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))

    def truncate(self, text: str, n: int) -> str:
        ids = self._enc.encode(text, disallowed_special=())
        return text if len(ids) <= n else self._enc.decode(ids[:n])


@lru_cache(maxsize=4)
def get_tokenizer(name: str):
    if name == "whitespace":
        return WhitespaceTokenizer()
    return TiktokenTokenizer(name)
