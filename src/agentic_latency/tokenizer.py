from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class TokenCounter:
    """Load a Hugging Face tokenizer only when an experiment is executed."""

    def __init__(self, model: str) -> None:
        from transformers import AutoTokenizer

        self._tokenizer: Any = AutoTokenizer.from_pretrained(model)

    def count_prompt(self, messages: list[Mapping[str, str]]) -> int:
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(self._tokenizer.encode(text))

    def count_output(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

