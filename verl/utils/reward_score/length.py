from __future__ import annotations

import re


REWARD_WORD_TOKENS = "reward_word_tokens"
REWARD_REASONING_TOKENS = "reward_reasoning_tokens"

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_reasoning(text: str) -> str:
    if not text:
        return ""
    match = _THINK_RE.search(text)
    return match.group(1).strip() if match else ""


def count_response_word_tokens(text: str) -> int:
    return len(text.split()) if text else 0


def count_reasoning_word_tokens(text: str) -> int:
    reasoning = extract_reasoning(text)
    return len(reasoning.split()) if reasoning else 0


def _tokenizer_input_ids(tokenizer, text: str, *, add_special_tokens=None):
    kwargs = {}
    if add_special_tokens is not None:
        kwargs["add_special_tokens"] = add_special_tokens
    try:
        tokenized = tokenizer(text, **kwargs)
    except TypeError:
        tokenized = tokenizer(text)
    return tokenized["input_ids"]


def count_reasoning_tokens(text: str, tokenizer, *, add_special_tokens=None) -> int:
    reasoning = extract_reasoning(text)
    if not reasoning:
        return 0
    return len(_tokenizer_input_ids(tokenizer, reasoning, add_special_tokens=add_special_tokens))


def count_reward_length(
    text: str,
    tokenizer=None,
    mode: str = REWARD_WORD_TOKENS,
) -> int:
    if mode == REWARD_WORD_TOKENS:
        return count_response_word_tokens(text)
    if mode == REWARD_REASONING_TOKENS:
        if tokenizer is None:
            raise ValueError(f"{REWARD_REASONING_TOKENS} requires a tokenizer")
        return count_reasoning_tokens(text, tokenizer, add_special_tokens=False)
    raise ValueError(f"Unknown reward length counting mode: {mode}")
