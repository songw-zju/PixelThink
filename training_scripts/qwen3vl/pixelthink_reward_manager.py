from __future__ import annotations

import asyncio
import os
import sys

from omegaconf import DictConfig
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pixelthink_seg_reward import (
    REWARD_WORD_TOKENS,
    REWARD_REASONING_TOKENS,
    count_reasoning_tokens,
    count_response_word_tokens,
    pixelthink_reward,
)


class PixelThinkRewardManager(RewardManagerBase):
    def __init__(self, config: DictConfig, tokenizer, compute_score=None, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        self.length_counting_mode = os.environ.get("PIXELTHINK_REWARD_LENGTH_MODE", REWARD_WORD_TOKENS)
        if self.length_counting_mode not in (REWARD_WORD_TOKENS, REWARD_REASONING_TOKENS):
            raise ValueError(f"Unknown reward length counting mode: {self.length_counting_mode}")

    def _get_uncertainty(self, data_item) -> float:
        ntb = data_item.non_tensor_batch
        if "uncertainty" in ntb:
            try:
                return float(ntb["uncertainty"])
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @staticmethod
    def _last_item(data: DataProto):
        try:
            tail = data[-1:]
            return tail[0]
        except Exception:
            return data[-1]

    async def run_single(self, data: DataProto) -> dict:
        data_item = self._last_item(data)

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
        difficulty = float(extra_info.get("difficulty", 3.0))
        uncertainty = self._get_uncertainty(data_item)

        loop = getattr(self, "loop", None) or asyncio.get_running_loop()
        solution_str = await loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        reward_length_word_tokens = count_response_word_tokens(solution_str)
        reasoning_token_count = count_reasoning_tokens(
            solution_str,
            self.tokenizer,
            add_special_tokens=False,
        )
        reward_length_used = (
            reasoning_token_count
            if self.length_counting_mode == REWARD_REASONING_TOKENS
            else reward_length_word_tokens
        )

        score = pixelthink_reward(
            solution_str,
            ground_truth,
            uncertainty_score=uncertainty,
            difficulty_score=difficulty,
            reward_word_token_count=reward_length_word_tokens,
            reasoning_token_count=reasoning_token_count,
            length_counting_mode=self.length_counting_mode,
        )

        return {
            "reward_score": float(score),
            "reward_extra_info": {
                "difficulty": difficulty,
                "uncertainty": uncertainty,
                "reward_length_word_tokens": reward_length_word_tokens,
                "reasoning_token_count": reasoning_token_count,
                "length_counting_mode": self.length_counting_mode,
                "reward_length_used": reward_length_used,
            },
        }
