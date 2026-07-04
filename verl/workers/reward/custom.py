# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import (
    PixelThinkRewardParams,
    seg_pixelthink_compute_score,
)
from verl.utils.reward_score.length import REWARD_WORD_TOKENS, count_reasoning_tokens, count_response_word_tokens


class CustomRewardManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: str,
        length_counting_mode: str = REWARD_WORD_TOKENS,
        pixelthink_reward_params: PixelThinkRewardParams | None = None,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.length_counting_mode = length_counting_mode
        self.pixelthink_reward_params = pixelthink_reward_params or PixelThinkRewardParams()
        self.debug = os.environ.get("PIXELTHINK_REWARD_DEBUG", "").lower() in {
            "1",
            "true",
            "yes",
        }
        if compute_score != "seg_pixelthink":
            raise NotImplementedError(f"Unsupported reward score: {compute_score}")
        self.compute_score = seg_pixelthink_compute_score

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print = 0
        if self.debug:
            print(f"[pixelthink_reward] batch_size={len(data)}")
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            uncertainty_score = data_item.batch["uncertainty"].item()
            difficulty_score = data_item.batch["difficulty"].item()
            ground_truth = data_item.non_tensor_batch["solution"]

            reward_word_token_count = count_response_word_tokens(response_str)
            reasoning_token_count = count_reasoning_tokens(
                response_str, self.tokenizer, add_special_tokens=False
            )

            score = self.compute_score(
                response_str,
                ground_truth,
                uncertainty_score=uncertainty_score,
                difficulty_score=difficulty_score,
                reward_word_token_count=reward_word_token_count,
                reasoning_token_count=reasoning_token_count,
                length_counting_mode=self.length_counting_mode,
                reward_params=self.pixelthink_reward_params,
            )
            reward_tensor[i, valid_response_length - 1] = score

            if self.debug and already_print < self.num_examine:
                already_print += 1
                print(
                    "[pixelthink_reward] sample "
                    f"idx={i} score={score} difficulty={difficulty_score:.4f} "
                    f"uncertainty={uncertainty_score:.4f} "
                    f"response_word_tokens={reward_word_token_count} "
                    f"reasoning_tokens={reasoning_token_count} "
                    f"prompt_chars={len(prompt_str)} response_chars={len(response_str)}"
                )

        return reward_tensor
