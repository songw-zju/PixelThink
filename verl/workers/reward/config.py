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
from dataclasses import dataclass


@dataclass
class RewardConfig:
    reward_type: str = "function"
    compute_score: str = "seg_pixelthink"
    length_counting_mode: str = "reward_word_tokens"
    pixelthink_hard_difficulty_threshold: float = 5.0
    pixelthink_medium_difficulty_threshold: float = 3.5
    pixelthink_hard_token_budget: float = 256.0
    pixelthink_easy_token_budget: float = 96.0
    pixelthink_uncertainty_budget_scale: float = 25.0
    pixelthink_length_penalty_beta: float = 1 / 500.0
    pixelthink_iou_threshold: float = 0.5
    pixelthink_box_l1_threshold: float = 10.0
    pixelthink_point_l1_threshold: float = 100.0

    def pixelthink_reward_params(self):
        from verl.utils.reward_score import PixelThinkRewardParams

        return PixelThinkRewardParams(
            hard_difficulty_threshold=self.pixelthink_hard_difficulty_threshold,
            medium_difficulty_threshold=self.pixelthink_medium_difficulty_threshold,
            hard_token_budget=self.pixelthink_hard_token_budget,
            easy_token_budget=self.pixelthink_easy_token_budget,
            uncertainty_budget_scale=self.pixelthink_uncertainty_budget_scale,
            length_penalty_beta=self.pixelthink_length_penalty_beta,
            iou_threshold=self.pixelthink_iou_threshold,
            box_l1_threshold=self.pixelthink_box_l1_threshold,
            point_l1_threshold=self.pixelthink_point_l1_threshold,
        )
