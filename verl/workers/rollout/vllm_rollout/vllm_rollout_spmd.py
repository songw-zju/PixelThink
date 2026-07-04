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
from contextlib import contextmanager
import os
from typing import Any, List, Union

import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer
from vllm import LLM, RequestOutput, SamplingParams

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.config import RolloutConfig


ROLLOUT_DEBUG = os.environ.get("PIXELTHINK_ROLLOUT_DEBUG", "").lower() in {
    "1",
    "true",
    "yes",
}


def _repeat_interleave(features: Union[torch.Tensor, List[Any]], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(features, torch.Tensor):
        return features.repeat_interleave(repeats, dim=0)
    else:
        return [feature for feature in features for _ in range(repeats)]


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if not config.enforce_eager and config.free_cache_engine:
            raise ValueError("CUDA graph should be disabled when `free_cache_engine` is True.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        vllm_init_kwargs = {}
        if config.limit_images > 0:
            vllm_init_kwargs = {"limit_mm_per_prompt": {"image": config.limit_images}}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=config.dtype,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
            max_model_len=config.prompt_length + config.response_length,
            max_num_batched_tokens=config.max_num_batched_tokens,
            enable_sleep_mode=True,
            distributed_executor_backend="external_launcher",
            disable_custom_all_reduce=True,
            disable_log_stats=config.disable_log_stats,
            enable_chunked_prefill=config.enable_chunked_prefill,
            **vllm_init_kwargs,
        )

        self.inference_engine.sleep(level=1)

        sampling_kwargs = {"max_tokens": config.response_length, "detokenize": False}
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)
        sampling_kwargs["logprobs"] = 5

        if ROLLOUT_DEBUG:
            print(f"[pixelthink_rollout] sampling_params={sampling_kwargs}")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)
        difficulty: torch.Tensor = prompts.batch["difficulty"]

        if ROLLOUT_DEBUG:
            print(f"[pixelthink_rollout] difficulty={difficulty}")
            print(f"[pixelthink_rollout] difficulty_shape={prompts.batch['difficulty'].shape}")
            print(f"[pixelthink_rollout] batch_size={batch_size}")

        do_sample = prompts.meta_info.get("do_sample", True)
        if not do_sample:
            kwargs = {
                "n": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
            }

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "images" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, images in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("images")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": {"image": images}})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        with self.update_sampling_params(**kwargs):
            completions: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params
            )
        response_ids = []
        uncertainty_scores = []
        for completion in completions:
            for output in completion.outputs:
                response_ids.append(output.token_ids)
                uncertainty = compute_uncertainty_from_logprob_dict(output.logprobs)
                if ROLLOUT_DEBUG:
                    print(f"[pixelthink_rollout] uncertainty={uncertainty}")
                uncertainty_scores.append(uncertainty)

        response_ids = pad_2d_list_to_length(
            response_ids, self.pad_token_id, max_length=self.config.response_length
        ).to(input_ids.device)

        if self.config.n > 1 and do_sample:
            batch_size = batch_size * self.config.n
            input_ids = _repeat_interleave(input_ids, self.config.n)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            difficulty = _repeat_interleave(difficulty, self.config.n)
            if "pixel_values" in non_tensor_batch.keys():
                non_tensor_batch["pixel_values"] = _repeat_interleave(non_tensor_batch["pixel_values"], self.config.n)
                non_tensor_batch["image_grid_thw"] = _repeat_interleave(
                    non_tensor_batch["image_grid_thw"], self.config.n
                )

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(
            response_ids=response_ids, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        uncertainty_tensor = torch.tensor(uncertainty_scores, dtype=torch.float32, device=input_ids.device)
        if ROLLOUT_DEBUG:
            print(f"[pixelthink_rollout] expanded_batch_size={batch_size}")
            print(f"[pixelthink_rollout] uncertainty_shape={uncertainty_tensor.shape}")
            print(f"[pixelthink_rollout] uncertainty_preview={uncertainty_tensor[:5]}")
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "uncertainty": uncertainty_tensor,
                "difficulty": difficulty,
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def compute_uncertainty_from_logprob_dict(logprob_dict: list) -> float:
    if not logprob_dict or not isinstance(logprob_dict, list):
        return 0.0

    uncertainty_list = []
    for token_topk in logprob_dict:
        if not token_topk or not isinstance(token_topk, dict):
            continue
        sorted_items = sorted(token_topk.items(), key=lambda x: x[1].rank)
        if len(sorted_items) < 2:
            continue
        logprob1 = sorted_items[0][1].logprob
        logprob2 = sorted_items[1][1].logprob
        p1 = torch.exp(torch.tensor(logprob1))
        p2 = torch.exp(torch.tensor(logprob2))
        uncertainty = 1.0 - (p1 - p2)
        uncertainty = float(torch.clamp(uncertainty, 0.0, 1.0).item())
        uncertainty_list.append(uncertainty)

    if not uncertainty_list:
        return 0.0
    return float(sum(uncertainty_list) / len(uncertainty_list))
