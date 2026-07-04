from __future__ import annotations

import json
import math
import re


REWARD_WORD_TOKENS = "reward_word_tokens"
REWARD_REASONING_TOKENS = "reward_reasoning_tokens"

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_reasoning(text: str) -> str:
    if not text:
        return ""
    m = _THINK_RE.search(text)
    return m.group(1).strip() if m else ""


def count_response_word_tokens(text: str) -> int:
    return len(text.split()) if text else 0


def count_reasoning_word_tokens(text: str) -> int:
    reasoning = extract_reasoning(text)
    return len(reasoning.split()) if reasoning else 0


def _tokenizer_input_ids(tokenizer, text: str, *, add_special_tokens=None):
    if add_special_tokens is not None:
        return tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
    return tokenizer(text)["input_ids"]


def count_reasoning_tokens(text: str, tokenizer, *, add_special_tokens=None) -> int:
    reasoning = extract_reasoning(text)
    if not reasoning:
        return 0
    return len(_tokenizer_input_ids(tokenizer, reasoning, add_special_tokens=add_special_tokens))


def count_reward_length(text: str, tokenizer=None, mode: str = REWARD_WORD_TOKENS) -> int:
    if mode == REWARD_WORD_TOKENS:
        return count_response_word_tokens(text)
    if mode == REWARD_REASONING_TOKENS:
        if tokenizer is None:
            raise ValueError(f"{REWARD_REASONING_TOKENS} requires a tokenizer")
        return count_reasoning_tokens(text, tokenizer, add_special_tokens=False)
    raise ValueError(f"Unknown reward length counting mode: {mode}")


def seg_thinking_format_reward(predict_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return 1.0 if re.fullmatch(pattern, predict_str, re.DOTALL) else 0.0


def seg_segmentation_format_reward(predict_str: str) -> float:
    try:
        m = re.search(r"{[^}]+}", predict_str)
        if not m:
            return 0.0
        data = json.loads(m.group(0))
        for key in ("bbox", "points_1", "points_2"):
            if key not in data:
                return 0.0
        if not isinstance(data["bbox"], list) or len(data["bbox"]) != 4:
            return 0.0
        for key in ("points_1", "points_2"):
            if not isinstance(data[key], list) or len(data[key]) != 2:
                return 0.0
        return 1.0
    except Exception:
        return 0.0


def _gt_box(ground_truth: str):
    m = re.search(r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>", ground_truth.strip())
    return [int(m.group(i)) for i in range(1, 5)] if m else None


def _gt_points(ground_truth: str):
    m = re.search(r"<points>\((\d+),(\d+)\),\((\d+),(\d+)\)</points>", ground_truth)
    return [[int(m.group(1)), int(m.group(2))], [int(m.group(3)), int(m.group(4))]] if m else None


def _pred_json(predict_str: str):
    m = re.search(r"{[^}]+}", predict_str)
    return json.loads(m.group(0)) if m else None


def seg_iou_reward(predict_str: str, ground_truth: str) -> float:
    def iou(b1, b2):
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = (ix2 - ix1 + 1) * (iy2 - iy1 + 1) if ix1 < ix2 and iy1 < iy2 else 0
        a1 = (b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1)
        a2 = (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1)
        return float(inter) / (a1 + a2 - inter)

    try:
        gt = _gt_box(ground_truth)
        data = _pred_json(predict_str)
        if gt and data and len(data.get("bbox", [])) == 4:
            if iou(data["bbox"], gt) > 0.5:
                return 1.0
    except Exception:
        pass
    return 0.0


def seg_box_l1_reward(predict_str: str, ground_truth: str) -> float:
    try:
        gt = _gt_box(ground_truth)
        data = _pred_json(predict_str)
        if gt and data and len(data.get("bbox", [])) == 4:
            b = data["bbox"]
            l1 = sum(abs(b[i] - gt[i]) for i in range(4)) / 4
            if l1 < 10:
                return 1.0
    except Exception:
        pass
    return 0.0


def seg_point_l1_reward(predict_str: str, ground_truth: str) -> float:
    def in_box(p, box):
        return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]

    def dist(p1, p2):
        d1 = (math.hypot(p1[0][0] - p2[0][0], p1[0][1] - p2[0][1])
              + math.hypot(p1[1][0] - p2[1][0], p1[1][1] - p2[1][1]))
        d2 = (math.hypot(p1[0][0] - p2[1][0], p1[0][1] - p2[1][1])
              + math.hypot(p1[1][0] - p2[0][0], p1[1][1] - p2[0][1]))
        return min(d1, d2) / 2

    try:
        gt = _gt_points(ground_truth)
        data = _pred_json(predict_str)
        if gt and data and len(data.get("bbox", [])) == 4:
            box = data["bbox"]
            p1 = [int(data["points_1"][0]), int(data["points_1"][1])]
            p2 = [int(data["points_2"][0]), int(data["points_2"][1])]
            if in_box(p1, box) and in_box(p2, box) and dist([p1, p2], gt) < 100:
                return 1.0
    except Exception:
        pass
    return 0.0


def pixelthink_reward(
    predict_str: str,
    ground_truth: str,
    *,
    uncertainty_score: float,
    difficulty_score: float,
    reasoning_token_count: int | None = None,
    reward_word_token_count: int | None = None,
    length_counting_mode: str = REWARD_WORD_TOKENS,
) -> float:
    acc = (
        seg_thinking_format_reward(predict_str)
        + seg_segmentation_format_reward(predict_str)
        + seg_iou_reward(predict_str, ground_truth)
        + seg_point_l1_reward(predict_str, ground_truth)
        + seg_box_l1_reward(predict_str, ground_truth)
    )

    if length_counting_mode == REWARD_WORD_TOKENS:
        length_used = reward_word_token_count if reward_word_token_count is not None else count_response_word_tokens(predict_str)
    elif length_counting_mode == REWARD_REASONING_TOKENS:
        if reasoning_token_count is None:
            raise ValueError(f"{REWARD_REASONING_TOKENS} requires reasoning_token_count")
        length_used = reasoning_token_count
    else:
        raise ValueError(f"Unknown reward length counting mode: {length_counting_mode}")

    def soft_penalty(budget):
        beta = 1 / 500.0
        delta = length_used - budget
        sc = -beta * delta if delta > 0 else beta * abs(delta)
        return max(0.0, min(1.0, 1 + sc))

    if difficulty_score >= 5.0:
        return acc * soft_penalty(256 + 25 * uncertainty_score)
    elif difficulty_score >= 3.5:
        return acc
    else:
        return acc * soft_penalty(96)
