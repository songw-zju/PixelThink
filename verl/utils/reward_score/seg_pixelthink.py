from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re

from verl.utils.reward_score.length import (
    REWARD_WORD_TOKENS,
    REWARD_REASONING_TOKENS,
    count_response_word_tokens,
)


REWARD_DEBUG = os.environ.get("PIXELTHINK_REWARD_DEBUG", "").lower() in {
    "1",
    "true",
    "yes",
}


@dataclass(frozen=True)
class PixelThinkRewardParams:
    hard_difficulty_threshold: float = 5.0
    medium_difficulty_threshold: float = 3.5
    hard_token_budget: float = 256.0
    easy_token_budget: float = 96.0
    uncertainty_budget_scale: float = 25.0
    length_penalty_beta: float = 1 / 500.0
    iou_threshold: float = 0.5
    box_l1_threshold: float = 10.0
    point_l1_threshold: float = 100.0

    def __post_init__(self):
        if self.hard_difficulty_threshold < self.medium_difficulty_threshold:
            raise ValueError("hard_difficulty_threshold must be >= medium_difficulty_threshold")
        positive_fields = {
            "hard_token_budget": self.hard_token_budget,
            "easy_token_budget": self.easy_token_budget,
            "uncertainty_budget_scale": self.uncertainty_budget_scale,
            "length_penalty_beta": self.length_penalty_beta,
            "iou_threshold": self.iou_threshold,
            "box_l1_threshold": self.box_l1_threshold,
            "point_l1_threshold": self.point_l1_threshold,
        }
        for name, value in positive_fields.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


DEFAULT_PIXELTHINK_REWARD_PARAMS = PixelThinkRewardParams()


def seg_thinking_format_reward(predict_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    return 1.0 if match else 0.0


def seg_segmentation_format_reward(predict_str: str) -> float:
    def is_valid_format(predict_str: str) -> bool:
        try:
            json_match = re.search(r'{[^}]+}', predict_str)
            if not json_match:
                return False
            json_str = json_match.group(0)
            data = json.loads(json_str)

            required_keys = ['bbox', 'points_1', 'points_2']
            for key in required_keys:
                if key not in data:
                    return False

            bbox = data['bbox']
            if not isinstance(bbox, list) or len(bbox) != 4:
                return False

            points_1 = data['points_1']
            points_2 = data['points_2']
            if not isinstance(points_1, list) or len(points_1) != 2:
                return False
            if not isinstance(points_2, list) or len(points_2) != 2:
                return False

            return True
        except Exception:
            return False
    return 1.0 if is_valid_format(predict_str) else 0.0

def seg_iou_reward(
    predict_str: str,
    ground_truth: str,
    *,
    iou_threshold: float = DEFAULT_PIXELTHINK_REWARD_PARAMS.iou_threshold,
) -> float:
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        area1 = (box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)
        area2 = (box2[2]-box2[0]+1)*(box2[3]-box2[1]+1)
        union = area1 + area2 - inter
        return float(inter)/union

    try:
        ground_truth = ground_truth.strip()
        gt_box_pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
        gt_match = re.search(gt_box_pattern, ground_truth)
        if gt_match:
            gt_bbox = [int(gt_match.group(1)), int(gt_match.group(2)), int(gt_match.group(3)), int(gt_match.group(4))]

        json_pattern = r'{[^}]+}'
        json_match = re.search(json_pattern, predict_str)
        if json_match:
            data = json.loads(json_match.group(0))
            bbox_key = 'bbox'
            if bbox_key and len(data[bbox_key]) == 4:
                content_bbox = data[bbox_key]
                if iou(content_bbox, gt_bbox) > iou_threshold:
                    return 1.0
    except Exception:
        pass
    return 0.0


def seg_box_l1_reward(
    predict_str: str,
    ground_truth: str,
    *,
    box_l1_threshold: float = DEFAULT_PIXELTHINK_REWARD_PARAMS.box_l1_threshold,
) -> float:
    def l1_distance(box1, box2):
        return (abs(box1[0]-box2[0]) + abs(box1[1]-box2[1]) + abs(box1[2]-box2[2]) + abs(box1[3]-box2[3])) / 4

    try:
        ground_truth = ground_truth.strip()
        gt_box_pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
        gt_match = re.search(gt_box_pattern, ground_truth)
        if gt_match:
            gt_bbox = [int(gt_match.group(1)), int(gt_match.group(2)), int(gt_match.group(3)), int(gt_match.group(4))]

        json_pattern = r'{[^}]+}'
        json_match = re.search(json_pattern, predict_str)
        if json_match:
            data = json.loads(json_match.group(0))
            bbox_key = 'bbox'
            if bbox_key and len(data[bbox_key]) == 4:
                content_bbox = data[bbox_key]
                if l1_distance(content_bbox, gt_bbox) < box_l1_threshold:
                    return 1.0
    except Exception:
        pass
    return 0.0


def seg_point_l1_reward(
    predict_str: str,
    ground_truth: str,
    *,
    point_l1_threshold: float = DEFAULT_PIXELTHINK_REWARD_PARAMS.point_l1_threshold,
) -> float:
    def points_in_box(point, bbox):
        return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

    def points_distance(points1, points2):
        dist1 = math.sqrt((points1[0][0]-points2[0][0])**2 + (points1[0][1]-points2[0][1])**2) + \
                math.sqrt((points1[1][0]-points2[1][0])**2 + (points1[1][1]-points2[1][1])**2)

        dist2 = math.sqrt((points1[0][0]-points2[1][0])**2 + (points1[0][1]-points2[1][1])**2) + \
                math.sqrt((points1[1][0]-points2[0][0])**2 + (points1[1][1]-points2[0][1])**2)
        return min(dist1, dist2) / 2

    try:
        gt_points_pattern = r'<points>\((\d+),(\d+)\),\((\d+),(\d+)\)</points>'
        gt_match = re.search(gt_points_pattern, ground_truth)
        if gt_match:
            gt_points = [[int(gt_match.group(1)), int(gt_match.group(2))], [int(gt_match.group(3)), int(gt_match.group(4))]]

        json_pattern = r'{[^}]+}'
        json_match = re.search(json_pattern, predict_str)

        if json_match:
            data = json.loads(json_match.group(0))
            bbox_key = 'bbox'
            if bbox_key and len(data[bbox_key]) == 4:
                content_bbox = data[bbox_key]
            points_keys = ['points_1', 'points_2']
            if len(points_keys) == 2:
                point1 = data[points_keys[0]]
                point2 = data[points_keys[1]]
                point1 = [int(point1[0]), int(point1[1])]
                point2 = [int(point2[0]), int(point2[1])]
                if points_in_box(point1, content_bbox) and points_in_box(point2, content_bbox):
                    if points_distance([point1, point2], gt_points) < point_l1_threshold:
                        return 1.0
    except Exception:
        pass
    return 0.0


def pixelthink_accuracy_reward(
    predict_str: str,
    ground_truth: str,
    params: PixelThinkRewardParams = DEFAULT_PIXELTHINK_REWARD_PARAMS,
) -> float:
    return (
        seg_thinking_format_reward(predict_str) +
        seg_segmentation_format_reward(predict_str) +
        seg_iou_reward(predict_str, ground_truth, iou_threshold=params.iou_threshold) +
        seg_point_l1_reward(
            predict_str,
            ground_truth,
            point_l1_threshold=params.point_l1_threshold,
        ) +
        seg_box_l1_reward(
            predict_str,
            ground_truth,
            box_l1_threshold=params.box_l1_threshold,
        )
    )


def pixelthink_length_used(
    predict_str: str,
    *,
    reasoning_token_count: int | None,
    reward_word_token_count: int | None,
    length_counting_mode: str,
) -> int:
    if length_counting_mode == REWARD_WORD_TOKENS:
        return reward_word_token_count if reward_word_token_count is not None else count_response_word_tokens(predict_str)
    if length_counting_mode == REWARD_REASONING_TOKENS:
        if reasoning_token_count is None:
            raise ValueError(f"{REWARD_REASONING_TOKENS} requires reasoning_token_count")
        return reasoning_token_count
    raise ValueError(f"Unknown reward length counting mode: {length_counting_mode}")


def pixelthink_length_budget(
    *,
    difficulty_score: float,
    uncertainty_score: float,
    params: PixelThinkRewardParams = DEFAULT_PIXELTHINK_REWARD_PARAMS,
) -> float | None:
    if difficulty_score >= params.hard_difficulty_threshold:
        return params.hard_token_budget + params.uncertainty_budget_scale * uncertainty_score
    if difficulty_score >= params.medium_difficulty_threshold:
        return None
    return params.easy_token_budget


def pixelthink_length_score(
    *,
    length_used: int,
    length_budget: float | None,
    params: PixelThinkRewardParams = DEFAULT_PIXELTHINK_REWARD_PARAMS,
) -> float:
    if length_budget is None:
        return 1.0
    delta = length_used - length_budget
    if delta > 0:
        score_delta = -params.length_penalty_beta * delta
    else:
        score_delta = params.length_penalty_beta * abs(delta)
    return max(0.0, min(1.0, 1 + score_delta))


def seg_pixelthink_compute_score(
    predict_str: str,
    ground_truth: str,
    uncertainty_score: float = 0.0,
    difficulty_score: float = 2.0,
    *,
    reasoning_token_count: int | None = None,
    reward_word_token_count: int | None = None,
    length_counting_mode: str = REWARD_WORD_TOKENS,
    reward_params: PixelThinkRewardParams = DEFAULT_PIXELTHINK_REWARD_PARAMS,
) -> float:
    acc_reward = pixelthink_accuracy_reward(predict_str, ground_truth, reward_params)
    length_used = pixelthink_length_used(
        predict_str,
        reasoning_token_count=reasoning_token_count,
        reward_word_token_count=reward_word_token_count,
        length_counting_mode=length_counting_mode,
    )
    length_budget = pixelthink_length_budget(
        difficulty_score=difficulty_score,
        uncertainty_score=uncertainty_score,
        params=reward_params,
    )

    if REWARD_DEBUG:
        if length_budget is None:
            branch = "medium"
        elif difficulty_score >= reward_params.hard_difficulty_threshold:
            branch = "hard"
        else:
            branch = "easy"
        print(f"[pixelthink_reward] difficulty_branch={branch}")

    return acc_reward * pixelthink_length_score(
        length_used=length_used,
        length_budget=length_budget,
        params=reward_params,
    )
