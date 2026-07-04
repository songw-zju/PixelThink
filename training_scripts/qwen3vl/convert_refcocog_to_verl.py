import argparse
import json
import os

import datasets


SEG_PROMPT = (
    "Please find '{Question}' with bbox and points."
    "Compare the difference between objects and find the most closely matched one."
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format."
    "i.e., <think> thinking process here </think>"
    "<answer>{Answer}</answer>"
)
SEG_PROMPT_ANSWER_HINT = "{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}"

DEFAULT_DIFFICULTY = 3.0


def load_difficulty(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    out = {}
    for item in records:
        d = item.get("difficulty_score_model")
        if isinstance(d, (int, float)) and 1.0 <= d <= 10.0:
            out[item.get("image_id")] = float(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Ricky06662/refCOCOg_9k_840")
    ap.add_argument("--difficulty-json", required=True,
                    help="path to refcoco_train_labeled_merged.json")
    ap.add_argument("--out", required=True, help="output directory for train.parquet")
    ap.add_argument("--num-proc", type=int, default=4)
    args = ap.parse_args()

    diff = load_difficulty(args.difficulty_json)
    print(f"difficulty table: {len(diff)} entries")

    ds = datasets.load_dataset(args.repo)["train"]
    print(f"source dataset: {len(ds)} rows, cols={ds.column_names}")

    n_missing = sum(1 for image_id in ds["id"] if image_id not in diff)

    def process(example, idx):
        question = example["problem"].lower().strip(".")
        content = "<image>" + SEG_PROMPT.format(Question=question, Answer=SEG_PROMPT_ANSWER_HINT)
        d = diff.get(example["id"])
        if d is None:
            d = DEFAULT_DIFFICULTY
        return {
            "data_source": "pixelthink/refcocog",
            "prompt": [{"role": "user", "content": content}],
            "images": [example["image"]],
            "ability": "reasoning_segmentation",
            "reward_model": {"style": "rule", "ground_truth": example["solution"]},
            "extra_info": {
                "split": "train",
                "index": idx,
                "id": example["id"],
                "difficulty": float(d),
                "question": question,
            },
        }

    ds = ds.map(process, with_indices=True, remove_columns=ds.column_names,
                num_proc=args.num_proc)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "train.parquet")
    ds.to_parquet(out_path)
    print(f"wrote {len(ds)} rows -> {out_path}")
    print(f"difficulty: {len(ds) - n_missing} matched, {n_missing} fell back to {DEFAULT_DIFFICULTY}")


if __name__ == "__main__":
    main()
