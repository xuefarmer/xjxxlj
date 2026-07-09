#!/usr/bin/env python3
"""
Split each CrossVid task's question JSON into train (2/3) and test (1/3).
Stratified: each task is split independently to preserve per-task distribution.
"""

import json
import os
import random
from pathlib import Path

SEED = 42
TRAIN_RATIO = 2 / 3  # ~66.7% train, ~33.3% test

QUESTION_DIR = Path(__file__).resolve().parent / "question"
TRAIN_DIR = QUESTION_DIR / "train"
TEST_DIR = QUESTION_DIR / "test"


def split_task(json_path: Path):
    """Split a single task JSON file into train/test."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        print(f"  ⚠️  {json_path.name}: empty or not a list, skipping")
        return

    # Deterministic shuffle
    rng = random.Random(SEED)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    split_point = round(len(data) * TRAIN_RATIO)
    train_indices = set(indices[:split_point])
    test_indices = set(indices[split_point:])

    train_data = [item for i, item in enumerate(data) if i in train_indices]
    test_data = [item for i, item in enumerate(data) if i in test_indices]

    return train_data, test_data


def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(QUESTION_DIR.glob("*.json"))
    if not json_files:
        print("❌ No JSON files found in question/")
        return

    print(f"Seed: {SEED} | Train ratio: {TRAIN_RATIO:.0%} | Test ratio: {1-TRAIN_RATIO:.0%}\n")

    total_all = 0
    total_train = 0
    total_test = 0

    for json_path in json_files:
        result = split_task(json_path)
        if result is None:
            continue

        train_data, test_data = result
        n_total = len(train_data) + len(test_data)
        actual_ratio = len(train_data) / n_total * 100

        # Save train
        train_path = TRAIN_DIR / json_path.name
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        # Save test
        test_path = TEST_DIR / json_path.name
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        print(f"  {json_path.name:12s} → train: {len(train_data):5d}  test: {len(test_data):5d}  "
              f"({actual_ratio:.0f}/{(100-actual_ratio):.0f}%)")

        total_all += n_total
        total_train += len(train_data)
        total_test += len(test_data)

    print(f"\n{'─'*50}")
    print(f"  Total: {total_all} → train: {total_train} ({total_train/total_all*100:.0f}%)  "
          f"test: {total_test} ({total_test/total_all*100:.0f}%)")
    print(f"  Train dir: {TRAIN_DIR}")
    print(f"  Test dir:  {TEST_DIR}")


if __name__ == "__main__":
    main()
