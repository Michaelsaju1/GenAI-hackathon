"""
Format labeled data into training examples using Llama-3.2 chat template.

Input:  data/labeled/all_labeled.jsonl
Output: data/training/{train,val,test}.jsonl
"""

import json
import random

from config import LABELED_DIR, TRAINING_DIR, TRAIN_RATIO, VAL_RATIO

INPUT_FILE = LABELED_DIR / "all_labeled.jsonl"

SYSTEM_MESSAGE = (
    "You are a geopolitical risk analyst specializing in North Korean military activity. "
    "Analyze GDELT event data and produce structured risk assessments in JSON format."
)


def format_example(item: dict) -> dict:
    """Format a labeled item into a chat-style training example."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": (
                    "Analyze the following GDELT event data for North Korean military "
                    "activity and produce a structured risk assessment.\n\n"
                    + item["summary_text"]
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(item["risk_assessment"], indent=2),
            },
        ]
    }


def main():
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # Load labeled data
    labeled = []
    with open(INPUT_FILE) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Skip items with failed labels
                if "risk_assessment" in item and item["risk_assessment"]:
                    labeled.append(item)
    print(f"Loaded {len(labeled)} labeled examples")

    # Format into training examples
    examples = [format_example(item) for item in labeled]

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(examples)

    # Split
    n = len(examples)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:]

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # Save
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = TRAINING_DIR / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for example in split_data:
                f.write(json.dumps(example) + "\n")
        print(f"  Saved {path} ({len(split_data)} examples)")

    # Token length estimation (rough: 1 token ≈ 4 chars)
    lengths = []
    for ex in examples:
        total_chars = sum(len(m["content"]) for m in ex["messages"])
        lengths.append(total_chars // 4)
    lengths.sort()
    print(f"\nEstimated token lengths:")
    print(f"  Min: {lengths[0]}, Max: {lengths[-1]}, Mean: {sum(lengths)//len(lengths)}")
    print(f"  P95: {lengths[int(len(lengths)*0.95)]}")

    # Print one example
    print(f"\n{'='*60}")
    print("Sample training example:")
    print(f"{'='*60}")
    for msg in examples[0]["messages"]:
        print(f"\n[{msg['role']}]")
        print(msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"])


if __name__ == "__main__":
    main()
