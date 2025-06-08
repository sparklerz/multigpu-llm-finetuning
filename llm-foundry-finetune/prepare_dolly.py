#!/usr/bin/env python
import os
import json
import pathlib
from datasets import load_dataset

# 1. Prepare output folder
OUT_DIR = pathlib.Path("data/dolly_15k_json")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load the Dolly dataset (only 'train' exists)
print("🔄 Loading Dolly-15K…")
raw = load_dataset("databricks/databricks-dolly-15k")["train"]

# 3. First split: 90% train, 10% temp
print("🔀 Splitting 90% train / 10% temp…")
split1 = raw.train_test_split(test_size=0.10, seed=42)
train_ds = split1["train"]
temp_ds  = split1["test"]

# 4. Further split temp into 50/50 → 5% val, 5% test (optional)
print("🔀 Splitting temp into 50% validation / 50% test…")
split2 = temp_ds.train_test_split(test_size=0.50, seed=42)
val_ds  = split2["train"]
test_ds = split2["test"]

# 5. A helper to dump a Dataset to JSONL
def dump(ds, name):
    print(f"✏️  Writing {name} ({len(ds)} examples)…")
    with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for ex in ds:
            f.write(
                json.dumps({
                    "prompt":   ex["instruction"],
                    "response": ex["response"],
                }, ensure_ascii=False)
                + "\n"
            )
    print(f"   ✓ Finished {name}")

# 6. Dump each split
dump(train_ds, "train")
dump(val_ds,   "validation")
dump(test_ds,  "test")

print(f"🎉 All splits dumped to {OUT_DIR}")