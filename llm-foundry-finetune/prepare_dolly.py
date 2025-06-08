from datasets import load_dataset
import json, pathlib, os

OUT_DIR = pathlib.Path("data/dolly_15k_json")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ds = load_dataset("databricks/databricks-dolly-15k")
for split in ("train", "test", "validation"):
    with open(OUT_DIR / f"{split}.jsonl", "w", encoding="utf-8") as f:
        for ex in ds[split]:
            f.write(
                json.dumps(
                    {
                        "prompt":   ex["instruction"],
                        "response": ex["response"],
                    },
                    ensure_ascii=False
                )
                + "\n"
            )
print("✓ Dolly 15 K dumped to", OUT_DIR)