#!/usr/bin/env python
import os
import json
import pathlib
from datasets import load_dataset, DownloadConfig

# 1. Increase HF Datasets download timeout & retries
os.environ["HF_DATASETS_DOWNLOAD_INFO"] = "1"
download_config = DownloadConfig(
    max_retries=10      # retry up to 10 times
)

# 2. Prepare output folder
OUT_DIR = pathlib.Path("data/dolly_15k_json")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 3. Log and load the Dolly dataset via HF with our custom download config
print("🔄 Starting to load the Dolly-15K dataset from Hugging Face…")
ds = load_dataset(
    "databricks/databricks-dolly-15k",
    download_config=download_config,
    use_auth_token=False
)
print(f"✅ Dataset loaded. Splits available: {list(ds.keys())}")

# 4. Dump splits to JSONL
for split, subset in ds.items():
    print(f"✏️  Writing split '{split}' with {len(subset)} examples…")
    with open(OUT_DIR / f"{split}.jsonl", "w", encoding="utf-8") as f:
        for ex in subset:
            f.write(
                json.dumps({
                    "prompt":   ex["instruction"],
                    "response": ex["response"],
                }, ensure_ascii=False)
                + "\n"
            )
    print(f"   ✓ Finished '{split}'")

print(f"🎉 Dolly-15K dumped to {OUT_DIR}")