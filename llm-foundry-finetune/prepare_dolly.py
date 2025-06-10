#!/usr/bin/env python
import os
import json
import pathlib
from datasets import load_dataset

# 1. Prepare output folder
# Always place the dataset at the repository root so that it matches the
# paths expected by the training config regardless of the current working
# directory when this script is executed.  Additionally create a symlink
# from ``llm-foundry-finetune/data`` to the repository root so that older
# configs that expect the data relative to ``llm-foundry-finetune`` also
# work.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_DIR = REPO_ROOT / "data" / "dolly_15k_txt"
BASE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = BASE_DIR

# Backwards compatibility: ``finetune_mpt7b.yaml`` previously looked for the
# dataset under ``llm-foundry-finetune/data``.  Create a symlink so both paths
# resolve to the same directory.
ALT_DIR = pathlib.Path(__file__).parent / "data" / "dolly_15k_txt"
if ALT_DIR != BASE_DIR:
    ALT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not ALT_DIR.exists():
        try:
            ALT_DIR.symlink_to(BASE_DIR, target_is_directory=True)
        except OSError:
            # Fall back to copying if symlinks are not supported
            import shutil
            shutil.copytree(BASE_DIR, ALT_DIR)

# Additional symlink so running training scripts from the installed
# `llmfoundry.scripts.train` directory can still resolve the dataset via
# the relative path ``data/dolly_15k_txt``.
TRAIN_SCRIPT_DIR = pathlib.Path(__file__).parent / "llm-foundry" / "scripts" / "train"
ALT_TRAIN_DIR = TRAIN_SCRIPT_DIR / "data" / "dolly_15k_txt"
if ALT_TRAIN_DIR != BASE_DIR:
    ALT_TRAIN_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not ALT_TRAIN_DIR.exists():
        try:
            ALT_TRAIN_DIR.symlink_to(BASE_DIR, target_is_directory=True)
        except OSError:
            import shutil
            shutil.copytree(BASE_DIR, ALT_TRAIN_DIR)

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
def dump_jsonl(ds, name):
    """Dump the dataset split to JSONL files.

    Files are written both at the dataset root (``{name}.jsonl``) and inside a
    split subfolder (``{name}/{name}.jsonl``).  Having the files in a subfolder
    mirrors the layout expected by some training configs and avoids confusing
    errors when loading the dataset with Hugging Face ``load_dataset``.
    """

    examples = [
        json.dumps({
            "prompt": ex["instruction"],
            "response": ex["response"],
        }, ensure_ascii=False) + "\n" for ex in ds
    ]

    # Root-level file (backwards compatibility)
    print(f"✏️  Writing {name}.jsonl ({len(ds)} examples)…")
    with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(examples)

    # Split subfolder file (preferred by HF loader)
    split_dir = BASE_DIR / name
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(examples)

    print(f"   ✓ Finished {name}.jsonl")

# New: dump a simple .txt file, one line per example
def dump_txt(ds, name):
    """Dump a plain-text version of the dataset split.

    The text files are written inside a ``{name}`` subdirectory so that Hugging
    Face's ``text`` loader can discover them using ``data_files={'train':
    'path/train.txt'}``.
    """

    lines = [f"{ex['instruction']} [SEP] {ex['response']}\n" for ex in ds]

    split_dir = BASE_DIR / name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"✏️  Writing {name}.txt ({len(ds)} examples)…")
    with open(split_dir / f"{name}.txt", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"   ✓ Finished {name}.txt")

if __name__ == "__main__":
    # JSONL still available if you need it:
    dump_jsonl(train_ds, "train")
    dump_jsonl(val_ds,   "validation")
    dump_jsonl(test_ds,  "test")

    # Plain-text for the `text` loader:
    dump_txt(train_ds, "train")
    dump_txt(val_ds,   "validation")
    dump_txt(test_ds,  "test")

print(f"🎉 All splits dumped to {BASE_DIR}")
