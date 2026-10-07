# ============================================================
# NB3 — TRAIN CORRECT CONFIGURATION
# ============================================================

import json
import os
import pathlib
import sys
import time


# ------------------------------------------------------------
# 0. PATH SETUP
# ------------------------------------------------------------

sys.path.insert(
    0,
    str(pathlib.Path.cwd() / "src")
)

sys.path.insert(
    0,
    str(pathlib.Path.cwd().parent / "src")
)


from labkit import data
from labkit import device
from labkit import generate
from labkit import modeling
from labkit import report
from labkit import train

from labkit.config import (
    SPECS,
    get_tier,
    training_epochs
)


ROOT = (
    pathlib.Path.cwd()
    if (pathlib.Path.cwd() / "data").exists()
    else pathlib.Path.cwd().parent
)


os.environ.setdefault(
    "COMPUTE_TIER",
    "T4"
)


TIER = get_tier(
    os.environ.get(
        "COMPUTE_TIER",
        "T4"
    )
)


SPEC = SPECS["correct"]


print("=" * 70)
print("NB3 — CORRECT TRAINING")
print("=" * 70)

print(
    f"Tier  : {TIER.name}"
)

print(
    f"Model : {TIER.model_id}"
)

print(
    f"Spec  : {SPEC.label}"
)

print()

print(
    device.banner()
)


# ============================================================
# 1. LOAD BASE MODEL
# ============================================================

print("\n" + "=" * 70)
print("1. LOAD BASE MODEL")
print("=" * 70)


model, tok = generate.load_base(

    TIER,

    load_in_4bit=SPEC.load_in_4bit
)


# ============================================================
# 2. MODEL ARCHITECTURE
# ============================================================

print("\n" + "=" * 70)
print("2. MODEL ARCHITECTURE")
print("=" * 70)


layer_summary = (
    modeling.layer_type_summary(
        model.config
    )
)


print(
    json.dumps(
        layer_summary,
        ensure_ascii=False,
        indent=2
    )
)


# ============================================================
# 3. RESOLVE LORA TARGET MODULES
# ============================================================

print("\n" + "=" * 70)
print("3. LoRA TARGET MODULES")
print("=" * 70)


targets = (
    modeling.resolve_target_modules(
        model,
        SPEC.target
    )
)


trainable_params = (
    modeling.count_lora_params(
        model,
        targets,
        SPEC.r
    )
)


print(
    "Placement:",
    SPEC.target
)


print(
    "Target modules:"
)

for module in targets:

    print(
        "  -",
        module
    )


print(
    f"\nTrainable LoRA params ≈ "
    f"{trainable_params / 1e6:.2f} M"
)


print(
    "\nPlacement detail:"
)


for row in modeling.describe_placement(
    model,
    SPEC.r
):

    print(
        "   ",
        row
    )


# ============================================================
# 4. LOAD TRAIN SPLIT
# ============================================================

print("\n" + "=" * 70)
print("4. LOAD TRAIN SPLIT")
print("=" * 70)


from datasets import Dataset


split_dir = (
    ROOT /
    "data" /
    "split"
)


assert split_dir.exists(), (
    "Không có data/split/. "
    "Bạn phải chạy NB1 trước."
)


train_file = (
    split_dir /
    "train.jsonl"
)


assert train_file.exists(), (
    "Không có train.jsonl. "
    "Chạy NB1 trước."
)


def load_jsonl(path):

    rows = []

    with open(
        path,
        encoding="utf-8"
    ) as fh:

        for line in fh:

            if line.strip():

                rows.append(
                    json.loads(line)
                )

    return rows


train_rows = load_jsonl(
    train_file
)


print(
    f"Train samples: {len(train_rows)}"
)


# ============================================================
# 5. BUILD TOKENIZED TRAIN DATASET
# ============================================================

MASK_MODE = os.environ.get(
    "MASK_MODE",
    "assistant-only"
)


print(
    "Mask mode:",
    MASK_MODE
)


rows = data.to_training_dataset(

    tok,

    train_rows,

    max_length=TIER.max_length,

    mask_mode=MASK_MODE
)


train_ds = Dataset.from_list(
    rows
)


# ------------------------------------------------------------
# Count supervised tokens
# ------------------------------------------------------------

supervised_tokens = sum(

    sum(
        1
        for token in row["labels"]
        if token != data.IGNORE_INDEX
    )

    for row in rows
)


total_tokens = sum(

    len(row["labels"])

    for row in rows
)


supervised_fraction = (
    supervised_tokens /
    total_tokens
)


print(
    train_ds
)


print(
    f"Supervised tokens:"
    f" {supervised_tokens}"
    f"/{total_tokens}"
    f" ({supervised_fraction:.2%})"
)


assert (
    0
    <
    supervised_tokens
    <
    total_tokens
), (
    "Mask đang cover toàn bộ hoặc không cover token nào."
)


assert supervised_fraction < 0.95, (
    "Mask có vẻ đang tính loss cả prompt."
)


# ============================================================
# 6. TRAINING CONFIG
# ============================================================

print("\n" + "=" * 70)
print("6. TRAINING CONFIG")
print("=" * 70)


from peft import LoraConfig
from trl import SFTConfig
from trl import SFTTrainer


# ------------------------------------------------------------
# Epochs
# NB4 phải dùng cùng số epoch / step này
# ------------------------------------------------------------

EPOCHS = training_epochs()


STEPS = train.planned_steps(

    len(rows),

    TIER,

    EPOCHS
)


print(
    f"EPOCHS = {EPOCHS}"
)

print(
    f"Optimizer steps = {STEPS}"
)


# ============================================================
# 7. SFT CONFIG
# ============================================================

output_dir = (
    ROOT /
    "adapters" /
    SPEC.key
)


want_sft = train.sft_config_kwargs(

    TIER,

    SPEC,

    output_dir=str(
        output_dir
    ),

    num_train_epochs=EPOCHS,

    mask_mode=MASK_MODE,

    total_steps=STEPS,
)


sft_kwargs, dropped = (
    train.filter_kwargs(

        SFTConfig,

        want_sft,

        label="SFTConfig"
    )
)


if dropped:

    print(
        "\n⚠ TRL không nhận các tham số:"
    )

    for key in dropped:

        print(
            "  -",
            key
        )


# ============================================================
# 8. LORA CONFIG
# ============================================================

want_lora = (
    train.lora_config_kwargs(
        SPEC,
        targets
    )
)


lora_kwargs, dropped_lora = (
    train.filter_kwargs(

        LoraConfig,

        want_lora,

        label="LoraConfig"
    )
)


if dropped_lora:

    print(
        "\n⚠ PEFT không nhận:"
    )

    for key in dropped_lora:

        print(
            "  -",
            key
        )


print(
    "\nSFT Config:"
)


print(
    json.dumps(
        {
            key: str(value)
            for key, value
            in sft_kwargs.items()
        },
        indent=2
    )
)


print(
    "\nLoRA Config:"
)


print(
    json.dumps(
        {
            key: str(value)
            for key, value
            in lora_kwargs.items()
        },
        indent=2
    )
)


# ============================================================
# 9. FREE UNUSED MEMORY
# ============================================================

print("\n" + "=" * 70)
print("9. PREPARE TRAINER")
print("=" * 70)


generate.free_memory()


# ============================================================
# 10. CREATE TRAINER
# ============================================================

trainer = SFTTrainer(

    model=model,

    args=SFTConfig(
        **sft_kwargs
    ),

    train_dataset=train_ds,

    processing_class=tok,

    peft_config=LoraConfig(
        **lora_kwargs
    ),
)


# ============================================================
# 11. PRECISION FIX
# ============================================================

fix = (
    train.align_trainable_precision(
        trainer.model
    )
)


print(
    "Precision fix:",
    fix
)


# ============================================================
# 12. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("12. TRAINING START")
print("=" * 70)


start_time = (
    time.perf_counter()
)


result = trainer.train()


elapsed = (
    time.perf_counter()
    -
    start_time
)


final_loss = (
    result.training_loss
)


print("\n" + "=" * 70)
print("TRAINING FINISHED")
print("=" * 70)


print(
    f"Elapsed time:"
    f" {elapsed:.2f} sec"
)


print(
    f"Final loss:"
    f" {final_loss:.6f}"
)


# ============================================================
# 13. SAVE ADAPTER IMMEDIATELY
# ============================================================

print("\n" + "=" * 70)
print("13. SAVE ADAPTER")
print("=" * 70)


output_dir.mkdir(
    parents=True,
    exist_ok=True
)


trainer.model.save_pretrained(
    output_dir
)


tok.save_pretrained(
    output_dir
)


print(
    "Adapter saved ->",
    output_dir
)


# ============================================================
# 14. SAVE RUN METRICS
# ============================================================

peak_vram = (
    generate.peak_vram_gb()
)


row = train.summarize_run(

    SPEC,

    TIER,

    targets,

    trainable_params,

    elapsed,

    peak_vram
)


row["final_loss"] = round(
    final_loss,
    4
)


row["mask_mode"] = (
    MASK_MODE
)


row["max_steps"] = (
    STEPS
)


report.append_row(

    row,

    results_dir=ROOT / "results"
)


print("\nRun summary:")


print(
    json.dumps(
        row,
        ensure_ascii=False,
        indent=2
    )
)


# ============================================================
# 15. VERIFY SAVED FILES
# ============================================================

print("\n" + "=" * 70)
print("15. NB3 FINAL CHECK")
print("=" * 70)


adapter_config = (
    output_dir /
    "adapter_config.json"
)


possible_weight_files = [

    output_dir /
    "adapter_model.safetensors",

    output_dir /
    "adapter_model.bin",
]


runs_csv = (
    ROOT /
    "results" /
    "runs.csv"
)


print(
    "[OK]"
    if adapter_config.exists()
    else "[MISSING]",
    adapter_config
)


weight_exists = any(
    path.exists()
    for path
    in possible_weight_files
)


print(
    "[OK]"
    if weight_exists
    else "[MISSING]",
    "adapter weights"
)


print(
    "[OK]"
    if runs_csv.exists()
    else "[MISSING]",
    runs_csv
)


assert adapter_config.exists(), (
    "Thiếu adapter_config.json"
)


assert weight_exists, (
    "Thiếu adapter_model weights"
)


assert runs_csv.exists(), (
    "Thiếu results/runs.csv"
)


print(
    "\n✅ NB3 HOÀN THÀNH"
)

print(
    f"Final loss : {final_loss:.4f}"
)

print(
    f"Peak VRAM  : {peak_vram:.2f} GB"
)

print(
    f"Steps      : {STEPS}"
)

print(
    f"Epochs     : {EPOCHS}"
)