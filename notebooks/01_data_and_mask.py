# ============================================================
# NB1 — DATA, CHAT TEMPLATE & MASK
# ============================================================

import json
import os
import pathlib
import sys

# ------------------------------------------------------------
# 0. Path setup
# ------------------------------------------------------------

sys.path.insert(0, str(pathlib.Path.cwd() / "src"))
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))

from labkit import data, report
from labkit.config import get_tier


ROOT = (
    pathlib.Path.cwd()
    if (pathlib.Path.cwd() / "data").exists()
    else pathlib.Path.cwd().parent
)

# Default dùng T4
os.environ.setdefault("COMPUTE_TIER", "T4")

TIER = get_tier(os.environ.get("COMPUTE_TIER", "T4"))

print("=" * 70)
print("NB1 — DATA, TEMPLATE & MASK")
print("=" * 70)
print(f"ROOT       : {ROOT}")
print(f"Tier       : {TIER.name}")
print(f"Model      : {TIER.model_id}")
print(f"max_length : {TIER.max_length}")


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_jsonl(path):
    rows = []

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    return rows


train_path = ROOT / "data" / "train_seed.jsonl"

assert train_path.exists(), f"Không tìm thấy dataset: {train_path}"

train_raw = load_jsonl(train_path)

print("\n" + "=" * 70)
print("1. DATASET")
print("=" * 70)

print(f"Số mẫu training: {len(train_raw)}")

assert len(train_raw) > 0, "Dataset rỗng"

print("\nVí dụ mẫu đầu tiên:")
print(
    json.dumps(
        train_raw[0],
        ensure_ascii=False,
        indent=2
    )[:1000]
)


# ============================================================
# 2. LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("2. TOKENIZER")
print("=" * 70)

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    TIER.model_id,
    trust_remote_code=True
)

print("Tokenizer loaded")
print("EOS token     :", tok.eos_token)
print("EOS token id  :", tok.eos_token_id)
print("PAD token     :", tok.pad_token)
print("PAD token id  :", tok.pad_token_id)


# ============================================================
# 3. CHECK CHAT TEMPLATE / THINKING
# ============================================================

print("\n" + "=" * 70)
print("3. CHAT TEMPLATE CHECK")
print("=" * 70)

check = data.thinking_survives(tok)

print("VERDICT:", check["verdict"])

print("\n--- Rendered string ---")
print(check["rendered"])

report.write_json(
    check,
    "template_check.json",
    results_dir=ROOT / "results"
)

print(
    "\nSaved:",
    ROOT / "results" / "template_check.json"
)


# ============================================================
# 4. BUILD SAMPLE MESSAGE
# ============================================================

sample = data.to_messages(train_raw[0])

print("\n" + "=" * 70)
print("4. SAMPLE MESSAGES")
print("=" * 70)

print(
    json.dumps(
        sample,
        ensure_ascii=False,
        indent=2
    )
)


# ============================================================
# 5. COMPARE MASK MODES
# ============================================================

print("\n" + "=" * 70)
print("5. MASK COMPARISON")
print("=" * 70)

for mode in (
    "assistant-only",
    "everything",
):

    ex = data.build_example(
        tok,
        sample,
        max_length=TIER.max_length,
        mask_mode=mode
    )

    print("\n" + "-" * 70)

    print(
        f"mode={mode}"
        f" | supervised={ex.n_supervised}/{ex.n_total}"
        f" | fraction={ex.supervised_fraction:.2%}"
    )

    print("\n--- TOKENS ĐƯỢC TÍNH LOSS ---")

    supervised_text = data.decode_supervised(
        tok,
        ex
    )

    print(supervised_text[:1000])


# ============================================================
# 6. MASK PROOF
# ============================================================

print("\n" + "=" * 70)
print("6. MASK PROOF")
print("=" * 70)

MASK_MODE = "assistant-only"

ex = data.build_example(
    tok,
    sample,
    max_length=TIER.max_length,
    mask_mode=MASK_MODE
)

supervised = data.decode_supervised(
    tok,
    ex
)

masked = data.decode_masked(
    tok,
    ex
)

answer = sample[-1]["content"][:40]

question_fragment = train_raw[0]["input"][:40]

proof = {
    "mask_mode": MASK_MODE,

    "n_supervised": ex.n_supervised,

    "n_total": ex.n_total,

    "supervised_fraction": round(
        ex.supervised_fraction,
        4
    ),

    "answer_is_supervised":
        answer in supervised,

    "question_is_masked":
        question_fragment not in supervised,

    "supervised_preview":
        supervised[:300],

    "masked_preview":
        masked[:300],
}


# ------------------------------------------------------------
# Assertions bắt buộc
# ------------------------------------------------------------

assert proof["answer_is_supervised"], (
    "Câu trả lời KHÔNG nằm trong loss. "
    "Mask đang sai."
)

assert proof["question_is_masked"], (
    "Câu hỏi ĐANG nằm trong loss. "
    "Mask đang sai."
)

assert 0 < proof["supervised_fraction"] < 0.95, (
    "supervised_fraction bất thường"
)


print(
    json.dumps(
        {
            k: v
            for k, v in proof.items()
            if not k.endswith("preview")
        },
        ensure_ascii=False,
        indent=2
    )
)


report.write_json(
    proof,
    "mask_proof.json",
    results_dir=ROOT / "results"
)

print(
    "\nSaved:",
    ROOT / "results" / "mask_proof.json"
)


# ============================================================
# 7. TOKEN LENGTH STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("7. TOKEN LENGTH STATS")
print("=" * 70)

lengths = []

for row in train_raw:

    messages = data.to_messages(row)

    example = data.build_example(
        tok,
        messages,
        max_length=8192
    )

    lengths.append(example.n_total)


stats = data.token_stats(lengths)

print(
    json.dumps(
        stats,
        ensure_ascii=False,
        indent=2
    )
)


report.write_json(
    stats,
    "token_stats.json",
    results_dir=ROOT / "results"
)


print(
    "\nSaved:",
    ROOT / "results" / "token_stats.json"
)


if stats["suggested_max_length"] != TIER.max_length:

    print(
        "\nWARNING:"
        f" p95 đề xuất max_length="
        f"{stats['suggested_max_length']}"
        f" nhưng tier hiện tại="
        f"{TIER.max_length}"
    )


# ============================================================
# 8. FIXED TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 70)
print("8. TRAIN / VALIDATION SPLIT")
print("=" * 70)

train_rows, val_rows = data.split(
    train_raw,
    train_frac=0.9,
    seed=42
)


split_dir = ROOT / "data" / "split"

split_dir.mkdir(
    parents=True,
    exist_ok=True
)


for name, rows in (
    ("train", train_rows),
    ("val", val_rows),
):

    output_file = split_dir / f"{name}.jsonl"

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as fh:

        for row in rows:

            fh.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"Saved {len(rows)} samples"
        f" -> {output_file}"
    )


# ============================================================
# 9. FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("NB1 FINAL CHECK")
print("=" * 70)

required_files = [

    ROOT / "results" / "template_check.json",

    ROOT / "results" / "mask_proof.json",

    ROOT / "results" / "token_stats.json",

    ROOT / "data" / "split" / "train.jsonl",

    ROOT / "data" / "split" / "val.jsonl",
]


all_ok = True

for file in required_files:

    exists = file.exists()

    print(
        "[OK]" if exists else "[MISSING]",
        file
    )

    all_ok &= exists


if all_ok:

    print("\n✅ NB1 HOÀN THÀNH")

else:

    raise RuntimeError(
        "NB1 chưa sinh đủ artefact"
    )