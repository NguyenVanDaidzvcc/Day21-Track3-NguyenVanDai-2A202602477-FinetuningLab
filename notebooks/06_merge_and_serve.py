# ============================================================
# NB6 — MERGE, VERIFY AFTER MERGE & ADAPTER HOT-SWAP
#
# Bonus B1:
#   1. Đánh giá adapter correct trước merge
#   2. Merge LoRA vào base model
#   3. Đánh giá lại sau merge
#   4. Điểm không được tụt quá 0.01
#   5. Hot-swap >= 2 adapter trên cùng một base model
# ============================================================

import json
import os
import pathlib
import sys


# ============================================================
# 0. PATH SETUP
# ============================================================

sys.path.insert(
    0,
    str(pathlib.Path.cwd() / "src")
)

sys.path.insert(
    0,
    str(pathlib.Path.cwd().parent / "src")
)


from labkit import evaluate as ev
from labkit import generate
from labkit import report

from labkit.config import get_tier


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


print("=" * 70)
print("NB6 — MERGE & MULTI-ADAPTER SERVING")
print("=" * 70)

print("ROOT :", ROOT)
print("Tier :", TIER.name)
print("Model:", TIER.model_id)


# ============================================================
# 1. HELPERS
# ============================================================

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


# ============================================================
# 2. CHECK REQUIRED FILES
# ============================================================

print("\n" + "=" * 70)
print("2. CHECK REQUIRED FILES")
print("=" * 70)


correct_adapter = (
    ROOT /
    "adapters" /
    "correct"
)


target_file = (
    ROOT /
    "data" /
    "eval_target.jsonl"
)


assert target_file.exists(), (
    f"Không tìm thấy {target_file}"
)


assert correct_adapter.exists(), (
    "Không tìm thấy adapters/correct/. "
    "Bạn phải chạy NB3 trước."
)


assert (
    correct_adapter /
    "adapter_config.json"
).exists(), (
    "Thiếu adapters/correct/adapter_config.json"
)


print(
    "✅ eval_target.jsonl"
)

print(
    "✅ adapters/correct"
)


# ============================================================
# 3. LOAD EVALUATION SET
# ============================================================

target = load_jsonl(
    target_file
)


# ------------------------------------------------------------
# EVAL_LIMIT=0 hoặc unset => FULL evaluation set
#
# Final bonus nên chạy FULL SET.
# ------------------------------------------------------------

EVAL_LIMIT = int(
    os.environ.get(
        "EVAL_LIMIT",
        "0"
    ) or 0
)


if EVAL_LIMIT:

    target = target[:EVAL_LIMIT]

    print(
        f"\n⚠ EVAL_LIMIT={EVAL_LIMIT}"
    )

    print(
        "⚠ Đây chỉ nên dùng để test nhanh."
    )

else:

    print(
        "\n✅ FULL EVALUATION SET"
    )


print(
    f"Scoring merge check on "
    f"{len(target)} target items"
)


assert len(target) > 0


# ============================================================
# 4. LOAD CORRECT ADAPTER
# ============================================================

print("\n" + "=" * 70)
print("4. LOAD BASE + CORRECT ADAPTER")
print("=" * 70)


from peft import PeftModel


model, tok = generate.load_base(
    TIER
)


model = PeftModel.from_pretrained(
    model,
    str(correct_adapter)
)


print(
    "✅ Correct adapter loaded"
)


# ============================================================
# 5. SCORE BEFORE MERGE
# ============================================================

print("\n" + "=" * 70)
print("5. SCORE BEFORE MERGE")
print("=" * 70)


target_prompts = [
    row["input"]
    for row in target
]


preds_before, latency_before = (
    generate.generate_batch(
        model,
        tok,
        target_prompts,
        system=generate.NAIVE_PROMPT,
        label="before_merge"
    )
)


before_scores = [

    ev.triage_field_accuracy(
        pred,
        row["label"]
    )

    for pred, row
    in zip(
        preds_before,
        target
    )
]


before = (
    sum(before_scores)
    /
    len(before_scores)
)


print(
    f"\nBefore merge target score:"
    f" {before:.4f}"
)

print(
    f"Latency:"
    f" {latency_before:.2f} ms"
)


# ============================================================
# 6. MERGE LoRA INTO BASE MODEL
# ============================================================

print("\n" + "=" * 70)
print("6. MERGE LoRA INTO BASE")
print("=" * 70)


merged = (
    model.merge_and_unload()
)


print(
    "✅ merge_and_unload() completed"
)


# ============================================================
# 7. SCORE AFTER MERGE
# ============================================================

print("\n" + "=" * 70)
print("7. SCORE AFTER MERGE")
print("=" * 70)


preds_after, latency_after = (
    generate.generate_batch(
        merged,
        tok,
        target_prompts,
        system=generate.NAIVE_PROMPT,
        label="after_merge"
    )
)


after_scores = [

    ev.triage_field_accuracy(
        pred,
        row["label"]
    )

    for pred, row
    in zip(
        preds_after,
        target
    )
]


after = (
    sum(after_scores)
    /
    len(after_scores)
)


delta = (
    after
    -
    before
)


print(
    f"\nBefore merge : {before:.4f}"
)

print(
    f"After merge  : {after:.4f}"
)

print(
    f"Delta        : {delta:+.4f}"
)

print(
    f"Before latency: "
    f"{latency_before:.2f} ms"
)

print(
    f"After latency : "
    f"{latency_after:.2f} ms"
)


# ============================================================
# 8. NO-REGRESSION ASSERT
# ============================================================

TOL = 0.01


print("\n" + "=" * 70)
print("8. MERGE REGRESSION CHECK")
print("=" * 70)


print(
    f"Tolerance:"
    f" {TOL}"
)


assert delta >= -TOL, (

    f"Điểm tụt {abs(delta):.4f} "
    f"sau merge. "

    f"Ngưỡng cho phép là {TOL}. "

    "Kiểm tra dtype khi merge. "
    "Nếu dùng DoRA cần PEFT đủ mới "
    "để merge magnitude đúng."
)


print(
    "✅ Score không tụt quá 0.01"
)


# ============================================================
# 9. SAVE MERGED MODEL
# ============================================================

print("\n" + "=" * 70)
print("9. SAVE MERGED MODEL")
print("=" * 70)


merged_dir = (
    ROOT /
    "adapters" /
    "merged"
)


merged_dir.mkdir(
    parents=True,
    exist_ok=True
)


merged.save_pretrained(
    merged_dir
)


tok.save_pretrained(
    merged_dir
)


print(
    "Merged model saved ->",
    merged_dir
)


# ============================================================
# 10. SAVE MERGE CHECK
# ============================================================

merge_result = {

    "before_merge":
        before,

    "after_merge":
        after,

    "delta":
        delta,

    "tolerance":
        TOL,

    "n":
        len(target),

    "eval_limit":
        EVAL_LIMIT or None,

    "full_eval":
        not bool(EVAL_LIMIT),

    "before_latency_ms":
        latency_before,

    "after_latency_ms":
        latency_after,
}


report.write_json(
    merge_result,
    "merge_check.json",
    results_dir=ROOT / "results"
)


merge_check_path = (
    ROOT /
    "results" /
    "merge_check.json"
)


assert merge_check_path.exists()


print(
    "\nSaved:",
    merge_check_path
)


print(
    json.dumps(
        merge_result,
        ensure_ascii=False,
        indent=2
    )
)


# ============================================================
# 11. FREE MERGED MODEL FROM VRAM
# ============================================================

del merged
del model


generate.free_memory()


# ============================================================
# 12. MULTI-ADAPTER HOT-SWAP
# ============================================================

print("\n" + "=" * 70)
print("12. MULTI-ADAPTER HOT-SWAP")
print("=" * 70)


# ------------------------------------------------------------
# Một base model duy nhất
# ------------------------------------------------------------

base_model, tok = generate.load_base(
    TIER
)


# ------------------------------------------------------------
# Nạp correct làm adapter đầu tiên
# ------------------------------------------------------------

model = PeftModel.from_pretrained(

    base_model,

    str(
        ROOT /
        "adapters" /
        "correct"
    ),

    adapter_name="correct"
)


available = [
    "correct"
]


# ------------------------------------------------------------
# Tìm thêm adapter từ NB4
# ------------------------------------------------------------

candidate_adapters = [

    "attn_only",

    "wrong_lr",

    "qlora",
]


for adapter_name in candidate_adapters:

    adapter_dir = (
        ROOT /
        "adapters" /
        adapter_name
    )

    adapter_config = (
        adapter_dir /
        "adapter_config.json"
    )

    if (
        adapter_dir.exists()
        and adapter_config.exists()
    ):

        try:

            model.load_adapter(
                str(adapter_dir),
                adapter_name=adapter_name
            )

            available.append(
                adapter_name
            )

            print(
                f"✅ Loaded adapter:"
                f" {adapter_name}"
            )

        except Exception as exc:

            print(
                f"⚠ Không load được "
                f"{adapter_name}:"
                f" {exc}"
            )


print(
    "\nAdapters đang được nạp:"
)

for name in available:

    print(
        " -",
        name
    )


# ============================================================
# BONUS REQUIREMENT:
# ít nhất 2 adapter cùng tồn tại trên một base.
# ============================================================

if len(available) < 2:

    print(
        "\n❌ CHƯA ĐỦ YÊU CẦU BONUS B1"
    )

    print(
        "Cần ít nhất 2 adapter."
    )

    print(
        "Hãy chạy NB4 để sinh "
        "attn_only hoặc qlora trước."
    )

else:

    print(
        f"\n✅ Có {len(available)} adapters"
        " trên cùng một base model"
    )


# ============================================================
# 13. TEST HOT-SWAP
# ============================================================

print("\n" + "=" * 70)
print("13. HOT-SWAP TEST")
print("=" * 70)


ticket = (
    target[0]["input"]
)


print(
    "\nTest ticket:"
)

print(
    ticket
)


hot_swap_results = {}


for name in available:

    print(
        "\n" + "-" * 70
    )

    print(
        f"Adapter: {name}"
    )


    # --------------------------------------------------------
    # Đây là thao tác hot-swap adapter.
    # Base model không được reload.
    # --------------------------------------------------------

    model.set_adapter(
        name
    )


    output, latency = (
        generate.generate_batch(
            model,
            tok,
            [ticket],
            system=generate.NAIVE_PROMPT,
            label=f"hot_swap/{name}"
        )
    )


    generated = output[0]


    hot_swap_results[name] = {

        "output":
            generated,

        "latency_ms":
            latency,
    }


    print(
        generated[:500]
    )

    print(
        f"\nLatency:"
        f" {latency:.2f} ms"
    )


# ============================================================
# 14. SAVE HOT-SWAP EVIDENCE
# ============================================================

hot_swap_result = {

    "base_model":
        TIER.model_id,

    "n_adapters":
        len(available),

    "adapters":
        available,

    "same_base_model":
        True,

    "hot_swap_pass":
        len(available) >= 2,

    "ticket":
        ticket,

    "results":
        hot_swap_results,
}


report.write_json(
    hot_swap_result,
    "hot_swap.json",
    results_dir=ROOT / "results"
)


print(
    "\nSaved:",
    ROOT /
    "results" /
    "hot_swap.json"
)


# ============================================================
# 15. FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("NB6 FINAL CHECK")
print("=" * 70)


checks = {

    "merge_check_json":
        (
            ROOT /
            "results" /
            "merge_check.json"
        ).exists(),

    "merged_model":
        (
            ROOT /
            "adapters" /
            "merged"
        ).exists(),

    "score_regression_ok":
        delta >= -TOL,

    "at_least_two_adapters":
        len(available) >= 2,

    "hot_swap_json":
        (
            ROOT /
            "results" /
            "hot_swap.json"
        ).exists(),
}


for key, value in checks.items():

    print(
        "✅" if value else "❌",
        key
    )


print()


if all(checks.values()):

    print(
        "🎉 NB6 / BONUS B1 HOÀN THÀNH"
    )

    print(
        "Bạn có bằng chứng:"
    )

    print(
        "1. Merge không làm target "
        "tụt quá 0.01"
    )

    print(
        "2. Có >=2 adapter"
        " trên cùng một base"
    )

    print(
        "3. Có hot-swap giữa các adapter"
    )

else:

    print(
        "⚠ NB6 chạy xong nhưng "
        "chưa đủ toàn bộ yêu cầu Bonus B1."
    )

    if len(available) < 2:

        print(
            "→ Chạy NB4 để tạo thêm "
            "attn_only / qlora."
        )