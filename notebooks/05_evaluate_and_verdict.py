# %% [markdown]
# # NB5 — Đánh giá bốn nhóm & PHÁN QUYẾT
#
# Đây là notebook cho điểm. Câu hỏi được chấm **không phải** "perplexity giảm bao nhiêu"
# mà là câu của deck §21:
#
# > **Bạn có chứng minh được bản fine-tune thắng baseline (b) — và bạn có phát hiện được
# > nếu nó KHÔNG thắng?**
#
# Bốn nhóm: **target · regression · format · latency**. Một run chỉ "đạt" khi vượt (b) ở
# target **và** không tụt general capability quá ngưỡng (deck §6.3).

# %%
import json
import os
import pathlib
import sys

sys.path.insert(
    0,
    str(pathlib.Path.cwd() / "src")
)

sys.path.insert(
    0,
    str(pathlib.Path.cwd().parent / "src")
)

from labkit import (
    evaluate as ev,
    generate,
    report
)

from labkit.config import get_tier


ROOT = (
    pathlib.Path.cwd()
    if (pathlib.Path.cwd() / "data").exists()
    else pathlib.Path.cwd().parent
)


TIER = get_tier(
    os.environ.get(
        "COMPUTE_TIER",
        "T4"
    )
)


def load_jsonl(p):

    return [
        json.loads(l)

        for l in open(
            p,
            encoding="utf-8"
        )

        if l.strip()
    ]


# ============================================================
# 1. LOAD EVAL DATA
# ============================================================

target = load_jsonl(
    ROOT /
    "data" /
    "eval_target.jsonl"
)


regression = load_jsonl(
    ROOT /
    "data" /
    "eval_regression.jsonl"
)


# ============================================================
# Phải giống slice dùng ở NB2.
#
# Final submission:
# EVAL_LIMIT phải unset / 0.
# ============================================================

EVAL_LIMIT = int(
    os.environ.get(
        "EVAL_LIMIT",
        "0"
    )
)


if EVAL_LIMIT:

    target = target[
        :EVAL_LIMIT
    ]

    regression = regression[
        :EVAL_LIMIT
    ]


# ============================================================
# 2. LOAD FROZEN BASELINES
# ============================================================

frozen = json.loads(

    (
        ROOT /
        "results" /
        "baselines_frozen.json"
    ).read_text(
        encoding="utf-8"
    )

)


base_b = ev.GroupScores(
    **{
        k: v

        for k, v
        in frozen[
            "baseline_b"
        ].items()

        if k != "extra"
    }
)


base_a = ev.GroupScores(
    **{
        k: v

        for k, v
        in frozen[
            "baseline_a"
        ].items()

        if k != "extra"
    }
)


# ============================================================
# Safety check:
# NB2 và NB5 phải dùng cùng số sample.
# ============================================================

if (
    frozen.get(
        "n_target"
    )
    != len(target)
):

    raise SystemExit(

        "eval slice mismatch: "
        f"baselines were frozen on "
        f"{frozen.get('n_target')} "
        f"target items, "

        f"this run has "
        f"{len(target)}. "

        "Set EVAL_LIMIT to the same "
        "value as NB2 "
        "(or unset both)."
    )


print(
    "baseline (b) target =",
    round(
        base_b.target,
        3
    ),
    "— đây là mốc phải vượt"
)


# %% [markdown]
# ## 1. Chấm một adapter

# %%
from peft import PeftModel


def score_adapter(
    adapter_dir: pathlib.Path,
    system_prompt: str | None,
    *,
    load_in_4bit: bool = False,
    with_regression: bool = True,
    label: str = "ft"
) -> tuple:

    """
    Score one adapter on:

        target
        format
        latency
        regression (optional)

    load_in_4bit phải giống cách adapter được train.

    QLoRA được học trên base 4-bit, nên khi evaluate
    cũng phải dùng 4-bit base.
    """

    # ========================================================
    # Load base
    # ========================================================

    model, tok = generate.load_base(
        TIER,
        load_in_4bit=load_in_4bit
    )


    # ========================================================
    # Load adapter
    # ========================================================

    model = PeftModel.from_pretrained(
        model,
        str(adapter_dir)
    )


    model.eval()


    # ========================================================
    # TARGET
    # ========================================================

    preds, lat = (
        generate.generate_batch(

            model,

            tok,

            [
                r["input"]
                for r in target
            ],

            system=system_prompt,

            label=f"{label}/target"
        )
    )


    tgt = sum(

        ev.triage_field_accuracy(
            p,
            r["label"]
        )

        for p, r
        in zip(
            preds,
            target
        )

    ) / len(target)


    # ========================================================
    # FORMAT
    # ========================================================

    fmt = sum(

        ev.has_required_keys(
            p,
            ev.TRIAGE_KEYS
        )

        for p in preds

    ) / len(preds)


    # ========================================================
    # REGRESSION
    # ========================================================

    rpreds = []

    reg = 0.0


    if with_regression:

        rpreds, _ = (
            generate.generate_batch(

                model,

                tok,

                [
                    r["instruction"]
                    for r in regression
                ],

                system=None,

                max_new_tokens=96,

                label=f"{label}/regression"
            )
        )


        reg = sum(

            ev.keyword_recall(
                p,
                r["keywords"]
            )

            for p, r
            in zip(
                rpreds,
                regression
            )

        ) / len(regression)


    # ========================================================
    # Reasoning trace
    # ========================================================

    trace = sum(

        ev.valid_reasoning_trace(
            p
        )

        for p in preds

    ) / len(preds)


    # ========================================================
    # FREE VRAM
    # ========================================================

    del model

    generate.free_memory()


    # ========================================================
    # GROUP SCORES
    # ========================================================

    scores = ev.GroupScores(

        target=tgt,

        regression=reg,

        format=fmt,

        latency_ms=lat,

        n=len(target),

        extra={
            "valid_trace_rate":
                round(
                    trace,
                    4
                )
        }
    )


    return (
        scores,
        preds,
        rpreds
    )


# ============================================================
# 3. SCORE CORRECT ADAPTER
#
# Fine-tune chỉ dùng NAIVE_PROMPT.
#
# Ý nghĩa:
# behavior đã được chuyển vào weights,
# nên không cần optimized prompt dài.
# ============================================================

scores_ft, preds_ft, rpreds_ft = (
    score_adapter(

        ROOT /
        "adapters" /
        "correct",

        generate.NAIVE_PROMPT
    )
)


print(
    "fine-tune:",
    scores_ft.as_dict()
)


# %% [markdown]
# ## 2. Bảng so sánh ba baseline

# ============================================================
# 4. THREE-WAY COMPARISON
# ============================================================

table = ev.comparison_table(
    {
        "(a) base + naive prompt":
            base_a,

        "(b) base + optimized prompt":
            base_b,

        "(c) LoRA fine-tune":
            scores_ft,
    }
)


print(
    report.markdown_table(
        table
    )
)


# %% [markdown]
# ## 3. Cổng hồi quy — phán quyết

# ============================================================
# 5. REGRESSION GATE / VERDICT
# ============================================================

verdict = ev.regression_gate(
    scores_ft,
    base_b
)


print(
    "PASSED"
    if verdict.passed
    else "FAILED"
)


for reason in verdict.reasons:

    print(
        " -",
        reason
    )


# ============================================================
# SAVE VERDICT
# ============================================================

report.write_json(

    {
        "comparison":
            table,

        "verdict":
            verdict.as_dict(),

        "valid_trace_rate":
            scores_ft.extra.get(
                "valid_trace_rate"
            ),
    },

    "verdict.json",

    results_dir=ROOT / "results"
)


# %% [markdown]
# ### Nếu FAILED — đừng sửa eval
#
# Một phán quyết FAILED **được chấm điểm đầy đủ** nếu bạn phân tích đúng. Deck §1: đôi
# khi kết luận đúng là *"bài toán này không cần fine-tune"*. Cái bị trừ điểm là:
# nới ngưỡng, làm yếu prompt (b), hay đổi tập eval sau khi thấy kết quả.
#
# Thứ tự chẩn đoán:
# 1. `format` thấp → template/mask (NB1), không phải LoRA
# 2. `regression` tụt → quên thảm hoạ → thêm 1–5% replay (deck §6.3)
# 3. `target` không nhúc nhích → xem lại LR (NB4 `wrong_lr`) trước khi đụng tới rank
# 4. Cả ba đều ổn nhưng vẫn thua (b) → prompt engineering đã thắng. Đó là một kết quả.


# %% [markdown]
# ## 4. Giải phẫu NB4 — chấm ba cấu hình sai trên CÙNG thang đo
#
# NB4 in ra `final_loss`. Đó là **loss huấn luyện**.
#
# Không được xếp hạng bằng final_loss.
#
# Phải xếp hạng bằng target score ở đây.


# ============================================================
# 6. AUTOPSY ALL CONTRASTS
# ============================================================

from labkit.config import (
    CONTRAST_KEYS,
    SPECS
)


autopsy = [
    {
        "run":
            "correct",

        "target":
            round(
                scores_ft.target,
                4
            ),

        "format":
            round(
                scores_ft.format,
                4
            ),

        "latency_ms":
            round(
                scores_ft.latency_ms,
                1
            ),

        "n":
            scores_ft.n,
    }
]


for key in CONTRAST_KEYS:

    adir = (
        ROOT /
        "adapters" /
        key
    )


    if not adir.exists():

        print(
            f"skip {key}: "
            f"{adir} chưa có "
            f"— chạy NB4 trước"
        )

        continue


    s_k, _, _ = score_adapter(

        adir,

        generate.NAIVE_PROMPT,

        load_in_4bit=
            SPECS[key].load_in_4bit,

        with_regression=False,

        label=key
    )


    autopsy.append(
        {
            "run":
                key,

            "target":
                round(
                    s_k.target,
                    4
                ),

            "format":
                round(
                    s_k.format,
                    4
                ),

            "latency_ms":
                round(
                    s_k.latency_ms,
                    1
                ),

            "n":
                s_k.n,
        }
    )


    print(
        f"{key}: "
        f"target={s_k.target:.3f}  "
        f"format={s_k.format:.3f}"
    )


print()


print(
    report.markdown_table(

        autopsy,

        [
            "run",
            "target",
            "format",
            "latency_ms",
            "n"
        ]
    )
)


# ============================================================
# SAVE AUTOPSY
# ============================================================

report.write_json(

    autopsy,

    "autopsy.json",

    results_dir=
        ROOT /
        "results"
)


# %% [markdown]
# ### Bảng này mới là câu trả lời cho ba câu hỏi ở cuối NB4
#
# Đặt nó cạnh cột `final_loss` của NB4.
#
# Nếu thứ tự hai bảng khác nhau,
# bạn vừa đo được lý do lab cũ kết luận sai:
# nó dừng lại ở chỉ số thay thế.


# %% [markdown]
# ## 5. Định tính — bắt buộc có cả ca THUA
#
# Chọn 5 ví dụ:
#
# ≥2 ca fine-tune thắng
#
# ≥2 ca fine-tune thua
#
# Chỉ chọn ca thắng là cherry-pick.


# ============================================================
# 7. QUALITATIVE CASES
# ============================================================

rows = []


for i, (prediction, row) in enumerate(
    zip(
        preds_ft,
        target
    )
):

    s_ft = (
        ev.triage_field_accuracy(
            prediction,
            row["label"]
        )
    )


    rows.append(
        {
            "i":
                i,

            "ticket":
                row["input"][:70],

            "ft_score":
                round(
                    s_ft,
                    2
                ),

            "ft_pred":
                prediction
                .replace(
                    "\n",
                    " "
                )[:90]
        }
    )


# ============================================================
# Sort từ tệ nhất → tốt nhất
# ============================================================

rows.sort(
    key=lambda x:
        x["ft_score"]
)


print(
    "--- 3 ca TỆ NHẤT "
    "(bắt buộc đưa vào report) ---"
)


print(
    report.markdown_table(

        rows[:3],

        [
            "i",
            "ticket",
            "ft_score",
            "ft_pred"
        ]
    )
)


print(
    "\n--- 3 ca TỐT NHẤT ---"
)


print(
    report.markdown_table(

        rows[-3:],

        [
            "i",
            "ticket",
            "ft_score",
            "ft_pred"
        ]
    )
)


# ============================================================
# SAVE QUALITATIVE
# ============================================================

report.write_json(

    rows,

    "qualitative.json",

    results_dir=
        ROOT /
        "results"
)


# ============================================================
# 8. FINAL CHECK
# ============================================================

print(
    "\n" +
    "=" * 70
)

print(
    "NB5 FINAL CHECK"
)

print(
    "=" * 70
)


required = [

    ROOT /
    "results" /
    "verdict.json",

    ROOT /
    "results" /
    "autopsy.json",

    ROOT /
    "results" /
    "qualitative.json",
]


for path in required:

    print(
        "✅"
        if path.exists()
        else "❌",
        path
    )


assert all(
    path.exists()
    for path in required
)


print(
    "\n✅ NB5 HOÀN THÀNH"
)

print(
    "Verdict:",
    "PASSED"
    if verdict.passed
    else "FAILED"
)

print(
    f"Baseline B target : "
    f"{base_b.target:.4f}"
)

print(
    f"Fine-tune target  : "
    f"{scores_ft.target:.4f}"
)

print(
    f"Fine-tune format  : "
    f"{scores_ft.format:.4f}"
)

print(
    f"Fine-tune regression: "
    f"{scores_ft.regression:.4f}"
)

print(
    f"Fine-tune latency : "
    f"{scores_ft.latency_ms:.1f} ms"
)