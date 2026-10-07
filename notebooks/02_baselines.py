# ============================================================
# NB2 — FREEZE EVAL + THREE BASELINES
# ============================================================

import hashlib
import json
import os
import pathlib
import sys


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
print("NB2 — BASELINES")
print("=" * 70)

print("ROOT :", ROOT)
print("Tier :", TIER.name)
print("Model:", TIER.model_id)


# ============================================================
# 1. LOAD EVALUATION DATA
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


target_path = (
    ROOT /
    "data" /
    "eval_target.jsonl"
)

regression_path = (
    ROOT /
    "data" /
    "eval_regression.jsonl"
)


assert target_path.exists(), (
    f"Không có {target_path}"
)

assert regression_path.exists(), (
    f"Không có {regression_path}"
)


target = load_jsonl(
    target_path
)

regression = load_jsonl(
    regression_path
)


# ============================================================
# IMPORTANT:
# FINAL SUBMISSION MUST USE FULL EVAL SET
# ============================================================

EVAL_LIMIT = int(
    os.environ.get(
        "EVAL_LIMIT",
        "0"
    )
)


if EVAL_LIMIT:

    target = target[:EVAL_LIMIT]

    regression = regression[:EVAL_LIMIT]

    print(
        f"\n⚠ EVAL_LIMIT={EVAL_LIMIT}"
    )

    print(
        "⚠ Đây chỉ là SMOKE MODE."
    )

    print(
        "⚠ Không được dùng kết quả này để nộp."
    )


print("\nEvaluation data:")

print(
    f"target     = {len(target)}"
)

print(
    f"regression = {len(regression)}"
)


assert len(target) > 0
assert len(regression) > 0


# ============================================================
# 2. LOAD BASE MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOAD BASE MODEL")
print("=" * 70)


model, tok = generate.load_base(
    TIER
)


generate.free_memory()


# ============================================================
# 3. SCORE FUNCTION
# ============================================================

def score_run(
    model,
    tok,
    system_prompt,
    label
):

    print(
        "\n" +
        "=" * 70
    )

    print(
        "RUN:",
        label
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # TARGET SET
    # --------------------------------------------------------

    prompts = [
        row["input"]
        for row in target
    ]


    preds, latency = (
        generate.generate_batch(
            model,
            tok,
            prompts,
            system=system_prompt,
            label=f"{label}/target"
        )
    )


    target_scores = []

    format_scores = []


    for pred, row in zip(
        preds,
        target
    ):

        target_scores.append(

            ev.triage_field_accuracy(
                pred,
                row["label"]
            )

        )

        format_scores.append(

            ev.has_required_keys(
                pred,
                ev.TRIAGE_KEYS
            )

        )


    tgt = (
        sum(target_scores)
        /
        len(target_scores)
    )


    fmt = (
        sum(format_scores)
        /
        len(format_scores)
    )


    # --------------------------------------------------------
    # REGRESSION SET
    # --------------------------------------------------------

    regression_prompts = [
        row["instruction"]
        for row in regression
    ]


    regression_preds, _ = (
        generate.generate_batch(
            model,
            tok,
            regression_prompts,
            system=None,
            max_new_tokens=96,
            label=f"{label}/regression"
        )
    )


    regression_scores = []


    for pred, row in zip(
        regression_preds,
        regression
    ):

        regression_scores.append(

            ev.keyword_recall(
                pred,
                row["keywords"]
            )

        )


    reg = (
        sum(regression_scores)
        /
        len(regression_scores)
    )


    # --------------------------------------------------------
    # GROUP SCORE
    # --------------------------------------------------------

    scores = ev.GroupScores(

        target=tgt,

        regression=reg,

        format=fmt,

        latency_ms=latency,

        n=len(target)
    )


    print(
        f"\n{label}"
    )

    print(
        f"target     = {tgt:.4f}"
    )

    print(
        f"regression = {reg:.4f}"
    )

    print(
        f"format     = {fmt:.4f}"
    )

    print(
        f"latency    = {latency:.2f} ms"
    )


    return (
        scores,
        preds,
        regression_preds
    )


# ============================================================
# 4. BASELINE A
# Base model + naive prompt
# ============================================================

scores_a, preds_a, regression_a = (
    score_run(
        model,
        tok,
        generate.NAIVE_PROMPT,
        "(a) base + naive prompt"
    )
)


# ============================================================
# 5. BASELINE B
# Base model + optimized prompt
# ============================================================

scores_b, preds_b, regression_b = (
    score_run(
        model,
        tok,
        generate.OPTIMIZED_PROMPT,
        "(b) base + optimized prompt"
    )
)


# ============================================================
# 6. FREEZE BASELINE
# ============================================================

print("\n" + "=" * 70)
print("FREEZE BASELINES")
print("=" * 70)


prompt_sha = hashlib.sha256(
    generate.OPTIMIZED_PROMPT.encode()
).hexdigest()[:16]


frozen = {

    "tier":
        TIER.name,

    "model":
        TIER.model_id,

    "baseline_a":
        scores_a.as_dict(),

    "baseline_b":
        scores_b.as_dict(),

    "optimized_prompt_sha":
        prompt_sha,

    "n_target":
        len(target),

    "n_regression":
        len(regression),

    "eval_limit":
        EVAL_LIMIT or None,

    "smoke_mode":
        bool(EVAL_LIMIT),
}


report.write_json(
    frozen,
    "baselines_frozen.json",
    results_dir=ROOT / "results"
)


print(
    json.dumps(
        frozen,
        ensure_ascii=False,
        indent=2
    )
)


# ============================================================
# 7. CHECK FILE
# ============================================================

output_file = (
    ROOT /
    "results" /
    "baselines_frozen.json"
)


assert output_file.exists(), (
    "Không tạo được baselines_frozen.json"
)


print(
    "\nSaved:",
    output_file
)


# ------------------------------------------------------------
# Submission warning
# ------------------------------------------------------------

if EVAL_LIMIT:

    print(
        "\n❌ NB2 đang chạy SMOKE MODE."
    )

    print(
        "Hãy bỏ EVAL_LIMIT và chạy lại NB2"
        " trước khi nộp."
    )

else:

    print(
        "\n✅ FULL EVAL SET"
    )

    print(
        "✅ NB2 HOÀN THÀNH"
    )


print("\nBaseline comparison:")

print(
    "A target:",
    scores_a.target
)

print(
    "B target:",
    scores_b.target
)

print(
    "\nTừ thời điểm này KHÔNG sửa:"
)

print(
    "- data/eval_target.jsonl"
)

print(
    "- data/eval_regression.jsonl"
)

print(
    "- generate.OPTIMIZED_PROMPT"
)