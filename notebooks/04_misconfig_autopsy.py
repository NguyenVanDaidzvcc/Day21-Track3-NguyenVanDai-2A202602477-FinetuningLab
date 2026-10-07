# %% [markdown]
# # NB4 — Giải phẫu cấu hình sai (phần quan trọng nhất của lab)
#
# Lab Day 21 **phiên bản cũ** lấy "quét rank r=8/16/64" làm thí nghiệm trung tâm, gắn
# LoRA vào `q_proj, v_proj`, và chấm bằng perplexity. Deck hiện tại gọi đúng ba thứ đó
# là **Lỗi #1, #2, #3** (§11.2–§11.4).
#
# Notebook này không xoá thí nghiệm cũ — nó **chạy lại thí nghiệm cũ như một đối chứng**,
# để bạn tự tay thấy danh tiếng *"LoRA học kém hơn full fine-tune"* xuất hiện rồi biến mất.
#
# Ba run đối chứng, **cùng số step**, chỉ đổi một biến mỗi lần:
#
# | Run | Đổi gì | Kỳ vọng |
# |---|---|---|
# | `attn_only` | chỉ q,v — **rank nâng lên cho bằng số tham số** | thua `correct` |
# | `wrong_lr` | LR thang full-FT (÷10) | loss gần như phẳng |
# | `qlora` | 4-bit thay bf16 | nhẹ hơn, chất lượng ? |
#
# > **Vì sao phải "bằng số tham số".** So `q,v @ r=16` với `all-linear @ r=16` là so
# > *ngân sách*, không phải so *vị trí* — và không chứng minh được gì. `matched_rank()`
# > giải ra rank đưa attention-only về đúng ngân sách của `correct`.

# %%
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path.cwd() / "src"))
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))

from labkit import data, generate, modeling, report, train
from labkit.config import CONTRAST_KEYS, SPECS, get_tier, training_epochs

ROOT = (
    pathlib.Path.cwd()
    if (pathlib.Path.cwd() / "data").exists()
    else pathlib.Path.cwd().parent
)

TIER = get_tier(
    os.environ.get("COMPUTE_TIER", "T4")
)

from datasets import Dataset


def load_jsonl(p):
    return [
        json.loads(l)
        for l in open(p, encoding="utf-8")
        if l.strip()
    ]


train_rows = load_jsonl(
    ROOT / "data" / "split" / "train.jsonl"
)

# Same pre-tokenized, NB1-verified mask as NB3 — the contrasts must differ only in the
# variable under test, and that includes using an identical loss mask.
_tok_for_data = None
train_ds = None


# %% [markdown]
# ## 1. Bảng vị trí × rank × số tham số
#
# Đọc bảng này **trước** khi chạy. Nó cho thấy vì sao so cùng-rank là không công bằng.

# %%
model, tok = generate.load_base(TIER)

placement = modeling.describe_placement(
    model,
    SPECS["correct"].r
)

print(
    report.markdown_table(
        placement
    )
)

del model

generate.free_memory()


# %% [markdown]
# ## 2. Ba run đối chứng — cùng ngân sách step

# %%
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def run_contrast(key: str) -> dict:

    spec = SPECS[key]

    global train_ds

    # ========================================================
    # Load base model
    # ========================================================

    model, tok = generate.load_base(
        TIER,
        load_in_4bit=spec.load_in_4bit
    )

    # ========================================================
    # Build dataset một lần
    # ========================================================

    if train_ds is None:

        train_ds = Dataset.from_list(
            data.to_training_dataset(
                tok,
                train_rows,
                max_length=TIER.max_length,
                mask_mode=os.environ.get(
                    "MASK_MODE",
                    "assistant-only"
                )
            )
        )

        print(
            "  train_ds:",
            train_ds
        )

    # ========================================================
    # Resolve LoRA target modules
    # ========================================================

    targets = modeling.resolve_target_modules(
        model,
        spec.target
    )

    # ========================================================
    # attn_only:
    # tự giải rank để số params gần bằng correct
    # ========================================================

    if spec.r is None:

        base_targets = (
            modeling.resolve_target_modules(
                model,
                "text-linear"
            )
        )

        r = modeling.matched_rank(
            model,
            base_targets,
            SPECS["correct"].r,
            targets
        )

        spec = spec.resolved(r)

        print(
            f"  matched rank for {key}: "
            f"r={r} "
            f"(alpha={spec.alpha})"
        )

    # ========================================================
    # Count trainable params
    # ========================================================

    trainable = (
        modeling.count_lora_params(
            model,
            targets,
            spec.r
        )
    )

    # ========================================================
    # SAME STEP BUDGET AS NB3
    # ========================================================

    max_steps = train.planned_steps(
        len(train_ds),
        TIER,
        training_epochs()
    )

    print(
        f"  max_steps={max_steps}"
    )

    print(
        f"  trainable_params={trainable:,}"
    )

    # ========================================================
    # SFT config
    # ========================================================

    want = train.sft_config_kwargs(
        TIER,
        spec,
        str(ROOT / "adapters" / key),
        max_steps=max_steps
    )

    sft_kwargs, _ = train.filter_kwargs(
        SFTConfig,
        want,
        label=f"SFTConfig[{key}]"
    )

    # ========================================================
    # LoRA config
    # ========================================================

    lora_kwargs, _ = train.filter_kwargs(
        LoraConfig,
        train.lora_config_kwargs(
            spec,
            targets
        ),
        label=f"LoraConfig[{key}]"
    )

    # ========================================================
    # Trainer
    # ========================================================

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            **sft_kwargs
        ),
        train_dataset=train_ds,
        processing_class=tok,
        peft_config=LoraConfig(
            **lora_kwargs
        )
    )

    # ========================================================
    # Precision fix
    #
    # qlora trên T4:
    # bf16 trainable weights + fp16 GradScaler
    # có thể lỗi nếu không recast
    # ========================================================

    fix = train.align_trainable_precision(
        trainer.model
    )

    if fix.get("recast"):

        print(
            f"  precision fix: "
            f"recast {fix['recast']}/"
            f"{fix['trainable_tensors']} "
            f"trainable tensors "
            f"bf16 -> fp32 "
            f"for the fp16 GradScaler"
        )

    # ========================================================
    # TRAIN
    # ========================================================

    t0 = time.perf_counter()

    res = trainer.train()

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    # ========================================================
    # SAVE ADAPTER
    # ========================================================

    out = (
        ROOT /
        "adapters" /
        key
    )

    trainer.model.save_pretrained(
        out
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    row = train.summarize_run(
        spec,
        TIER,
        targets,
        trainable,
        elapsed,
        generate.peak_vram_gb()
    )

    row["final_loss"] = round(
        res.training_loss,
        4
    )

    row["max_steps"] = (
        max_steps
    )

    row["teaches"] = (
        spec.teaches
    )

    report.append_row(
        row,
        results_dir=ROOT / "results"
    )

    # ========================================================
    # FREE VRAM
    # ========================================================

    del trainer
    del model

    generate.free_memory()

    return row


# ============================================================
# Resume support
#
# NB4 là notebook lâu nhất.
#
# Có adapter rồi => skip.
#
# FORCE_RETRAIN=1
# => train lại tất cả.
#
# ONLY=attn_only
# => chỉ chạy một contrast.
# ============================================================

FORCE_RETRAIN = (
    os.environ
    .get("FORCE_RETRAIN", "")
    .lower()
    in {
        "1",
        "true",
        "yes"
    }
)


ONLY = [
    k
    for k
    in os.environ.get(
        "ONLY",
        ""
    ).split(",")
    if k.strip()
]


rows = []


for key in (
    ONLY or CONTRAST_KEYS
):

    if key not in SPECS:

        raise SystemExit(
            f"unknown run {key!r}; "
            f"expected some of "
            f"{CONTRAST_KEYS}"
        )

    done = (
        ROOT /
        "adapters" /
        key /
        "adapter_model.safetensors"
    ).exists()

    if done and not FORCE_RETRAIN:

        print(
            f"skip {key}: "
            f"adapters/{key}/ "
            f"already trained "
            f"(FORCE_RETRAIN=1 to redo, "
            f"or delete the directory)"
        )

        continue

    print(
        "=" * 70
    )

    print(
        f"RUN {key}: "
        f"{SPECS[key].label}"
    )

    print(
        f"     "
        f"{SPECS[key].teaches}"
    )

    rows.append(
        run_contrast(key)
    )


# ============================================================
# Đọc runs.csv để bảng luôn có cả correct + contrasts
# ============================================================

if not rows:

    print(
        "nothing retrained this session "
        "— reading runs.csv "
        "for the table"
    )


_seen = {}


for r in report.read_rows(
    "runs.csv",
    results_dir=ROOT / "results"
):

    if r.get("run"):

        _seen[
            r["run"]
        ] = r


rows = [

    _seen[k]

    for k in [
        "correct",
        *CONTRAST_KEYS
    ]

    if k in _seen

] or rows


# %% [markdown]
# ## 3. Bảng đối chứng
#
# Cả bốn run — `correct` ở NB3 và ba run ở đây — chạy **cùng một số optimizer step**
# (`train.planned_steps(...)`, xem `labkit.config.training_epochs`). Nên loss so được
# trực tiếp: khác biệt duy nhất giữa mỗi contrast và `correct` là đúng một biến.
#
# > Trước đây NB4 cố định `max_steps=60` trong khi NB3 chạy 30 step, và phần này bảo bạn
# > tự chạy lại `correct` cho công bằng. Đó là bug: contrast được huấn luyện gấp đôi
# > baseline mà nó bị đem ra so.

# %%

cols = [
    "run",
    "label",
    "r",
    "trainable_params",
    "learning_rate",
    "final_loss",
    "train_seconds",
    "peak_vram_gb"
]


print(
    report.markdown_table(
        rows,
        cols
    )
)


# %% [markdown]
# ### ⚠ `final_loss` ở đây là LOSS HUẤN LUYỆN — đừng xếp hạng bằng nó
#
# Cột này rẻ (có sẵn từ lúc train) và **không phải** thang đo dùng để kết luận. Chính
# lab này gọi "chấm bằng chỉ số thay thế thay vì bằng năng lực trên tác vụ" là **Lỗi
# #3** — nếu bạn xếp hạng bốn run bằng `final_loss`, bạn đang mắc đúng lỗi đó.
#
# Với 225 mẫu và 30 step, một adapter r=283 có thể ép loss huấn luyện **thấp hơn**
# `correct` mà vẫn tệ hơn trên tập target. Loss thấp có thể chỉ là ghi nhớ.
#
# > **NB5 §4 chấm cả ba adapter này trên tập target** — cùng thang đo đã dùng cho
# > `correct`. Đó mới là bảng để trả lời ba câu dưới đây. Nếu thứ tự của hai bảng khác
# > nhau, hãy nói thẳng điều đó trong REPORT.md: bạn vừa đo được lý do lab cũ kết luận sai.
#
# **Về `grad_norm: nan` ở dòng log đầu tiên:** đó là `GradScaler` của fp16 đang dò thang
# — vài step đầu tràn số và bị bỏ qua, đúng theo thiết kế. Bình thường. Cái *không*
# bình thường là `nan` kéo dài suốt run: khi đó run đã chết và loss cuối vô nghĩa.


# %% [markdown]
# ## 4. Câu hỏi phải trả lời trong REPORT.md
#
# Trả lời bằng bảng **NB5 §4** (điểm target), rồi đối chiếu với `final_loss` ở trên.
#
# 1. `attn_only` có **cùng số tham số huấn luyện** với `correct`. Trên tập target nó
#    thắng, thua, hay hoà? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?
# 2. `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau bao nhiêu? Nếu chỉ nhìn
#    loss mà không biết LR, bạn sẽ kết luận gì — và kết luận đó có đúng không?
# 3. `qlora` tiết kiệm bao nhiêu VRAM, và **trả giá bằng gì**? Nhà cung cấp khuyến nghị
#    *không* dùng QLoRA cho dòng model này (deck §13) — số đo của bạn có ủng hộ điều đó không?