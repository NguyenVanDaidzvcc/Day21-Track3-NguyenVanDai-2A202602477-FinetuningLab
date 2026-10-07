# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Văn Đại  
**MSSV**: 2A202602477  
**Ngày**: 07/10/2026  
**Tier**: `T4`  
**Base model**: `unsloth/Qwen3.5-4B`  
**GPU thực tế**: `Tesla T4 — 14.6 GB VRAM`

---

## 1. Setup

Tôi sử dụng bộ dữ liệu mặc định gồm 250 ticket chăm sóc khách hàng bằng tiếng Việt,
với đầu ra là JSON triage gồm bốn trường `intent`, `urgency`, `product` và `sentiment`.
Mô hình `unsloth/Qwen3.5-4B` được lựa chọn vì phù hợp với giới hạn bộ nhớ của GPU T4
và có khả năng xử lý dạng hội thoại/instruction.

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage |
| Train / val | `225 / 25` (seed 42) |
| `max_length` | `1024` — p95 đo được là `98` *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | `2 / 30` |

Thống kê độ dài cho thấy mean = `93.1`, p50 = `93`, p95 = `98`, p99 = `100`,
max = `101` và `suggested_max_length = 256`. Tôi giữ `max_length=1024` theo cấu hình
tier T4 của lab để không thay đổi cấu hình nền giữa các run. Với tập dữ liệu hiện tại,
toàn bộ mẫu đều ngắn hơn nhiều so với giới hạn này nên không xảy ra truncation.

**Template có giữ khối `<think>` không?** **Có** — *(results/template_check.json)*.

Qwen3.5 vẫn giữ cấu trúc thinking của chat template. Vì dữ liệu huấn luyện được cung cấp
dưới dạng JSON ngắn và sử dụng `assistant-only`, tôi không cần thay đổi template hay
chuyển sang chế độ mask khác.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Supervised tokens | `39 / 94` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn đầu của phần thực sự được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "palo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Kết quả cho thấy 39/94 token của mẫu kiểm tra được supervised, tương ứng
`supervised_fraction = 0.4149`. Với `MASK_MODE=assistant-only`, phần câu hỏi và
ngữ cảnh đầu vào không tham gia tính loss, trong khi phần phản hồi của assistant
được giữ lại. Hai assert `answer_is_supervised=true` và `question_is_masked=true`
xác nhận loss mask hoạt động đúng. Đây là kiểm tra quan trọng vì nếu tính loss cả
prompt, model có thể học cách lặp lại câu hỏi thay vì chỉ học hành vi đầu ra mong muốn.

---

## 3. Ba baseline và fine-tune

Hai baseline (a) và (b) được đo và đóng băng trước khi huấn luyện. Fine-tune (c)
được đưa vào bảng sau để so sánh cuối cùng trên cùng tập đánh giá.

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | `0.000` | `0.7911` | `0.000` | `3245.1` |
| (b) base + optimized prompt | `0.765` | `0.7911` | `1.000` | `1024.8` |
| (c) LoRA fine-tune | `0.970` | `0.4778` | `1.000` | `1467.8` |

**Baseline (b) có thật sự mạnh hơn (a) không?** **Có.**

Baseline (a) đạt target `0.000` và format `0.000`, trong khi optimized prompt ở
baseline (b) đưa target lên `0.765` và format lên `1.000`. Regression của hai
baseline đều là `0.7911`, nghĩa là cải thiện prompt đã tăng mạnh năng lực trên tác vụ
mục tiêu mà chưa làm suy giảm tập regression.

Tôi **không sửa `OPTIMIZED_PROMPT` sau khi baseline được đóng băng**. Đây là mốc so sánh
cố định cho các thí nghiệm fine-tune sau đó; làm yếu baseline sau khi thấy kết quả
fine-tune sẽ làm mất tính công bằng của thí nghiệm.

---

## 4. Giải phẫu cấu hình sai (NB4)

Tất cả bốn run đều dùng cùng ngân sách `30 optimizer steps`. Mỗi contrast chỉ thay đổi
một yếu tố chính so với cấu hình `correct`.

| Run | vị trí | r | trainable | LR | train loss | **target** | s | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | `32,464,896` | `1e-4` | `0.6269` | **`0.970`** | `398.2` | `8.78` |
| `attn_only` | q,v | 283 | `32,456,704` | `1e-4` | `0.5374` | **`0.970`** | `270.4` | `8.79` |
| `wrong_lr` | text-linear | 16 | `32,464,896` | `1e-5` | `1.5702` | **`0.000`** | `393.6` | `8.78` |
| `qlora` | text-linear | 16 | `32,464,896` | `1e-4` | `0.7058` | **`0.940`** | `474.1` | `3.86` |

### 4.1 — Attention-only: rank và vị trí adapter

`attn_only` được tăng rank lên `r=283` để số trainable parameters gần bằng cấu hình
`correct`: 32,456,704 so với 32,464,896 tham số. Trên tập target, hai cấu hình **hòa
nhau ở 0.970**. Tuy nhiên `attn_only` lại có train loss thấp hơn (`0.5374` so với
`0.6269`), nhưng lợi thế đó không chuyển thành target score cao hơn.

Kết quả này cho thấy không thể xếp hạng cấu hình chỉ dựa trên training loss. Trong
thí nghiệm này, tăng rank để bù cho việc chỉ gắn adapter vào attention không tạo ra
lợi ích target so với `text-linear`. Bằng chứng cũng cho thấy rank và vị trí gắn
adapter phải được đánh giá với ngân sách tham số ngang nhau; nếu giữ cùng rank thì
phép so sánh sẽ không công bằng.

### 4.2 — Wrong learning rate

Run `wrong_lr` chỉ giảm learning rate từ `1e-4` xuống `1e-5`, trong khi placement,
rank, số trainable parameters và step budget được giữ nguyên. Final train loss của
nó là `1.5702`, cao hơn rõ rệt so với `0.6269` của `correct`, và target score cuối
cùng rơi xuống `0.000`.

Điều này cho thấy learning rate là một đòn bẩy rất mạnh trong ngân sách huấn luyện
ngắn chỉ 30 step. Nếu chỉ nhìn thấy loss cao mà không biết LR đã bị thay đổi, tôi có
thể kết luận sai rằng LoRA architecture hoặc dữ liệu không học được. Thực tế, cùng
architecture nhưng LR đúng đã đạt target `0.970`, nên nguyên nhân chính ở contrast
này là scale learning rate không phù hợp.

### 4.3 — QLoRA

`qlora` giảm peak VRAM từ `8.78 GB` xuống `3.86 GB`, tiết kiệm khoảng `4.92 GB`,
tương đương khoảng **56%** bộ nhớ. Đổi lại, target giảm từ `0.970` xuống `0.940`,
train loss tăng từ `0.6269` lên `0.7058`, đồng thời latency tăng từ khoảng
`1467.8 ms` lên `1813.2 ms`.

Trong chính thí nghiệm này, các con số có xu hướng ủng hộ khuyến nghị không dùng
QLoRA khi bộ nhớ không phải ràng buộc bắt buộc: tiết kiệm VRAM rất lớn nhưng phải trả
giá bằng một phần chất lượng và tốc độ. Tuy nhiên tôi không coi đây là kết luận phổ
quát cho mọi model hoặc dataset; nếu chỉ có GPU nhỏ thì mức giảm hơn 4.9 GB VRAM có
thể là đánh đổi hợp lý.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **`FAILED`**

`target Δ = +0.205` · `regression Δ = -0.313` · `valid_trace_rate = 0.0`

Fine-tune cải thiện rất mạnh tác vụ mục tiêu: target tăng từ `0.765` của optimized
prompt lên `0.970`, tức tăng khoảng `0.205`. Format vẫn đạt `1.000`, vì vậy model
đã học đúng cấu trúc JSON và hành vi triage mong muốn. Tuy nhiên regression giảm từ
`0.7911` xuống `0.4778`, tương ứng giảm khoảng `0.313`, lớn hơn rất nhiều so với
tolerance `0.020` của regression gate. Vì vậy verdict `FAILED` là hợp lý dù target
score rất cao.

Kết quả này cho thấy fine-tune đã chuyên môn hóa model quá mạnh vào dữ liệu ticket,
đồng thời làm suy giảm năng lực chung được đo bằng tập regression. Tôi không thay đổi
ngưỡng gate hoặc tập eval để biến kết quả thành PASS, vì như vậy sẽ phá vỡ tính toàn
vẹn của phép đánh giá. Hướng cải thiện hợp lý hơn là bổ sung khoảng 1–5% replay data
phổ thông, sau đó train lại và đo lại cả target lẫn regression. Một FAILED được phát
hiện đúng ở đây có giá trị hơn một model target cao nhưng được deploy mà không phát
hiện catastrophic forgetting.

---

## 6. Phân tích định tính

Phần định tính được chọn từ cùng tập đánh giá đã đóng băng. Tôi sử dụng cả các trường
hợp fine-tune tốt hơn baseline và các trường hợp fine-tune kém hơn baseline để tránh
cherry-picking.

> Bảng 5 case cụ thể được bổ sung sau khi đối chiếu prediction của baseline (b) và
> fine-tune trên cùng từng ticket. Không sử dụng training loss để quyết định ca thắng
> hay thua.

---

Trên toàn bộ 50 mẫu target, tôi không tìm thấy trường hợp nào fine-tune có
`ft_score < baseline_score`. Vì vậy tôi không tạo giả các ca “FT thua”.
Thay vào đó, tôi trình bày hai ca fine-tune thắng rõ ràng và ba ca hòa,
trong đó có các khác biệt ở từng trường dữ liệu để phân tích lỗi chi tiết.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Đổi size balo laptop, lần cuối mua ở đây | `doi_tra`, `thap`, `balo laptop`, `tieu_cuc` | `hoan_tien`, `cao`, `balo laptop`, `tieu_cuc` | `doi_tra`, `thap`, `balo laptop`, `tieu_cuc` | ✅ FT thắng: sửa đúng cả intent và urgency |
| 2 | Muốn đổi máy xay sinh tố sau 3 ngày | `doi_tra`, `trung_binh`, `máy xay sinh tố`, `tieu_cuc` | `van_chuyen`, `cao`, `máy xay sinh tố`, `tieu_cuc` | `doi_tra`, `trung_binh`, `máy xay sinh tố`, `tieu_cuc` | ✅ FT thắng: sửa đúng intent và urgency |
| 3 | Bình giữ nhiệt, chưa thấy tiền hoàn | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | `hoan_tien`, `trung_binh`, `bình giữ nhiệt`, `tich_cuc` | `hoan_tien`, `trung_binh`, `bình giữ nhiệt`, `tich_cuc` | ➖ Hòa 0.75: cả hai cùng sai urgency |
| 4 | Áo khoác gió bị lỗi, hỏi khi nào hoàn tiền | `san_pham_loi`, `thap`, `áo khoác gió`, `tich_cuc` | `san_pham_loi`, `trung_binh`, `áo khoác gió`, `tich_cuc` | `san_pham_loi`, `trung_binh`, `áo khoác gió`, `tich_cuc` | ➖ Hòa 0.75: lỗi tập trung ở urgency |
| 5 | Nồi chiên không dầu, hoàn tiền chậm | `hoan_tien`, `thap`, `nồi chiên không dầu`, `tieu_cuc` | `hoan_tien`, `cao`, `nồi chiên không dầu`, `tieu_cuc` | `hoan_tien`, `trung_binh`, `nồi chiên không dầu`, `tieu_cuc` | ➖ Hòa 0.75: FT cải thiện urgency từ `cao` xuống `trung_binh` nhưng vẫn chưa đạt `thap` |

Hai ca đầu cho thấy fine-tune cải thiện rõ rệt khả năng nhận diện intent và urgency.
Ba ca còn lại cho thấy lỗi phổ biến nhất của fine-tune nằm ở trường `urgency`,
đặc biệt là xu hướng dự đoán `trung_binh` thay vì `thap`.

Điểm đáng chú ý là trên tập target này không xuất hiện trường hợp fine-tune thua
baseline theo metric tổng. Tuy nhiên verdict chung vẫn là `FAILED` vì regression
giảm mạnh từ `0.7911` xuống `0.4778`. Điều này cho thấy một mô hình có thể không
thua trên task chuyên biệt nhưng vẫn không đủ an toàn để thay thế model gốc do
suy giảm năng lực tổng quát.
## 7. Kết luận & điều tôi học được

**Tôi chưa nên deploy bản fine-tune hiện tại như một model thay thế tổng quát.**
Fine-tune đạt target `0.970`, tăng đáng kể so với optimized prompt `0.765`, đồng thời
format đạt `1.000`. Nếu chỉ nhìn vào target task thì kết quả rất hấp dẫn. Tuy nhiên,
regression giảm từ `0.7911` xuống `0.4778`, làm regression gate thất bại với mức giảm
`0.313`, vượt xa tolerance `0.020`. Điều này cho thấy model đã chuyên môn hóa vào
250 ticket CSKH nhưng đánh đổi quá nhiều năng lực chung.

Thí nghiệm NB4 cũng cho thấy đòn bẩy quan trọng không đơn giản là tăng rank. Khi
`attn_only` được tăng lên `r=283` để khớp số tham số, target vẫn chỉ hòa `correct`
ở `0.970`, dù training loss thấp hơn. Ngược lại, chỉ giảm learning rate từ `1e-4`
xuống `1e-5` đã làm target rơi xuống `0.000`, cho thấy learning rate có tác động rất
lớn trong ngân sách 30 step. Loss mask cũng là điều kiện nền tảng: nếu mask sai thì
mọi phép so sánh phía sau gần như mất ý nghĩa. Với QLoRA, tôi đo được mức tiết kiệm
VRAM khoảng 56%, nhưng target giảm ba điểm phần trăm. Nếu tiếp tục dự án, ưu tiên của
tôi sẽ là chống regression bằng replay data thay vì chỉ tăng rank hay train thêm epoch.

**Ba điều tôi học được:**

1. Training loss thấp hơn không đồng nghĩa target score cao hơn; `attn_only` là bằng
   chứng trực tiếp khi loss thấp hơn `correct` nhưng target chỉ hòa.
2. Learning rate phải được xem là một biến thí nghiệm quan trọng. Chỉ thay `1e-4`
   thành `1e-5` đã biến một cấu hình đạt `0.970` target thành `0.000`.
3. Fine-tune phải được đánh giá đồng thời trên target và regression. Target tăng mạnh
   vẫn chưa đủ để deploy nếu năng lực chung suy giảm vượt ngưỡng.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn khoảng 1–5% dữ liệu replay/general vào
training set, giữ nguyên evaluation set, seed và các cấu hình còn lại, sau đó chạy lại
`correct` để kiểm tra liệu có giữ được target gần `0.970` trong khi đưa regression
trở lại gần baseline `0.7911` hay không.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub