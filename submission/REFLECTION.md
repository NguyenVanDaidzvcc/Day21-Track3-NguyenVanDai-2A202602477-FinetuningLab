# Reflection — Lab 21
**Họ tên:** Nguyễn Văn Đại  
**MSSV:** 2A202602477 
*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là fine-tune có thể cải thiện rất mạnh task chuyên biệt nhưng
vẫn không đủ tốt để deploy. LoRA tăng target score từ `0.765` lên `0.970`, nhưng regression
lại giảm từ `0.7911` xuống `0.4778`, khiến verdict cuối cùng là `FAILED`.

Tôi cũng bất ngờ khi `attn_only` có train loss thấp hơn `correct` (`0.5374` so với
`0.6269`) nhưng target score của hai cấu hình đều bằng `0.970`. Điều này cho thấy
training loss thấp hơn không đồng nghĩa model tốt hơn trên task thực tế.
**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở NB4 vì phải train nhiều cấu hình LoRA khác nhau:
`attn_only`, `wrong_lr` và `qlora`. NB4 mất khoảng 21.8 phút trên T4, lâu hơn rõ rệt
so với các bước khác.

Ban đầu tôi nghĩ phần khó nhất sẽ là viết code fine-tuning, nhưng thực tế phần tốn thời gian
hơn là giữ các thí nghiệm công bằng, chạy cùng số step, chờ GPU và kiểm tra lại kết quả bằng
full evaluation set. Tôi cũng phải chạy lại NB2 và NB5 vì lần đầu `EVAL_LIMIT=8` chỉ là
smoke evaluation chứ chưa phải kết quả dùng để nộp
**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước lab này tôi nghĩ chỉ cần training loss giảm mạnh và target task tốt hơn thì có thể xem
fine-tuning là thành công. Sau bài lab, tôi không còn tin điều đó.

Model `correct` đạt target `0.970`, nhưng regression giảm hơn `0.31`. Điều này cho thấy
một model có thể học rất tốt dữ liệu chuyên biệt nhưng đồng thời quên một phần năng lực chung.
Vì vậy fine-tuning cần được đánh giá bằng cả target set và regression set, không chỉ bằng
training loss hoặc một metric duy nhất.
**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để đọc cấu trúc repo và rubric, giải thích các notebook, hỗ trợ viết
NB1–NB6, xử lý lỗi môi trường Windows/Colab, đọc kết quả `verify.py` và hỗ trợ hoàn thiện
REPORT.md.

AI assistant hữu ích trong việc tăng tốc thao tác nhưng không thể được tin tuyệt đối. Một số
hướng dẫn ban đầu cần kiểm tra lại với code thật trong repo, đặc biệt là cấu hình
`EVAL_LIMIT`. Lần chạy đầu vẫn dùng `EVAL_LIMIT=8`, vì vậy chỉ đánh giá 8 mẫu và tôi phải
chạy lại NB2/NB5 trên đủ 50 mẫu. Tôi rút ra rằng mọi đề xuất từ AI cần được kiểm chứng bằng
rubric, source code và `verify.py` thay vì copy rồi coi là đúng ngay.
**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên của tôi sẽ không phải là train model ngay. Tôi sẽ cùng khách hàng xác định rõ
task, output schema, metric thành công và tạo một evaluation set đại diện trước khi fine-tune.

Sau đó tôi sẽ chạy một baseline mạnh bằng prompt engineering và đóng băng tập evaluation.
Chỉ khi baseline chưa đáp ứng yêu cầu thì tôi mới fine-tune. Khi train, tôi sẽ kiểm tra loss
mask, theo dõi target và regression riêng biệt, đồng thời chuẩn bị replay/general data để
giảm nguy cơ catastrophic forgetting.