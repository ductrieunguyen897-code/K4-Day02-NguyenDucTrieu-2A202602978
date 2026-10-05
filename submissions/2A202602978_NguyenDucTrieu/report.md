# Lab Day 2 — Nguyễn Đức Triều · 2A202602978

CHƯA HOÀN TẤT THỰC NGHIỆM: các ô trống là chưa đo, không phải điểm 0.

## 1. Tóm tắt

So sánh 5 backbone với cùng công thức; ablation 3 trục A/B/C; 1-view và 4 biến thể suy luận. Chọn theo macro-F1 val; khi hòa chọn ECE thấp hơn rồi độ trễ thấp hơn. Test chỉ mở sau khi khóa cấu hình. Chưa có đủ kết quả thì chưa thể kết luận cấu hình nào tốt hơn.

## 2. Dữ liệu và thiết lập

DeepWeeds fold 0 nguyên bản, 9 lớp; không gộp val vào train. AdamW, 12 epoch mặc định, batch 64, LR backbone/head 1e-4/1e-3, weight decay 0.05 (trừ bias/norm), warmup 1 epoch rồi cosine theo bước. ImageNet mean/std; train crop+flip 224, val resize 256 + center crop 224. Cấu hình thực tế và phiên bản được lưu riêng trong runs/<exp_id>/seed<k>/. Macro-F1 trung bình 9 lớp, ECE 15 bin, std mẫu ddof=1. Seed sàng 0; chung kết 0,1,2. GMAC là ước lượng THOP, có thể thiếu toán tử attention; không coi đó là phép đếm chính xác cho transformer.

![Phân bố lớp](eda/class_counts.png)

![Ảnh theo lớp](eda/examples.png)

![Augmentation](eda/augmentation.png)

## 3. Backbone

*Chưa có kết quả chạy thật.*

## 4. Công thức huấn luyện

*Chưa có kết quả chạy thật.*

T01–T08 thay một yếu tố từ T00. T09 kết hợp lựa chọn val tốt nhất theo từng trục; đây là tìm kiếm cấu hình, không chứng minh hiệu ứng độc lập. Chênh lệch sàng chỉ từ một seed.

## 5. Suy luận

*Chưa có kết quả chạy thật.*

Đo forward và gộp view trên thiết bị, không tính đọc ảnh, tiền xử lý CPU hay truyền CPU→GPU. Warmup 10, đo 100 lượt, đồng bộ CUDA; batch 1 và 32. Five-crop dùng ảnh resize 256, crop 224 ở bốn góc và giữa. Temperature được khớp riêng trên val mỗi seed; ECE val sau fit là số nội mẫu.

## 6. Chung kết

*Chưa có kết quả chạy thật.*

Ma trận nhầm lẫn và ảnh lỗi được xuất vào analysis/ theo từng seed. Không điều chỉnh mô hình sau khi xem các ảnh này.

## 7. Kết luận và khuyến nghị

Chỉ kết luận sau khi có đầy đủ 3 seed cho cả mốc và cấu hình cuối. So sánh chênh lệch macro-F1 với std; nếu nhỏ hơn nhiễu, ghi “không phân biệt được”. Với robot, lọc cấu hình theo p95 đo trên thiết bị đích ≤ ngân sách 30–100 ms, rồi chọn macro-F1 val cao nhất. Thời gian trên GPU Colab không thay thế đo trên robot.

## 8. Hạn chế

Một fold; sàng một seed; chọn nhiều cấu hình trên cùng val có thể quá khớp val. Chia ngẫu nhiên không bảo đảm tổng quát hóa sang địa điểm/mùa khác. Nên chạy nhiều fold và đo trực tiếp trên thiết bị triển khai. Chưa có quan sát trực tiếp của người học về các cặp lớp dễ nhầm; cần bổ sung dựa trên ảnh EDA và ảnh lỗi.

## 9. Phụ lục

Notebook: [code/lab_day2.ipynb](code/lab_day2.ipynb). Các bảng trên sinh từ log và predictions bằng code/report.py; results.xlsx chứa cùng dữ liệu. eval.py gốc được dùng để score/grade; grade.txt lưu đầu ra tự chấm.