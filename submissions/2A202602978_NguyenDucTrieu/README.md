# DeepWeeds — Nguyễn Đức Triều · 2A202602978

## Trạng thái

Đã cài đặt pipeline và kiểm thử CPU. **Chưa có kết quả huấn luyện GPU, chưa hoàn tất bài nộp.** `results.xlsx` và `report.md` hiện phản ánh đúng các kết quả đã có; ô trống là chưa đo. Không sử dụng số giả. Máy local không có CUDA hoạt động.

## Chạy trên Kaggle hoặc Colab bằng git clone

Notebook chính: `code/lab_day2.ipynb`, lấy trực tiếp từ mẫu `starter/lab_day2.ipynb`. Giữ nguyên tất cả ô Markdown, ô tải dữ liệu và ô `eval.py score/grade`; điền các ô TODO. Ô môi trường cài thư viện và clone repo GitHub của người học.

1. Commit và push mã bài làm mới trong `submissions/2A202602978_NguyenDucTrieu/` lên repo `https://github.com/ductrieunguyen897-code/K4-Day02-NguyenDucTRieu-2A202602978`. File đang có trên máy local không tự xuất hiện trong bản clone trên Kaggle/Colab. Trợ lý chưa commit/push thay bạn.
2. Import `code/lab_day2.ipynb` vào Kaggle hoặc Colab. Bật GPU; trên Kaggle bật Internet.
3. Chạy ô cài đặt: notebook tự `git clone` nếu chưa có repo, rồi import mã từ thư mục bài nộp `submissions/2A202602978_NguyenDucTrieu/code`. Không cần Add Input hoặc upload ZIP mã nguồn.
4. Chạy lần lượt các ô theo mẫu: tải dữ liệu → Bước 0–5. Kết quả ở `/kaggle/working` (Kaggle) hoặc `/content` (Colab).
5. Cuối notebook tạo `submission_2A202602978.zip`. Đọc báo cáo và ảnh lỗi, bổ sung nhận xét trước khi nộp. Lưu/tải output trước khi phiên kết thúc; notebook không mount Google Drive.

Lệnh `mkdir -p .../submissions/...` và `cp -r .../starter .../code` trong hướng dẫn dùng để khởi tạo bài làm. Thư mục code ở repo này đã hoàn thiện nên bỏ qua bước sao chép, tránh ghi đè bằng bộ khung. Nếu repo đã clone từ trước, cập nhật bằng `git -C <REPO_DIR> pull --ff-only` trước khi import module; nếu đã import mã cũ, khởi động lại kernel.

**Link notebook chạy thật:** chưa có; cần lưu vào tài khoản người học và bổ sung URL. Chưa chạy huấn luyện GPU thật.

### Tiếp tục khi phiên ngắt

Mã huấn luyện lưu checkpoint sau mỗi epoch nhưng không tự sao lưu ra ngoài phiên Kaggle. Lưu output của phiên và thêm output cũ làm Input ở phiên mới, rồi sao chép `runs/`, `predictions/`, `curves/`, `eda/`, các file `selected_*.json` và `final_lock.json` (nếu có) về đúng `/kaggle/working` trước khi tiếp tục. Tải lại ảnh nếu cần. Giữ nguyên cấu hình/phiên bản thư viện.

Nếu đã có `final_lock.json`, bỏ qua Bước 1–3 và tiếp tục Bước 4; không chọn lại mô hình. Khi test bị ngắt giữa lượt, marker sẽ yêu cầu kiểm tra và ghi rõ sự cố thay vì tự chạy test lại. Các ô score/grade chỉ đọc dự đoán đã lưu.

## Thí nghiệm

- B01–B05: ResNet-50, ResNeXt-50, ConvNeXt-Tiny, DeiT-Small, EfficientNet-B0; cùng baseline 12 epoch, batch 64, seed 0.
- T00 baseline; T01 scratch; T02 frozen; T03 color; T04 Mixup; T05 CutMix; T06 label smoothing; T07 focal; T08 weighted CE; T09 kết hợp yếu tố được chọn trên val.
- I00 1-view; I01 horizontal flip trung bình xác suất; I02 horizontal flip trung bình logit; I03 five-crop; I04 temperature scaling. Có thêm hàm ensemble, multiscale, EMA và fuse BN để mở rộng.
- Chung kết F01 và mốc T00: seed 0, 1, 2. Khóa `final_lock.json` trước khi suy luận test. Mỗi lượt test có marker và receipt để ngăn chạy lặp âm thầm.
- Temperature chỉ fit val. Tiêu chí chọn checkpoint: macro-F1 9 lớp cao nhất, hòa lấy epoch sớm. Giữa phương pháp suy luận: F1, rồi ECE, rồi p50.

Baseline cố định preprocessing ImageNet: crop 224, val resize 256 + center crop. Ghi đầy đủ tag và nguồn pretrained do timm phân giải trong `environment.json`. Seed không thay đổi split. GMAC dùng THOP và ghi rõ có thể bỏ sót toán tử attention.

## Chạy bằng terminal

Tại gốc repo, dùng Python có torch/torchvision phù hợp GPU:

```bash
pip install -r submissions/2A202602978_NguyenDucTrieu/code/requirements.txt
python submissions/2A202602978_NguyenDucTrieu/code/prepare.py --data data --output submissions/2A202602978_NguyenDucTrieu/eda --sanity
python submissions/2A202602978_NguyenDucTrieu/code/experiments.py all --root submissions/2A202602978_NguyenDucTrieu --data data
python submissions/2A202602978_NguyenDucTrieu/code/report.py --root submissions/2A202602978_NguyenDucTrieu
python -m unittest discover -s submissions/2A202602978_NguyenDucTrieu/tests -v
```

`experiments.py` hỗ trợ từng giai đoạn: `backbones`, `training`, `inference`, `final`. Mọi huấn luyện dùng `train.run(Config(...))`. Mỗi epoch lưu state optimizer/scheduler/scaler/EMA/RNG để resume. Không lưu dự đoán test ở giai đoạn sàng.

`eval.py score`/`grade` được gọi tự động ở cuối. `report.py` chỉ đọc predictions có sẵn để tạo Excel/báo cáo, không chạy model lại. Thời gian công bố là đo thật sau warmup, đồng bộ CUDA, 100 lượt; không tính decoding, CPU preprocessing, host→device transfer.

## Giới hạn và tái lập

Không bảo đảm hoàn thành mọi thí nghiệm trong một phiên GPU Kaggle. Xem `history.csv` epoch đầu để tính ngân sách thực tế. Nếu giảm epoch hoặc batch, đổi đồng nhất cho các backbone và ghi lý do; không sửa cùng exp_id đã chạy. GPU/phiên bản khác có thể khác kết quả dù seed giống nhau. Phiên bản thực tế được lưu theo từng run trong `environment.json`.

Tests ở `tests/test_starter.py` của repo yêu cầu bộ khung phải còn TODO; bốn file starter đã được sửa từ trước nên test kiểm tra stub có thể thất bại. Bài làm nằm riêng ở thư mục này; không sửa `eval.py` hoặc ghi đè các thay đổi sẵn có trong starter.
