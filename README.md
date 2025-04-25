# Hệ Thống Nhận Dạng Khuôn Mặt

Hệ thống nhận dạng khuôn mặt sử dụng FaceNet và SQLite để quản lý và nhận dạng người trong ảnh. Hệ thống gồm hai ứng dụng chính: ứng dụng command line (001x1.py) và giao diện web (002.py).

![image](https://github.com/user-attachments/assets/ded8244a-04ee-434b-816a-b435c92d0211)


## Cài đặt

### Yêu cầu

- Python 3.8+
- PyTorch
- OpenCV
- FaceNet-PyTorch
- SQLite3
- Flask (cho web interface)

### Cài đặt thư viện

```bash
pip install torch torchvision opencv-python facenet-pytorch scikit-learn flask
```

Nếu muốn sử dụng GPU:

```bash
pip install torch==2.4.1 torchvision==0.19.1
```

## Cấu trúc thư mục

├── data_files/ # Thư mục chứa dữ liệu
│ ├── face_recognition.db # Database SQLite
│ ├── embeddings/ # Thư mục chứa embedding vectors
│ └── faces/ # Thư mục chứa ảnh khuôn mặt đã cắt
│ ├── face_1/ # Thư mục cho face_id 1
│ ├── face_2/ # Thư mục cho face_id 2
│ └── ...
├── 001x1.py # Ứng dụng command line
├── 002.py # Ứng dụng web
└── templates/ # Thư mục chứa template HTML
├── index.html
└── face_detail.html

## Ứng dụng Command Line (001x1.py)

### Chức năng chính

- Nhận diện khuôn mặt trong ảnh hoặc thư mục chứa ảnh
- Trích xuất embedding vector cho mỗi khuôn mặt sử dụng facenet_pytorch
- So sánh khuôn mặt mới với cơ sở dữ liệu có sẵn
- Quản lý người dùng và khuôn mặt trong cơ sở dữ liệu SQLite

### Các tính năng chi tiết

- Khởi tạo mô hình MTCNN và InceptionResnetV1 từ facenet_pytorch
- Tạo cấu trúc thư mục và database ban đầu
- Phát hiện khuôn mặt trong ảnh
- Trích xuất vector đặc trưng (embedding)
- Tính toán và kiểm tra trùng lặp ảnh bằng MD5 hash
- Gán khuôn mặt cho người dùng
- Gộp nhiều khuôn mặt vào một người dùng
- Tái học đặc trưng nhận dạng cho người dùng

### Cách sử dụng

```bash
# Xử lý ảnh đơn lẻ
python 001x1.py --image <đường dẫn ảnh>

# Xử lý thư mục
python 001x1.py --folder <đường dẫn thư mục>

# Gán khuôn mặt cho người dùng
python 001x1.py --assign --face_id <id> --user_name <tên>

# Gộp nhiều khuôn mặt vào một người dùng
python 001x1.py --merge --face_ids <id1,id2,...> --user_name <tên>

# Tái học embedding cho một người dùng
python 001x1.py --relearn --user_id <id>
```

## Ứng dụng Web (002.py)

### Chức năng chính

- Cung cấp giao diện web để xem, quản lý khuôn mặt và ảnh
- Cho phép gán khuôn mặt cho người dùng mới hoặc có sẵn
- Xóa ảnh hoặc xóa khuôn mặt
- Hiển thị thông tin chi tiết về khuôn mặt và ảnh

### Các tính năng chi tiết

- Server Flask với các routes:
  - `/`: Hiển thị tất cả khuôn mặt
  - `/face/<face_id>`: Chi tiết khuôn mặt và các ảnh liên quan
  - `/delete_image/<image_id>`: Xóa ảnh
  - `/reassign_image`: Gán lại ảnh cho người dùng khác
  - `/delete_faces`: Xóa khuôn mặt
  - `/batch_update`: Cập nhật hàng loạt
- Tạo tự động các template HTML
- Chuyển đổi ảnh sang base64 để hiển thị trên web
- Di chuyển ảnh giữa các thư mục khi gán lại
- Cập nhật embedding khi gán lại ảnh cho người dùng khác

### Cách sử dụng

```bash
# Chạy ứng dụng web
python 002.py

# Sau đó truy cập vào địa chỉ
# http://localhost:5000
```

## Mô hình dữ liệu

### Bảng `users`

- `id`: ID người dùng (primary key)
- `name`: Tên người dùng

### Bảng `faceIds`

- `id`: ID khuôn mặt (primary key)
- `embedding_path`: Đường dẫn đến file embedding
- `user_id`: ID người dùng (foreign key)
- `created_at`: Thời gian tạo

### Bảng `face_images`

- `id`: ID của ảnh (primary key)
- `face_id`: ID khuôn mặt (foreign key)
- `file_path`: Đường dẫn đến file ảnh
- `md5_hash`: MD5 hash của ảnh để phát hiện trùng lặp

## Quy trình làm việc

1. Sử dụng 001x1.py để quét và nhận dạng khuôn mặt trong ảnh mới
   - Phát hiện khuôn mặt trong ảnh
   - Trích xuất embedding vector
   - So sánh với database để xác định người
   - Lưu khuôn mặt mới vào hệ thống

2. Sử dụng 002.py để quản lý các khuôn mặt đã nhận dạng
   - Xem danh sách tất cả khuôn mặt
   - Xem chi tiết của một khuôn mặt và các ảnh liên quan
   - Gán lại ảnh cho người khác
   - Xóa ảnh hoặc khuôn mặt
   - Cập nhật embedding khi có thay đổi

## Công nghệ sử dụng

- **MTCNN**: Phát hiện khuôn mặt trong ảnh
- **InceptionResnetV1**: Trích xuất vector đặc trưng (embedding)
- **FaceNet-PyTorch**: Thư viện cung cấp mô hình nhận dạng
- **PyTorch**: Framework học máy
- **Flask**: Framework web
- **SQLite**: Cơ sở dữ liệu
- **OpenCV**: Xử lý ảnh

## Hướng phát triển tiếp theo

- Cải thiện giao diện người dùng web
- Tách template, CSS, JavaScript thành các file riêng biệt
- Thêm tính năng tìm kiếm khuôn mặt
- Cải thiện tốc độ xử lý
- Thêm chức năng nhận dạng theo thời gian thực
- Xử lý hình ảnh theo batch với nhiều CPU/GPU

## Mối quan hệ giữa 001x1.py và 002.py

- **Cơ sở dữ liệu chung**: Cả hai ứng dụng đều sử dụng cùng một cơ sở dữ liệu SQLite
- **Công nghệ nhận dạng**: Cả hai đều sử dụng facenet_pytorch để đảm bảo nhất quán
- **Quy trình bổ sung**: 001x1.py tập trung vào nhận dạng khuôn mặt mới, 002.py tập trung vào quản lý khuôn mặt đã nhận dạng

## Lưu ý quan trọng

- Cần chạy 001x1.py trước để khởi tạo cấu trúc thư mục và database
- Khi gán lại ảnh cho người dùng khác, embedding vector sẽ được tự động cập nhật
- Luôn sao lưu thư mục `data_files` trước khi thực hiện các thay đổi lớn

---

*Sản phẩm này được phát triển như một dự án học tập và thực hành, không nên sử dụng trong môi trường sản xuất thực tế mà không có kiểm thử kỹ lưỡng.*

