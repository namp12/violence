# Web Application - Hướng Dẫn Sử Dụng

## 🌐 Giới Thiệu

Web application cho phép bạn sử dụng hệ thống nhận dạng bạo lực qua trình duyệt web, không cần chạy scripts CLI.

**Tính năng**:
- 📤 Upload video và phân tích tự động
- 📷 Phát hiện real-time từ webcam
- 📊 Dashboard xem lịch sử và thống kê
- 🔌 REST API để tích hợp vào hệ thống khác

## 🚀 Cài Đặt

### 1. Cài đặt dependencies

```bash
# Cài web dependencies
pip install -r requirements_web.txt
```

### 2. Đảm bảo model đã được train

Web app cần file model:
- `models/checkpoints/best_model.h5`

Nếu chưa có, chạy training trước:
```bash
python scripts/train.py
```

## 📖 Sử Dụng

### Khởi động server

```bash
# Chạy từ thư mục gốc
python web/app.py
```

Server sẽ start tại: **http://localhost:5000**

Mở trình duyệt và truy cập URL trên.

## 🎯 Các Tính Năng

### 1. Upload Video

1. Click vào tab "📤 Upload Video"
2. Kéo thả video hoặc click để chọn file
3. Hỗ trợ: MP4, AVI, MOV, MKV (tối đa 100MB)
4. Chờ phân tích (15-30 giây)
5. Xem kết quả với độ tin cậy

**Kết quả hiển thị**:
- 🔴 **Violent**: Phát hiện bạo lực
- 🟢 **Non-Violent**: Video an toàn
- Độ tin cậy (%)

### 2. Webcam Real-time

1. Click vào tab "📷 Webcam Real-time"
2. Click "Bắt Đầu" và cho phép truy cập camera
3. Webcam sẽ phân tích real-time
4. Kết quả cập nhật liên tục
5. Click "Dừng" để dừng lại

**Hiển thị**:
- Video stream trực tiếp
- Kết quả prediction real-time
- Cảnh báo khi phát hiện bạo lực
- Confidence score

### 3. Lịch Sử & Thống Kê

1. Click vào tab "📊 Lịch Sử"
2. Xem tất cả detections
3. Filter theo:
   - Tất cả
   - Upload
   - Webcam
4. Click "🔄 Refresh" để cập nhật

**Thống kê hiển thị**:
- Tổng số detections
- Số lượng violent
- Số lượng non-violent

## 🔌 API Documentation

### Health Check
```http
GET /api/status
```
**Response**:
```json
{
  "status": "ok",
  "ml_ready": true,
  "model_loaded": true,
  "database_connected": true
}
```

### Upload Video
```http
POST /api/upload
Content-Type: multipart/form-data

video: <file>
```

**Response**:
```json
{
  "success": true,
  "filename": "video.mp4",
  "message": "Video uploaded successfully"
}
```

### Predict Video
```http
POST /api/predict
Content-Type: application/json

{
  "filename": "video.mp4"
}
```

**Response**:
```json
{
  "success": true,
  "result": {
    "prediction": "Violent",
    "confidence": 0.87,
    "confidence_percent": "87.00%",
    "is_violent": true
  }
}
```

### Get History
```http
GET /api/history?limit=50&offset=0&source=upload
```

**Response**:
```json
{
  "success": true,
  "detections": [
    {
      "id": 1,
      "video_name": "test.mp4",
      "prediction": "Violent",
      "confidence": 0.85,
      "timestamp": "2026-01-19T10:30:00",
      "source": "upload"
    }
  ],
  "count": 1
}
```

### Get Statistics
```http
GET /api/statistics?days=7
```

**Response**:
```json
{
  "success": true,
  "daily_stats": [...],
  "totals": {
    "total": 100,
    "violent": 30,
    "non_violent": 70
  }
}
```

## 🔒 WebSocket Events

### Client → Server

**Connect to Webcam**:
```javascript
socket.emit('start_webcam');
```

**Send Frame**:
```javascript
socket.emit('webcam_frame', {
  frame: '<base64_image>'
});
```

**Stop Webcam**:
```javascript
socket.emit('stop_webcam');
```

### Server → Client

**Prediction Result**:
```javascript
socket.on('prediction', (result) => {
  // result = { prediction, confidence, is_violent, buffer_size }
});
```

## 🐛 Troubleshooting

### Port đã được sử dụng
```bash
# Thay đổi port trong web/app.py
# Dòng cuối: socketio.run(app, port=5001)
```

### CORS errors
Nếu frontend khác domain, update `CORS_ORIGINS` trong `web/config.py`

### Model not ready
```
ML model not ready. Please complete training first.
```
→ Chạy training: `python scripts/train.py`

### Webcam không hoạt động
- Kiểm tra browser đã cho phép camera
- Thử browser khác (Chrome/Edge recommended)
- Kiểm tra camera không bị app khác sử dụng

### Database errors
Delete database và restart:
```bash
rm web/detections.db
python web/app.py
```

## 📦 Cấu Trúc Files

```
web/
├── app.py              # Flask main app
├── config.py           # Configuration
├── database.py         # Database handler
├── api/
│   ├── video_handler.py    # Video upload/prediction
│   └── webcam_handler.py   # Webcam streaming
├── static/
│   ├── css/
│   │   └── style.css       # Styles
│   ├── js/
│   │   └── app.js          # Frontend logic
│   └── uploads/            # Uploaded videos
└── templates/
    └── index.html          # Main page
```

## 🚀 Deployment (Production)

### Option 1: Gunicorn (Recommended)

```bash
# Install
pip install gunicorn

# Run
gunicorn -w 4 -b 0.0.0.0:5000 --worker-class eventlet -w 1 web.app:app
```

### Option 2: Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt -r requirements_web.txt
EXPOSE 5000
CMD ["python", "web/app.py"]
```

Build & Run:
```bash
docker build -t violence-detection .
docker run -p 5000:5000 violence-detection
```

### Option 3: Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /socket.io {
        proxy_pass http://localhost:5000/socket.io;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## 🔐 Security (Production)

1. **Change SECRET_KEY** trong `web/config.py`
2. **Disable DEBUG** mode
3. **Set CORS_ORIGINS** to specific domain
4. **Use HTTPS** (SSL certificate)
5. **Rate limiting** cho upload API
6. **File validation** nghiêm ngặt hơn

## 📝 Notes

- Web app **KHÔNG ẢNH HƯỞNG** đến CLI scripts
- Tất cả scripts (`train.py`, `predict.py`, etc.) vẫn hoạt động bình thường
- Database (SQLite) dễ dàng migrate sang SQL Server nếu cần
- Webcam streaming yêu cầu HTTPS khi deploy production

---

**Developed by**: Violence Detection Team  
**Version**: 1.0.0  
**Last Updated**: January 2026
