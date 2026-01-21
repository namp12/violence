# Hướng Dẫn Setup Email Alerts

## 📧 Tổng Quan

Hệ thống sẽ **tự động gửi email cảnh báo** đến **nnam38789@gmail.com** khi phát hiện hành vi bạo lực.

## 🔧 Setup Email (QUAN TRỌNG)

### Bước 1: Tạo Gmail App Password

Email alerts sử dụng Gmail SMTP. Bạn cần tạo "App Password" (không dùng mật khẩu Gmail thường).

**Cách tạo App Password**:

1. Truy cập: https://myaccount.google.com/security
2. Bật **2-Step Verification** (nếu chưa có)
3. Tìm **App passwords**
4. Chọn **Mail** và **Windows Computer**
5. Copy password (16 ký tự, dạng: `xxxx xxxx xxxx xxxx`)

### Bước 2: Set Environment Variables

#### Windows (PowerShell):
```powershell
# Set cho session hiện tại
$env:EMAIL_SENDER = "your-gmail@gmail.com"
$env:EMAIL_PASSWORD = "your-app-password-here"
$env:EMAIL_ALERTS_ENABLED = "true"
$env:EMAIL_MIN_CONFIDENCE = "0.7"

# Hoặc set vĩnh viễn
[System.Environment]::SetEnvironmentVariable('EMAIL_SENDER', 'your-gmail@gmail.com', 'User')
[System.Environment]::SetEnvironmentVariable('EMAIL_PASSWORD', 'your-app-password', 'User')
[System.Environment]::SetEnvironmentVariable('EMAIL_ALERTS_ENABLED', 'true', 'User')
[System.Environment]::SetEnvironmentVariable('EMAIL_MIN_CONFIDENCE', '0.7', 'User')
```

#### Linux/Mac:
```bash
export EMAIL_SENDER="your-gmail@gmail.com"
export EMAIL_PASSWORD="your-app-password"
export EMAIL_ALERTS_ENABLED="true"
export EMAIL_MIN_CONFIDENCE="0.7"

# Thêm vào ~/.bashrc hoặc ~/.zshrc để vĩnh viễn
```

### Bước 3: Test Email Connection

```bash
python -c "from web.email_notifier import EmailNotifier; EmailNotifier().test_connection()"
```

Nếu thành công, bạn sẽ nhận được test email tại **nnam38789@gmail.com**.

## ⚙️ Cấu Hình

### Environment Variables

| Variable | Giá trị Mặc Định | Mô tả |
|---------|------------------|-------|
| `EMAIL_SENDER` | `your-email@gmail.com` | Gmail của bạn (để gửi) |
| `EMAIL_PASSWORD` | `your-app-password` | Gmail App Password |
| `EMAIL_ALERTS_ENABLED` | `true` | Bật/tắt email alerts |
| `EMAIL_MIN_CONFIDENCE` | `0.7` | Ngưỡng confidence tối thiểu (0-1) |

### Receiver Email

Email luôn gửi đến: **nnam38789@gmail.com** (hardcoded)

Nếu muốn thay đổi, edit `web/email_notifier.py` dòng 24:
```python
self.recipient_email = 'your-email@gmail.com'
```

## 📨 Email Alert Format

Khi phát hiện bạo lực, email sẽ chứa:

✅ **Subject**: 🚨 CẢNH BÁO: Phát Hiện Bạo Lực!  
✅ **HTML Template** đẹp mắt với:
- Mức độ confidence (%)
- Nguồn (Upload video hoặc Webcam)
- Tên video/camera
- Thời gian phát hiện
- Link đến web app

## 🧪 Test Email

### Test 1: Test Connection
```python
from web.email_notifier import EmailNotifier

notifier = EmailNotifier()
notifier.test_connection()  # Gửi test email
```

### Test 2: Test Alert Email
```python
from web.email_notifier import EmailNotifier

notifier = EmailNotifier()
notifier.send_alert(
    video_name='Test Video',
    confidence=0.95,
    source='upload'
)
```

Kiểm tra inbox của **nnam38789@gmail.com**.

## 🚀 Sử Dụng

### Automatic (Recommended)

Email tự động gửi khi:
1. Upload video → Phát hiện bạo lực → Gửi email
2. Webcam real-time → Phát hiện bạo lực → Gửi email

**Điều kiện gửi**:
- Prediction = "Violent"
- Confidence ≥ 0.7 (70%)
- Email alerts enabled

### Start Web App với Email

```bash
# Set environment variables
$env:EMAIL_SENDER = "your-gmail@gmail.com"
$env:EMAIL_PASSWORD = "your-app-password"

# Start server
python web/app.py
```

## ⚠️ Troubleshooting

### "Authentication failed"
- Kiểm tra EMAIL_PASSWORD là **App Password** (không phải mật khẩu Gmail)
- Đảm bảo 2-Step Verification đã bật

### "Connection timed out"
- Kiểm tra firewall/antivirus
- Kiểm tra internet connection
- Thử port 465 thay vì 587 (edit `email_notifier.py`)

### Email không gửi
- Check console log: `✓ Email alert sent` hoặc `✗ Failed to send email`
- Kiểm tra confidence ≥ min_confidence (0.7)
- Check `EMAIL_ALERTS_ENABLED=true`

### Gmail blocked email
- Truy cập https://myaccount.google.com/lesssecureapps
- Hoặc check https://accounts.google.com/DisplayUnlockCaptcha

## 🔐 Security Tips

1. **Không commit** App Password vào Git
2. **Sử dụng** environment variables
3. **Tạo email riêng** cho ứng dụng nếu production
4. **Log** email failures để debug
5. **Rate limit** để tránh spam (tự động trong code)

## 📋 Tóm Tắt Quick Start

```bash
# 1. Tạo Gmail App Password
# 2. Set environment variables
$env:EMAIL_SENDER = "your-gmail@gmail.com"
$env:EMAIL_PASSWORD = "xxxx xxxx xxxx xxxx"

# 3. Test
python -c "from web.email_notifier import EmailNotifier; EmailNotifier().test_connection()"

# 4. Start web app
python web/app.py

# 5. Test bằng cách upload violent video
# 6. Check email tại nnam38789@gmail.com
```

## ✉️ Email Template Preview

```
┌──────────────────────────────────────┐
│    🚨 CẢNH BÁO BẠO LỰC                │
├──────────────────────────────────────┤
│                                       │
│  ⚠️ Phát hiện bạo lực với độ tin cậy │
│     cao!                              │
│                                       │
│         95.3% Tin cậy                 │
│                                       │
│  Nguồn: 📤 Video Upload               │
│  Video: test_vid.mp4                  │
│  Thời gian: 19/01/2026 18:00:00       │
│  Trạng thái: VIOLENT                  │
│                                       │
│    [Xem Chi Tiết Trên Hệ Thống]      │
│                                       │
└──────────────────────────────────────┘
```

---

**Email Recipient**: nnam38789@gmail.com  
**System**: Violence Detection AI  
**Powered by**: 3D CNN Deep Learning
