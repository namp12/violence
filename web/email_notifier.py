"""
Email notification service for violence detection alerts.
Sends email when violence is detected.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
import os

class EmailNotifier:
    """Handle email notifications for violence detection."""
    
    def __init__(self, smtp_server='smtp.gmail.com', smtp_port=587):
        """
        Initialize email notifier.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP port
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        
        # Email credentials (from environment variables for security)
        self.sender_email = os.environ.get('EMAIL_SENDER', 'your-email@gmail.com')
        self.sender_password = os.environ.get('EMAIL_PASSWORD', 'your-app-password')
        
        # Recipient
        self.recipient_email = 'nnam38789@gmail.com'
        
        # Alert settings
        self.enabled = os.environ.get('EMAIL_ALERTS_ENABLED', 'true').lower() == 'true'
        self.min_confidence = float(os.environ.get('EMAIL_MIN_CONFIDENCE', '0.7'))
    
    def send_alert(self, video_name, confidence, source='upload', timestamp=None):
        """
        Send email alert for violence detection.
        
        Args:
            video_name: Name of the video/source
            confidence: Detection confidence (0-1)
            source: 'upload' or 'webcam'
            timestamp: Detection timestamp
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            print("Email alerts disabled")
            return False
        
        if confidence < self.min_confidence:
            print(f"Confidence {confidence} below threshold {self.min_confidence}, skipping email")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = '🚨 CẢNH BÁO: Phát Hiện Bạo Lực!'
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            
            # Timestamp
            if timestamp is None:
                timestamp = datetime.now()
            time_str = timestamp.strftime('%d/%m/%Y %H:%M:%S')
            
            # Create HTML content
            html_content = self._create_html_alert(
                video_name, confidence, source, time_str
            )
            
            # Attach HTML
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print(f"✓ Email alert sent to {self.recipient_email}")
            return True
        
        except Exception as e:
            print(f"✗ Failed to send email: {str(e)}")
            return False
    
    def _create_html_alert(self, video_name, confidence, source, time_str):
        """Create HTML content for alert email."""
        
        source_icon = '📤' if source == 'upload' else '📷'
        source_text = 'Video Upload' if source == 'upload' else 'Webcam Real-time'
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            margin: 0;
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}
        .icon {{
            font-size: 48px;
            margin-bottom: 10px;
        }}
        .content {{
            padding: 30px;
        }}
        .alert-box {{
            background: #fef2f2;
            border-left: 4px solid #ef4444;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 4px;
        }}
        .alert-box h2 {{
            margin: 0 0 10px 0;
            color: #dc2626;
            font-size: 18px;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #e5e7eb;
        }}
        .info-row:last-child {{
            border-bottom: none;
        }}
        .label {{
            font-weight: 600;
            color: #6b7280;
        }}
        .value {{
            color: #111827;
            font-weight: 500;
        }}
        .confidence {{
            font-size: 32px;
            font-weight: bold;
            color: #ef4444;
            text-align: center;
            margin: 20px 0;
        }}
        .footer {{
            background: #f9fafb;
            padding: 20px;
            text-align: center;
            color: #6b7280;
            font-size: 14px;
        }}
        .btn {{
            display: inline-block;
            background: #ef4444;
            color: white;
            padding: 12px 30px;
            text-decoration: none;
            border-radius: 6px;
            margin-top: 20px;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="icon">🚨</div>
            <h1>CẢNH BÁO BẠO LỰC</h1>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">
                Hệ thống đã phát hiện hành vi bạo lực
            </p>
        </div>
        
        <div class="content">
            <div class="alert-box">
                <h2>⚠️ Phát hiện bạo lực với độ tin cậy cao!</h2>
                <p style="margin: 5px 0 0 0; color: #6b7280;">
                    Vui lòng kiểm tra ngay để xác nhận và xử lý.
                </p>
            </div>
            
            <div class="confidence">
                {confidence * 100:.1f}% Tin cậy
            </div>
            
            <div style="margin-top: 20px;">
                <div class="info-row">
                    <span class="label">Nguồn:</span>
                    <span class="value">{source_icon} {source_text}</span>
                </div>
                <div class="info-row">
                    <span class="label">Video/Camera:</span>
                    <span class="value">{video_name}</span>
                </div>
                <div class="info-row">
                    <span class="label">Thời gian:</span>
                    <span class="value">{time_str}</span>
                </div>
                <div class="info-row">
                    <span class="label">Trạng thái:</span>
                    <span class="value" style="color: #ef4444; font-weight: 700;">
                        VIOLENT
                    </span>
                </div>
            </div>
            
            <div style="text-align: center;">
                <a href="http://localhost:5000" class="btn">
                    Xem Chi Tiết Trên Hệ Thống
                </a>
            </div>
        </div>
        
        <div class="footer">
            <p style="margin: 0 0 10px 0;">
                <strong>Violence Detection System</strong>
            </p>
            <p style="margin: 0; font-size: 12px;">
                Email tự động được gửi từ hệ thống AI nhận dạng bạo lực.<br>
                Powered by 3D CNN Deep Learning
            </p>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def test_connection(self):
        """Test email connection and send test email."""
        try:
            msg = MIMEMultipart()
            msg['Subject'] = '✓ Test Email - Violence Detection System'
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            
            html = """
            <html>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
                <h2 style="color: #10b981;">✓ Kết nối email thành công!</h2>
                <p>Hệ thống Violence Detection đã sẵn sàng gửi cảnh báo.</p>
                <p>Email này xác nhận rằng cấu hình email đã được thiết lập đúng.</p>
                <hr>
                <p style="color: #6b7280; font-size: 14px;">
                    Violence Detection System © 2026
                </p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print(f"✓ Test email sent successfully to {self.recipient_email}")
            return True
        
        except Exception as e:
            print(f"✗ Email test failed: {str(e)}")
            print("\nPlease check:")
            print("  1. EMAIL_SENDER and EMAIL_PASSWORD environment variables")
            print("  2. Gmail App Password (not regular password)")
            print("  3. Internet connection")
            return False
