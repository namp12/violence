"""
Quick demo script to show all capabilities.
Run this after training is complete.
"""
import os
import sys

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def main():
    print_header("Violence Detection Demo")
    
    print("\n📹 Dự án nhận dạng bạo lực CNN 3D")
    print("\nChọn chế độ demo:\n")
    
    print("1. 🎬 Test trên video có sẵn")
    print("2. 📁 Test batch videos từ thư mục")
    print("3. 📷 Real-time detection từ webcam")
    print("4. 📊 Xem kết quả đánh giá model")
    print("5. 🚪 Thoát")
    
    choice = input("\nNhập lựa chọn (1-5): ").strip()
    
    if choice == '1':
        print_header("Test trên video đơn")
        video_path = input("Nhập đường dẫn video (hoặc Enter để dùng mặc định): ").strip()
        if not video_path:
            video_path = "Real Life Violence Dataset/Violence/V_1.mp4"
        
        print(f"\nĐang phân tích video: {video_path}")
        os.system(f'python scripts/predict.py --video_path "{video_path}"')
    
    elif choice == '2':
        print_header("Test batch videos")
        folder_path = input("Nhập đường dẫn thư mục: ").strip()
        if not folder_path:
            folder_path = "Real Life Violence Dataset/Violence"
        
        print(f"\nĐang phân tích tất cả videos trong: {folder_path}")
        os.system(f'python scripts/predict.py --video_path "{folder_path}" --batch')
    
    elif choice == '3':
        print_header("Real-time Webcam Detection")
        print("\n⚠️  Đảm bảo model đã được train!")
        print("Controls: 'q' để thoát, 'r' để reset buffer\n")
        
        camera = input("Camera index (mặc định 0): ").strip()
        if not camera:
            camera = "0"
        
        print("\n🎥 Khởi động camera...")
        os.system(f'python scripts/realtime_detect.py --camera {camera}')
    
    elif choice == '4':
        print_header("Kết quả đánh giá model")
        print("\nĐang load kết quả evaluation...")
        
        # Check if evaluation results exist
        if os.path.exists('models/evaluation_results/metrics.txt'):
            with open('models/evaluation_results/metrics.txt', 'r') as f:
                print(f.read())
        else:
            print("⚠️  Chưa có kết quả đánh giá. Chạy evaluate.py trước!")
            run_eval = input("\nChạy evaluation ngay? (y/n): ").strip().lower()
            if run_eval == 'y':
                os.system('python scripts/evaluate.py')
    
    elif choice == '5':
        print("\n👋 Tạm biệt!")
        sys.exit(0)
    
    else:
        print("\n❌ Lựa chọn không hợp lệ!")
    
    # Ask to continue
    print("\n" + "-" * 70)
    continue_demo = input("Tiếp tục demo? (y/n): ").strip().lower()
    if continue_demo == 'y':
        main()
    else:
        print("\n👋 Cảm ơn bạn đã sử dụng!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo đã dừng!")
