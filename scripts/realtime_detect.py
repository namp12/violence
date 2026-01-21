"""
Real-time violence detection using webcam.
Captures video from camera and detects violence in real-time.
"""
import cv2
import numpy as np
from collections import deque
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import get_config
from tensorflow import keras


class RealtimeViolenceDetector:
    """Real-time violence detector using webcam."""
    
    def __init__(self, model_path, config_path='config.yaml', buffer_size=16):
        """
        Initialize real-time detector.
        
        Args:
            model_path: Path to trained model
            config_path: Path to config file
            buffer_size: Number of frames to buffer (should match training)
        """
        self.config = get_config(config_path)
        self.buffer_size = buffer_size
        self.frame_height = self.config.video['frame_height']
        self.frame_width = self.config.video['frame_width']
        self.class_names = self.config.get('prediction.class_names', ['Non-Violent', 'Violent'])
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
        
        # Frame buffer to store recent frames
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Detection results
        self.current_prediction = None
        self.current_confidence = 0.0
        
    def preprocess_frame(self, frame):
        """
        Preprocess a single frame.
        
        Args:
            frame: OpenCV frame (BGR)
            
        Returns:
            Preprocessed frame
        """
        # Resize
        frame_resized = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        return frame_normalized
    
    def predict(self):
        """
        Make prediction on current buffer.
        
        Returns:
            Tuple of (class_name, confidence)
        """
        if len(self.frame_buffer) < self.buffer_size:
            return None, 0.0
        
        # Convert buffer to numpy array
        frames = np.array(list(self.frame_buffer))
        
        # Add batch dimension
        frames_batch = np.expand_dims(frames, axis=0)
        
        # Predict
        prediction = self.model.predict(frames_batch, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        class_name = self.class_names[predicted_class]
        
        return class_name, float(confidence)
    
    def draw_results(self, frame, prediction, confidence):
        """
        Draw prediction results on frame.
        
        Args:
            frame: Original frame
            prediction: Predicted class name
            confidence: Confidence score
            
        Returns:
            Frame with results drawn
        """
        height, width = frame.shape[:2]
        
        # Determine color based on prediction
        if prediction == 'Violent':
            color = (0, 0, 255)  # Red for violent
            bg_color = (0, 0, 200)
        else:
            color = (0, 255, 0)  # Green for non-violent
            bg_color = (0, 200, 0)
        
        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), bg_color, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw prediction text
        cv2.putText(frame, f"Status: {prediction}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw buffer status
        buffer_status = f"Buffer: {len(self.frame_buffer)}/{self.buffer_size}"
        cv2.putText(frame, buffer_status, (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw warning if violent
        if prediction == 'Violent' and confidence > 0.7:
            warning_text = "!!! VIOLENCE DETECTED !!!"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
            text_x = (width - text_size[0]) // 2
            
            # Blinking effect
            import time
            if int(time.time() * 2) % 2 == 0:
                cv2.putText(frame, warning_text, (text_x, height - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        
        return frame
    
    def run(self, camera_index=0, skip_frames=2):
        """
        Run real-time detection.
        
        Args:
            camera_index: Camera device index (0 for default webcam)
            skip_frames: Process every N frames (to improve performance)
        """
        # Open camera
        print(f"Opening camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("✓ Camera opened successfully")
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 'r' to reset buffer")
        print("\nStarting detection...\n")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                frame_count += 1
                
                # Process every skip_frames
                if frame_count % skip_frames == 0:
                    # Preprocess and add to buffer
                    preprocessed = self.preprocess_frame(frame)
                    self.frame_buffer.append(preprocessed)
                    
                    # Make prediction if buffer is full
                    if len(self.frame_buffer) == self.buffer_size:
                        self.current_prediction, self.current_confidence = self.predict()
                
                # Draw results on frame
                if self.current_prediction is not None:
                    frame = self.draw_results(frame, self.current_prediction, 
                                            self.current_confidence)
                else:
                    cv2.putText(frame, "Buffering frames...", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Violence Detection - Real-time', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('r'):
                    print("Resetting buffer...")
                    self.frame_buffer.clear()
                    self.current_prediction = None
                    self.current_confidence = 0.0
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("✓ Camera released")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Real-time violence detection using webcam')
    parser.add_argument('--model_path', type=str,
                       default='models/checkpoints/best_model.h5',
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--skip_frames', type=int, default=2,
                       help='Process every N frames (default: 2)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Real-time Violence Detection")
    print("=" * 70)
    
    # Create detector
    detector = RealtimeViolenceDetector(
        model_path=args.model_path,
        config_path=args.config
    )
    
    # Run detection
    detector.run(camera_index=args.camera, skip_frames=args.skip_frames)


if __name__ == '__main__':
    main()
