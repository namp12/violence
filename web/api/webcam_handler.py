"""
Webcam streaming handler using WebSocket.
"""
import base64
import numpy as np
import cv2
from collections import deque
import sys
from pathlib import Path
import time

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class WebcamHandler:
    """Handle real-time webcam detection via WebSocket."""
    
    def __init__(self, model, config, buffer_size=16):
        """
        Initialize webcam handler.
        
        Args:
            model: Loaded Keras model
            config: Config object
            buffer_size: Number of frames to buffer
        """
        self.model = model
        self.config = config
        self.buffer_size = buffer_size
        
        # Get config
        video_config = config.video
        self.frame_height = video_config['frame_height']
        self.frame_width = video_config['frame_width']
        self.class_names = config.get('prediction.class_names', ['Non-Violent', 'Violent'])
        self.confidence_threshold = config.get('prediction.confidence_threshold', 0.7)
        
        # Frame buffers for each client (session_id → deque)
        self.buffers = {}
        
        # Alert tracking: session_id -> last_alert_time
        # This prevents spamming database/email for every violent frame
        self.last_alert_times = {}
        self.alert_cooldown = 5.0  # seconds between alerts for same session
    
    def get_buffer(self, session_id):
        """Get or create buffer for session."""
        if session_id not in self.buffers:
            self.buffers[session_id] = deque(maxlen=self.buffer_size)
        return self.buffers[session_id]
    
    def clear_buffer(self, session_id):
        """Clear buffer for session."""
        if session_id in self.buffers:
            self.buffers[session_id].clear()
    
    def remove_buffer(self, session_id):
        """Remove buffer for disconnected session."""
        if session_id in self.buffers:
            del self.buffers[session_id]
        if session_id in self.last_alert_times:
            del self.last_alert_times[session_id]
    
    def process_frame(self, frame_data, session_id):
        """
        Process a single frame from webcam.
        
        Args:
            frame_data: Base64 encoded frame data
            session_id: Client session ID
            
        Returns:
            Dict with prediction results or None if buffer not full
        """
        try:
            # Decode base64 frame
            frame_bytes = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return None
            
            # Preprocess frame
            frame_resized = cv2.resize(frame, (self.frame_width, self.frame_height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            
            # Add to buffer
            buffer = self.get_buffer(session_id)
            buffer.append(frame_normalized)
            
            # Predict if buffer full
            if len(buffer) == self.buffer_size:
                return self._predict_buffer(buffer)
            
            return None
        
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None
    
    def should_trigger_alert(self, session_id, is_violent, confidence):
        """
        Check if an alert should be triggered based on cooldown.
        
        Args:
            session_id: Client session ID
            is_violent: Whether current prediction is violent
            confidence: Confidence score
            
        Returns:
            True if alert should be triggered, False otherwise
        """
        # Only trigger alerts for violent predictions above threshold
        if not is_violent or confidence < self.confidence_threshold:
            return False
        
        current_time = time.time()
        last_alert = self.last_alert_times.get(session_id, 0)
        
        # Check if cooldown period has passed
        if current_time - last_alert >= self.alert_cooldown:
            self.last_alert_times[session_id] = current_time
            return True
        
        return False
    
    def _predict_buffer(self, buffer):
        """
        Make prediction on buffer.
        
        Args:
            buffer: Deque of frames
            
        Returns:
            Dict with prediction results
        """
        try:
            # Convert buffer to numpy array
            frames = np.array(list(buffer))
            
            # Add batch dimension
            frames_batch = np.expand_dims(frames, axis=0)
            
            # Predict
            prediction = self.model.predict(frames_batch, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
            class_name = self.class_names[predicted_class]
            
            result = {
                'prediction': class_name,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.2f}%",
                'is_violent': class_name == 'Violent',
                'buffer_size': len(buffer)
            }
            
            return result
        
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None
