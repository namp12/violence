"""
Video upload and prediction handler.
"""
import os
import sys
from pathlib import Path
from werkzeug.utils import secure_filename
import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.video_utils import extract_frames_from_video
from src.utils.config import get_config


class VideoHandler:
    """Handle video upload and prediction."""
    
    def __init__(self, model, config, upload_folder):
        """
        Initialize video handler.
        
        Args:
            model: Loaded Keras model
            config: Config object
            upload_folder: Path to upload directory
        """
        self.model = model
        self.config = config
        self.upload_folder = Path(upload_folder)
        self.class_names = config.get('prediction.class_names', ['Non-Violent', 'Violent'])
    
    def save_upload(self, file):
        """
        Save uploaded file.
        
        Args:
            file: FileStorage object from Flask
            
        Returns:
            Path to saved file or None if error
        """
        try:
            filename = secure_filename(file.filename)
            filepath = self.upload_folder / filename
            
            # Add timestamp if file exists
            if filepath.exists():
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                name, ext = filename.rsplit('.', 1)
                filename = f"{name}_{timestamp}.{ext}"
                filepath = self.upload_folder / filename
            
            file.save(str(filepath))
            return filepath
        
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            return None
    
    def predict_video(self, video_path):
        """
        Predict violence in video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with prediction results or None if error
        """
        try:
            # Extract frames
            video_config = self.config.video
            frames = extract_frames_from_video(
                str(video_path),
                num_frames=video_config['num_frames'],
                frame_size=(video_config['frame_height'], video_config['frame_width'])
            )
            
            if frames is None:
                return None
            
            # Add batch dimension
            frames_batch = np.expand_dims(frames, axis=0)
            
            # Predict
            prediction = self.model.predict(frames_batch, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
            class_name = self.class_names[predicted_class]
            
            return {
                'prediction': class_name,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.2f}%",
                'is_violent': class_name == 'Violent'
            }
        
        except Exception as e:
            print(f"Error predicting video: {str(e)}")
            return None
