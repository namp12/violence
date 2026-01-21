"""
Prediction script for violence detection on new videos.
"""
import sys
import os
from pathlib import Path
import argparse
import numpy as np
from tensorflow import keras

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import get_config
from src.data.video_utils import extract_frames_from_video


def predict_video(video_path, model_path, config_path='config.yaml', verbose=True):
    """
    Predict violence in a video.
    
    Args:
        video_path: Path to the video file
        model_path: Path to the trained model
        config_path: Path to configuration file
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (predicted_class, confidence, class_name)
    """
    # Load configuration
    config = get_config(config_path)
    class_names = config.get('prediction.class_names', ['Non-Violent', 'Violent'])
    
    # Load model
    if verbose:
        print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None, None, None
    
    model = keras.models.load_model(model_path)
    
    if verbose:
        print("✓ Model loaded")
    
    # Extract frames from video
    if verbose:
        print(f"Processing video: {video_path}")
    
    video_config = config.video
    frames = extract_frames_from_video(
        video_path,
        num_frames=video_config['num_frames'],
        frame_size=(video_config['frame_height'], video_config['frame_width'])
    )
    
    if frames is None:
        print(f"Error: Could not process video {video_path}")
        return None, None, None
    
    if verbose:
        print(f"✓ Extracted {len(frames)} frames")
    
    # Make prediction
    if verbose:
        print("Making prediction...")
    
    # Add batch dimension
    frames_batch = np.expand_dims(frames, axis=0)
    
    # Predict
    prediction = model.predict(frames_batch, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    class_name = class_names[predicted_class]
    
    return predicted_class, confidence, class_name


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Predict violence in video')
    parser.add_argument('--video_path', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--model_path', type=str,
                       default='models/saved_models/violence_detection_final.h5',
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple videos from a directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Violence Detection - Prediction")
    print("=" * 70)
    
    if args.batch:
        # Process all videos in directory
        video_dir = Path(args.video_path)
        if not video_dir.is_dir():
            print(f"Error: {args.video_path} is not a directory")
            return
        
        video_files = list(video_dir.glob('*.mp4')) + \
                     list(video_dir.glob('*.avi')) + \
                     list(video_dir.glob('*.mov'))
        
        print(f"\nFound {len(video_files)} videos to process\n")
        
        results = []
        for video_file in video_files:
            predicted_class, confidence, class_name = predict_video(
                str(video_file),
                args.model_path,
                args.config,
                verbose=False
            )
            
            if predicted_class is not None:
                results.append({
                    'video': video_file.name,
                    'prediction': class_name,
                    'confidence': confidence
                })
                print(f"✓ {video_file.name}: {class_name} ({confidence:.2%})")
            else:
                print(f"✗ {video_file.name}: Failed to process")
        
        # Summary
        print("\n" + "=" * 70)
        print(f"Processed {len(results)}/{len(video_files)} videos")
        
        violent_count = sum(1 for r in results if r['prediction'] == 'Violent')
        print(f"Violent: {violent_count}")
        print(f"Non-Violent: {len(results) - violent_count}")
        print("=" * 70)
        
    else:
        # Process single video
        if not os.path.exists(args.video_path):
            print(f"Error: Video not found at {args.video_path}")
            return
        
        predicted_class, confidence, class_name = predict_video(
            args.video_path,
            args.model_path,
            args.config,
            verbose=True
        )
        
        if predicted_class is not None:
            print("\n" + "=" * 70)
            print("PREDICTION RESULT")
            print("=" * 70)
            print(f"Video: {args.video_path}")
            print(f"Prediction: {class_name}")
            print(f"Confidence: {confidence:.2%}")
            print("=" * 70)
            
            # Visual indicator
            print("\n" + "🔴" * 30 if class_name == 'Violent' else "🟢" * 30)
            print(f"  {class_name.upper()} DETECTED" if class_name == 'Violent' 
                  else f"  {class_name.upper()}")
            print("🔴" * 30 if class_name == 'Violent' else "🟢" * 30)


if __name__ == '__main__':
    main()
