"""
Video processing utilities.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def extract_frames_from_video(video_path: str, num_frames: int = 16,
                              frame_size: Tuple[int, int] = (112, 112)) -> Optional[np.ndarray]:
    """
    Extract a fixed number of frames from a video file.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        frame_size: Target size for frames (width, height)
        
    Returns:
        Numpy array of shape (num_frames, height, width, 3) or None if error
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return None
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < num_frames:
            print(f"Warning: Video has only {total_frames} frames, requested {num_frames}")
            # We'll duplicate some frames if needed
        
        # Calculate frame indices to extract (evenly spaced)
        if total_frames >= num_frames:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            # If video has fewer frames, repeat some frames
            frame_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
            # Pad with repeated frames
            while len(frame_indices) < num_frames:
                frame_indices = np.append(frame_indices, frame_indices[-1])
        
        frames = []
        current_frame_idx = 0
        
        for target_idx in frame_indices:
            # Seek to target frame
            while current_frame_idx < target_idx:
                ret = cap.grab()
                if not ret:
                    break
                current_frame_idx += 1
            
            # Read the frame
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame at index {target_idx}")
                break
            
            # Resize frame
            frame = cv2.resize(frame, frame_size)
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frames.append(frame)
            current_frame_idx += 1
        
        cap.release()
        
        # Convert to numpy array
        if len(frames) == num_frames:
            frames_array = np.array(frames, dtype=np.float32)
            # Normalize to [0, 1]
            frames_array = frames_array / 255.0
            return frames_array
        else:
            print(f"Could not extract {num_frames} frames from {video_path}")
            return None
            
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None


def get_video_info(video_path: str) -> dict:
    """
    Get video information.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return {}
    
    info = {
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


def save_frames_as_video(frames: np.ndarray, output_path: str, fps: int = 30):
    """
    Save frames as a video file.
    
    Args:
        frames: Numpy array of frames (num_frames, height, width, 3)
        output_path: Path to save the video
        fps: Frames per second
    """
    if len(frames) == 0:
        print("No frames to save")
        return
    
    # Denormalize if needed
    if frames.max() <= 1.0:
        frames = (frames * 255).astype(np.uint8)
    
    height, width = frames.shape[1:3]
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to: {output_path}")
