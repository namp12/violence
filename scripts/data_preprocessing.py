"""
Data preprocessing script for violence detection.
This script processes raw videos and creates train/val/test splits.
"""
import sys
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import get_config
from src.data.video_utils import extract_frames_from_video, get_video_info


def create_directories(config):
    """Create necessary directories for processed data."""
    processed_path = Path(config.data['processed_data_path'])
    
    # Create directories
    for split in ['train', 'val', 'test']:
        for category in ['violent', 'non_violent']:
            dir_path = processed_path / split / category
            dir_path.mkdir(parents=True, exist_ok=True)
    
    print("✓ Directories created")


def get_video_files(raw_data_path):
    """Get all video files from raw data directory."""
    raw_path = Path(raw_data_path)
    
    # Get violent videos
    violent_path = raw_path / 'Violence'
    violent_videos = []
    if violent_path.exists():
        violent_videos = list(violent_path.glob('*.mp4')) + \
                        list(violent_path.glob('*.avi')) + \
                        list(violent_path.glob('*.mov'))
    
    # Get non-violent videos
    non_violent_path = raw_path / 'NonViolence'
    non_violent_videos = []
    if non_violent_path.exists():
        non_violent_videos = list(non_violent_path.glob('*.mp4')) + \
                            list(non_violent_path.glob('*.avi')) + \
                            list(non_violent_path.glob('*.mov'))
    
    print(f"✓ Found {len(violent_videos)} violent videos")
    print(f"✓ Found {len(non_violent_videos)} non-violent videos")
    
    return violent_videos, non_violent_videos


def split_data(videos, train_ratio, val_ratio, test_ratio, random_seed=42):
    """Split videos into train, validation and test sets."""
    np.random.seed(random_seed)
    
    # Shuffle videos
    videos = list(videos)
    np.random.shuffle(videos)
    
    # Calculate split sizes
    total = len(videos)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Split
    train_videos = videos[:train_size]
    val_videos = videos[train_size:train_size + val_size]
    test_videos = videos[train_size + val_size:]
    
    return train_videos, val_videos, test_videos


def process_videos(videos, output_dir, category, num_frames, frame_size):
    """
    Process videos and save as numpy arrays.
    
    Args:
        videos: List of video paths
        output_dir: Output directory
        category: 'violent' or 'non_violent'
        num_frames: Number of frames to extract
        frame_size: Frame size (height, width)
    """
    output_path = Path(output_dir) / category
    output_path.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for i, video_path in enumerate(tqdm(videos, desc=f"Processing {category}")):
        try:
            # Extract frames
            frames = extract_frames_from_video(
                str(video_path),
                num_frames=num_frames,
                frame_size=frame_size
            )
            
            if frames is not None:
                # Save as numpy array
                output_file = output_path / f"{video_path.stem}_{i:04d}.npy"
                np.save(output_file, frames)
                successful += 1
            else:
                failed += 1
                print(f"Failed to process: {video_path.name}")
        
        except Exception as e:
            failed += 1
            print(f"Error processing {video_path.name}: {str(e)}")
    
    print(f"  ✓ Successfully processed: {successful}")
    print(f"  ✗ Failed: {failed}")
    
    return successful, failed


def main():
    """Main preprocessing function."""
    print("=" * 60)
    print("Violence Detection - Data Preprocessing")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/6] Loading configuration...")
    config = get_config()
    
    # Create directories
    print("\n[2/6] Creating directories...")
    create_directories(config)
    
    # Get video files
    print("\n[3/6] Scanning for videos...")
    raw_data_path = config.data['raw_data_path']
    violent_videos, non_violent_videos = get_video_files(raw_data_path)
    
    if len(violent_videos) == 0 or len(non_violent_videos) == 0:
        print("\n⚠ Error: No videos found!")
        print(f"Please ensure videos are in:")
        print(f"  - {raw_data_path}/Violence/")
        print(f"  - {raw_data_path}/NonViolence/")
        return
    
    # Split data
    print("\n[4/6] Splitting data...")
    split_config = config.split
    
    violent_train, violent_val, violent_test = split_data(
        violent_videos,
        split_config['train_ratio'],
        split_config['val_ratio'],
        split_config['test_ratio'],
        split_config['random_seed']
    )
    
    non_violent_train, non_violent_val, non_violent_test = split_data(
        non_violent_videos,
        split_config['train_ratio'],
        split_config['val_ratio'],
        split_config['test_ratio'],
        split_config['random_seed']
    )
    
    print(f"  Train: {len(violent_train)} violent, {len(non_violent_train)} non-violent")
    print(f"  Val:   {len(violent_val)} violent, {len(non_violent_val)} non-violent")
    print(f"  Test:  {len(violent_test)} violent, {len(non_violent_test)} non-violent")
    
    # Process videos
    print("\n[5/6] Processing videos...")
    
    video_config = config.video
    num_frames = video_config['num_frames']
    frame_size = (video_config['frame_height'], video_config['frame_width'])
    processed_path = config.data['processed_data_path']
    
    total_success = 0
    total_failed = 0
    
    # Process training set
    print("\n  Processing TRAIN set...")
    s, f = process_videos(violent_train, processed_path + '/train', 'violent', 
                         num_frames, frame_size)
    total_success += s
    total_failed += f
    
    s, f = process_videos(non_violent_train, processed_path + '/train', 'non_violent', 
                         num_frames, frame_size)
    total_success += s
    total_failed += f
    
    # Process validation set
    print("\n  Processing VAL set...")
    s, f = process_videos(violent_val, processed_path + '/val', 'violent', 
                         num_frames, frame_size)
    total_success += s
    total_failed += f
    
    s, f = process_videos(non_violent_val, processed_path + '/val', 'non_violent', 
                         num_frames, frame_size)
    total_success += s
    total_failed += f
    
    # Process test set
    print("\n  Processing TEST set...")
    s, f = process_videos(violent_test, processed_path + '/test', 'violent', 
                         num_frames, frame_size)
    total_success += s
    total_failed += f
    
    s, f = process_videos(non_violent_test, processed_path + '/test', 'non_violent', 
                         num_frames, frame_size)
    total_success += s
    total_failed += f
    
    # Summary
    print("\n[6/6] Preprocessing Complete!")
    print("=" * 60)
    print(f"Total videos processed: {total_success}")
    print(f"Total failures: {total_failed}")
    print(f"Success rate: {100 * total_success / (total_success + total_failed):.1f}%")
    print("=" * 60)
    print(f"\nProcessed data saved to: {processed_path}")
    print("\n✓ Ready for training!")


if __name__ == '__main__':
    main()
