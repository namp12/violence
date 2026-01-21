"""
Dataset loader for violence detection.
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional
import random


class ViolenceDataset:
    """Dataset class for loading and processing violence detection data."""
    
    def __init__(self, data_dir: str, batch_size: int = 8, 
                 augmentation: bool = False, shuffle: bool = True):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing processed data (violent/ and non_violent/)
            batch_size: Batch size for training
            augmentation: Whether to apply data augmentation
            shuffle: Whether to shuffle the dataset
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.shuffle = shuffle
        
        # Load data paths
        self.violent_files = sorted(list((self.data_dir / 'violent').glob('*.npy')))
        self.non_violent_files = sorted(list((self.data_dir / 'non_violent').glob('*.npy')))
        
        # Create labels
        self.violent_labels = [1] * len(self.violent_files)
        self.non_violent_labels = [0] * len(self.non_violent_files)
        
        # Combine
        self.all_files = self.violent_files + self.non_violent_files
        self.all_labels = self.violent_labels + self.non_violent_labels
        
        # Create dataset indices
        self.indices = list(range(len(self.all_files)))
        
        if self.shuffle:
            random.shuffle(self.indices)
        
        print(f"Dataset loaded: {len(self.violent_files)} violent, "
              f"{len(self.non_violent_files)} non-violent videos")
    
    def __len__(self):
        """Return number of batches."""
        return len(self.all_files) // self.batch_size
    
    def get_dataset_size(self):
        """Return total number of samples."""
        return len(self.all_files)
    
    def load_sample(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Load a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (frames, label)
        """
        file_path = self.all_files[idx]
        label = self.all_labels[idx]
        
        # Load frames
        frames = np.load(file_path)
        
        # Apply augmentation if needed
        if self.augmentation:
            frames = self.augment_frames(frames)
        
        return frames, label
    
    def augment_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to frames.
        
        Args:
            frames: Input frames (num_frames, height, width, channels)
            
        Returns:
            Augmented frames
        """
        # Random horizontal flip
        if random.random() > 0.5:
            frames = frames[:, :, ::-1, :]
        
        # Random brightness adjustment
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            frames = np.clip(frames * brightness_factor, 0, 1)
        
        # Random rotation (small angle)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            frames = self.rotate_frames(frames, angle)
        
        return frames
    
    def rotate_frames(self, frames: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate frames by a given angle.
        
        Args:
            frames: Input frames
            angle: Rotation angle in degrees
            
        Returns:
            Rotated frames
        """
        import cv2
        
        rotated_frames = []
        height, width = frames.shape[1:3]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        for frame in frames:
            # Denormalize for rotation
            frame_uint8 = (frame * 255).astype(np.uint8)
            rotated = cv2.warpAffine(frame_uint8, rotation_matrix, (width, height))
            # Normalize back
            rotated = rotated.astype(np.float32) / 255.0
            rotated_frames.append(rotated)
        
        return np.array(rotated_frames)
    
    def get_batch(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of data.
        
        Args:
            batch_idx: Batch index
            
        Returns:
            Tuple of (batch_frames, batch_labels)
        """
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.all_files))
        
        batch_frames = []
        batch_labels = []
        
        for i in range(start_idx, end_idx):
            idx = self.indices[i]
            frames, label = self.load_sample(idx)
            batch_frames.append(frames)
            batch_labels.append(label)
        
        # Convert to numpy arrays
        batch_frames = np.array(batch_frames)
        batch_labels = np.array(batch_labels)
        
        # Convert labels to categorical (one-hot encoding)
        batch_labels_categorical = tf.keras.utils.to_categorical(batch_labels, num_classes=2)
        
        return batch_frames, batch_labels_categorical
    
    def get_tf_dataset(self) -> tf.data.Dataset:
        """
        Create TensorFlow dataset.
        
        Returns:
            tf.data.Dataset
        """
        def generator():
            for i in range(len(self)):
                frames, labels = self.get_batch(i)
                for j in range(len(frames)):
                    yield frames[j], labels[j]
        
        # Get output signature
        sample_frames, sample_label = self.load_sample(0)
        sample_label_cat = tf.keras.utils.to_categorical([sample_label], num_classes=2)[0]
        
        output_signature = (
            tf.TensorSpec(shape=sample_frames.shape, dtype=tf.float32),
            tf.TensorSpec(shape=sample_label_cat.shape, dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def on_epoch_end(self):
        """Shuffle indices at the end of each epoch."""
        if self.shuffle:
            random.shuffle(self.indices)


def create_data_generators(train_dir: str, val_dir: str, test_dir: str,
                           batch_size: int = 8, augmentation: bool = True):
    """
    Create data generators for train, validation and test sets.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        test_dir: Test data directory
        batch_size: Batch size
        augmentation: Whether to use augmentation for training
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = ViolenceDataset(train_dir, batch_size=batch_size, 
                                   augmentation=augmentation, shuffle=True)
    val_dataset = ViolenceDataset(val_dir, batch_size=batch_size, 
                                 augmentation=False, shuffle=False)
    test_dataset = ViolenceDataset(test_dir, batch_size=batch_size, 
                                  augmentation=False, shuffle=False)
    
    return train_dataset, val_dataset, test_dataset
