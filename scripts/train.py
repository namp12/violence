"""
Training script for violence detection model.
"""
import sys
import os
from pathlib import Path
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import get_config
from src.models.cnn3d import create_and_compile_model
from src.data.dataset import create_data_generators
from src.utils.metrics import plot_training_history


def setup_callbacks(config):
    """Setup training callbacks."""
    callbacks = []
    callback_config = config.training.get('callbacks', {})
    
    # Model checkpoint
    if callback_config.get('model_checkpoint', {}).get('enabled', True):
        checkpoint_path = callback_config['model_checkpoint']['filepath']
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=callback_config['model_checkpoint'].get('monitor', 'val_accuracy'),
            save_best_only=callback_config['model_checkpoint'].get('save_best_only', True),
            verbose=1
        )
        callbacks.append(checkpoint)
        print(f"  ✓ ModelCheckpoint: {checkpoint_path}")
    
    # Early stopping
    if callback_config.get('early_stopping', {}).get('enabled', True):
        early_stop = keras.callbacks.EarlyStopping(
            monitor=callback_config['early_stopping'].get('monitor', 'val_loss'),
            patience=callback_config['early_stopping'].get('patience', 10),
            restore_best_weights=callback_config['early_stopping'].get('restore_best_weights', True),
            verbose=1
        )
        callbacks.append(early_stop)
        print(f"  ✓ EarlyStopping: patience={callback_config['early_stopping'].get('patience', 10)}")
    
    # Reduce learning rate
    if callback_config.get('reduce_lr', {}).get('enabled', True):
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor=callback_config['reduce_lr'].get('monitor', 'val_loss'),
            factor=callback_config['reduce_lr'].get('factor', 0.5),
            patience=callback_config['reduce_lr'].get('patience', 5),
            min_lr=callback_config['reduce_lr'].get('min_lr', 0.00001),
            verbose=1
        )
        callbacks.append(reduce_lr)
        print(f"  ✓ ReduceLROnPlateau: factor={callback_config['reduce_lr'].get('factor', 0.5)}")
    
    # TensorBoard
    if callback_config.get('tensorboard', {}).get('enabled', True):
        log_dir = callback_config['tensorboard'].get('log_dir', 'models/logs')
        log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        print(f"  ✓ TensorBoard: {log_dir}")
    
    return callbacks


def train(config_path='config.yaml', epochs=None, batch_size=None):
    """
    Train the violence detection model.
    
    Args:
        config_path: Path to configuration file
        epochs: Number of epochs (overrides config if provided)
        batch_size: Batch size (overrides config if provided)
    """
    print("=" * 70)
    print("Violence Detection - Model Training")
    print("=" * 70)
    
    # Load configuration
    print("\n[1/7] Loading configuration...")
    config = get_config(config_path)
    
    # Override config if arguments provided
    if epochs is not None:
        config.config['training']['epochs'] = epochs
    if batch_size is not None:
        config.config['training']['batch_size'] = batch_size
    
    training_config = config.training
    print(f"  Epochs: {training_config['epochs']}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Learning rate: {training_config['initial_learning_rate']}")
    
    # Check GPU availability
    print("\n[2/7] Checking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  ✓ Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            print(f"    - {gpu.name}")
    else:
        print("  ⚠ No GPU found, using CPU (training will be slower)")
    
    # Create data generators
    print("\n[3/7] Loading datasets...")
    processed_path = config.data['processed_data_path']
    
    train_dir = os.path.join(processed_path, 'train')
    val_dir = os.path.join(processed_path, 'val')
    test_dir = os.path.join(processed_path, 'test')
    
    # Check if processed data exists
    if not os.path.exists(train_dir):
        print(f"\n  ✗ Error: Processed data not found at {processed_path}")
        print("  Please run data_preprocessing.py first!")
        return
    
    augmentation_enabled = config.augmentation.get('enabled', True)
    
    train_dataset, val_dataset, test_dataset = create_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=training_config['batch_size'],
        augmentation=augmentation_enabled
    )
    
    print(f"  ✓ Train samples: {train_dataset.get_dataset_size()}")
    print(f"  ✓ Val samples: {val_dataset.get_dataset_size()}")
    print(f"  ✓ Test samples: {test_dataset.get_dataset_size()}")
    print(f"  ✓ Data augmentation: {'Enabled' if augmentation_enabled else 'Disabled'}")
    
    # Create model
    print("\n[4/7] Creating model...")
    model = create_and_compile_model(config.config)
    
    print(f"  ✓ Model created: {model.name}")
    print(f"  ✓ Total parameters: {model.count_params():,}")
    
    # Model summary
    print("\n  Model Architecture:")
    model.summary()
    
    # Setup callbacks
    print("\n[5/7] Setting up callbacks...")
    callbacks = setup_callbacks(config)
    
    # Train model
    print("\n[6/7] Training model...")
    print("=" * 70)
    
    history = model.fit(
        x=np.array([train_dataset.get_batch(i)[0] for i in range(len(train_dataset))]).reshape(-1, *config.model['input_shape']),
        y=np.array([train_dataset.get_batch(i)[1] for i in range(len(train_dataset))]).reshape(-1, 2),
        validation_data=(
            np.array([val_dataset.get_batch(i)[0] for i in range(len(val_dataset))]).reshape(-1, *config.model['input_shape']),
            np.array([val_dataset.get_batch(i)[1] for i in range(len(val_dataset))]).reshape(-1, 2)
        ),
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print("\n[7/7] Saving final model...")
    final_model_dir = 'models/saved_models'
    os.makedirs(final_model_dir, exist_ok=True)
    
    final_model_path = os.path.join(final_model_dir, 'violence_detection_final.h5')
    model.save(final_model_path)
    print(f"  ✓ Model saved to: {final_model_path}")
    
    # Plot training history
    history_plot_path = os.path.join(final_model_dir, 'training_history.png')
    plot_training_history(history, save_path=history_plot_path)
    
    # Print final results
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print("=" * 70)
    
    print(f"\n✓ Model saved to: {final_model_path}")
    print(f"✓ Training history plot: {history_plot_path}")
    print("\nNext steps:")
    print("  1. Run evaluate.py to test the model")
    print("  2. Use predict.py to make predictions on new videos")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train violence detection model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    
    args = parser.parse_args()
    
    train(config_path=args.config, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
