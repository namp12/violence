"""
Evaluation script for violence detection model.
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
from src.data.dataset import ViolenceDataset
from src.utils.metrics import (calculate_metrics, plot_confusion_matrix,
                               print_classification_report)


def evaluate_model(model_path, config_path='config.yaml', save_results=True):
    """
    Evaluate the trained model on test set.
    
    Args:
        model_path: Path to the trained model
        config_path: Path to configuration file
        save_results: Whether to save evaluation results
    """
    print("=" * 70)
    print("Violence Detection - Model Evaluation")
    print("=" * 70)
    
    # Load configuration
    print("\n[1/5] Loading configuration...")
    config = get_config(config_path)
    class_names = config.get('prediction.class_names', ['Non-Violent', 'Violent'])
    print(f"  Classes: {class_names}")
    
    # Load model
    print(f"\n[2/5] Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"  ✗ Error: Model not found at {model_path}")
        return
    
    model = keras.models.load_model(model_path)
    print(f"  ✓ Model loaded successfully")
    print(f"  ✓ Total parameters: {model.count_params():,}")
    
    # Load test dataset
    print("\n[3/5] Loading test dataset...")
    processed_path = config.data['processed_data_path']
    test_dir = os.path.join(processed_path, 'test')
    
    if not os.path.exists(test_dir):
        print(f"  ✗ Error: Test data not found at {test_dir}")
        print("  Please run data_preprocessing.py first!")
        return
    
    test_dataset = ViolenceDataset(
        test_dir,
        batch_size=config.training.get('batch_size', 8),
        augmentation=False,
        shuffle=False
    )
    
    print(f"  ✓ Test samples: {test_dataset.get_dataset_size()}")
    
    # Make predictions
    print("\n[4/5] Making predictions...")
    all_predictions = []
    all_labels = []
    
    for i in range(len(test_dataset)):
        batch_frames, batch_labels = test_dataset.get_batch(i)
        
        # Predict
        predictions = model.predict(batch_frames, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(batch_labels, axis=1)
        
        all_predictions.extend(predicted_classes)
        all_labels.extend(true_classes)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_dataset)} batches")
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    print(f"  ✓ Predictions complete: {len(all_predictions)} samples")
    
    # Calculate metrics
    print("\n[5/5] Calculating metrics...")
    metrics = calculate_metrics(all_labels, all_predictions, class_names)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Overall Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1-Score:          {metrics['f1_score']:.4f}")
    print("=" * 70)
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    print("-" * 70)
    for class_name in class_names:
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics.get(f'{class_name}_precision', 0):.4f}")
        print(f"  Recall:    {metrics.get(f'{class_name}_recall', 0):.4f}")
        print(f"  F1-Score:  {metrics.get(f'{class_name}_f1', 0):.4f}")
    
    # Classification report
    print_classification_report(all_labels, all_predictions, class_names)
    
    # Save results if requested
    if save_results:
        results_dir = config.get('evaluation.results_dir', 'models/evaluation_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save confusion matrix
        cm_path = os.path.join(results_dir, 'confusion_matrix.png')
        plot_confusion_matrix(all_labels, all_predictions, class_names, save_path=cm_path)
        
        # Save classification report
        report_path = os.path.join(results_dir, 'classification_report.txt')
        print_classification_report(all_labels, all_predictions, class_names, save_path=report_path)
        
        # Save metrics to file
        metrics_path = os.path.join(results_dir, 'metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 70 + "\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Overall Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"Precision:         {metrics['precision']:.4f}\n")
            f.write(f"Recall:            {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:          {metrics['f1_score']:.4f}\n")
            f.write("=" * 70 + "\n")
        
        print(f"\n✓ Results saved to: {results_dir}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Evaluate violence detection model')
    parser.add_argument('--model_path', type=str, 
                       default='models/saved_models/violence_detection_final.h5',
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.config, save_results=not args.no_save)


if __name__ == '__main__':
    main()
