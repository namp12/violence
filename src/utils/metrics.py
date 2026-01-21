"""
Custom metrics and evaluation utilities.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     class_names: list = None) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    # Per-class metrics
    if class_names is not None:
        precision_per_class, recall_per_class, f1_per_class, _ = \
            precision_recall_fscore_support(y_true, y_pred, average=None)
        
        for i, class_name in enumerate(class_names):
            metrics[f'{class_name}_precision'] = float(precision_per_class[i])
            metrics[f'{class_name}_recall'] = float(recall_per_class[i])
            metrics[f'{class_name}_f1'] = float(f1_per_class[i])
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: list, save_path: str = None,
                         figsize: Tuple[int, int] = (8, 6)):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the figure
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close()


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                class_names: list, save_path: str = None):
    """
    Print and optionally save classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the report
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print("=" * 60)
    print(report)
    print("=" * 60)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 60 + "\n")
            f.write(report)
            f.write("=" * 60 + "\n")
        print(f"\nClassification report saved to: {save_path}")


def plot_training_history(history, save_path: str = None,
                         figsize: Tuple[int, int] = (12, 4)):
    """
    Plot training history.
    
    Args:
        history: Keras History object
        save_path: Path to save the figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.close()
