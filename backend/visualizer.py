"""
Visualization utilities for Human Activity Recognition project
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, Any, List, Optional
import logging

from .config import Config

logger = logging.getLogger(__name__)


class Visualizer:
    """Visualization utilities for training and results"""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Setup matplotlib style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Set default figure parameters
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                             save_path: Optional[str] = None) -> None:
        """
        Plot training history
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot accuracy
        axes[0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            axes[0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        
        axes[0].set_title('Model Accuracy Over Time')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Plot loss
        axes[1].plot(history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        
        axes[1].set_title('Model Loss Over Time')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            normalize: bool = True, save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the plot (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
            cmap = 'Blues'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
            cmap = 'Blues'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap=cmap,
            xticklabels=self.config.labels,
            yticklabels=self.config.labels,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot classification report as heatmap
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot (optional)
        """
        # Get classification report as dictionary
        report = classification_report(
            y_true, y_pred, 
            target_names=self.config.labels, 
            output_dict=True
        )
        
        # Extract metrics for each class
        metrics_data = []
        for label in self.config.labels:
            if label in report:
                metrics_data.append([
                    report[label]['precision'],
                    report[label]['recall'],
                    report[label]['f1-score']
                ])
        
        metrics_df = pd.DataFrame(
            metrics_data,
            index=self.config.labels,
            columns=['Precision', 'Recall', 'F1-Score']
        )
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            metrics_df, 
            annot=True, 
            fmt='.3f', 
            cmap='YlOrRd',
            cbar_kws={'label': 'Score'},
            vmin=0,
            vmax=1
        )
        
        plt.title('Classification Metrics by Activity', fontsize=16, pad=20)
        plt.xlabel('Metrics', fontsize=14)
        plt.ylabel('Activity', fontsize=14)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Classification report saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_signal_data(self, X_sample: np.ndarray, activity_label: str = None,
                        save_path: Optional[str] = None) -> None:
        """
        Plot signal data for a single sample
        
        Args:
            X_sample: Signal data of shape (n_timesteps, n_features)
            activity_label: Activity label for the sample
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, signal_name in enumerate(self.config.signal_types):
            axes[i].plot(X_sample[:, i], linewidth=2)
            axes[i].set_title(f'{signal_name.replace("_", " ").title()}')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Normalized Value')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, len(X_sample))
        
        title = f'Signal Data'
        if activity_label:
            title += f' - {activity_label}'
        
        fig.suptitle(title, fontsize=16, y=0.98)
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Signal plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_class_distribution(self, y: np.ndarray, subset_name: str = "Dataset",
                              save_path: Optional[str] = None) -> None:
        """
        Plot class distribution
        
        Args:
            y: Labels array
            subset_name: Name of the dataset subset
            save_path: Path to save the plot (optional)
        """
        unique, counts = np.unique(y, return_counts=True)
        percentages = counts / len(y) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        bars = ax1.bar(range(len(unique)), counts, color=sns.color_palette("husl", len(unique)))
        ax1.set_title(f'{subset_name} - Class Distribution (Counts)')
        ax1.set_xlabel('Activity')
        ax1.set_ylabel('Number of Samples')
        ax1.set_xticks(range(len(unique)))
        ax1.set_xticklabels([self.config.labels[i] for i in unique], rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(counts, labels=[self.config.labels[i] for i in unique], autopct='%1.1f%%',
               colors=sns.color_palette("husl", len(unique)))
        ax2.set_title(f'{subset_name} - Class Distribution (Percentages)')
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_signal_statistics(self, X: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        Plot statistical properties of signals
        
        Args:
            X: Signal data of shape (n_samples, n_timesteps, n_features)
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Mean values
        means = np.mean(X, axis=(0, 1))
        bars1 = axes[0, 0].bar(range(len(self.config.signal_types)), means)
        axes[0, 0].set_title('Mean Signal Values')
        axes[0, 0].set_ylabel('Mean Value')
        axes[0, 0].set_xticks(range(len(self.config.signal_types)))
        axes[0, 0].set_xticklabels([s.replace('_', '\n') for s in self.config.signal_types], 
                                  rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Standard deviations
        stds = np.std(X, axis=(0, 1))
        bars2 = axes[0, 1].bar(range(len(self.config.signal_types)), stds)
        axes[0, 1].set_title('Signal Standard Deviations')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].set_xticks(range(len(self.config.signal_types)))
        axes[0, 1].set_xticklabels([s.replace('_', '\n') for s in self.config.signal_types], 
                                  rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Signal correlation matrix
        # Reshape data for correlation calculation
        X_reshaped = X.reshape(-1, X.shape[-1])
        correlation_matrix = np.corrcoef(X_reshaped.T)
        
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            fmt='.2f', 
            xticklabels=[s.replace('_', '\n') for s in self.config.signal_types],
            yticklabels=[s.replace('_', '\n') for s in self.config.signal_types],
            ax=axes[1, 0],
            cmap='coolwarm',
            center=0
        )
        axes[1, 0].set_title('Signal Correlation Matrix')
        
        # Signal variance over time
        time_variance = np.var(X, axis=0)
        im = axes[1, 1].imshow(time_variance.T, aspect='auto', cmap='viridis')
        axes[1, 1].set_title('Signal Variance Over Time')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Signal Type')
        axes[1, 1].set_yticks(range(len(self.config.signal_types)))
        axes[1, 1].set_yticklabels([s.replace('_', '\n') for s in self.config.signal_types])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 1])
        cbar.set_label('Variance')
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Signal statistics plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_prediction_confidence(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot prediction confidence distribution
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            save_path: Path to save the plot (optional)
        """
        y_pred = np.argmax(y_pred_proba, axis=1)
        confidences = np.max(y_pred_proba, axis=1)
        correct_predictions = (y_true == y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confidence distribution for correct vs incorrect predictions
        axes[0].hist(confidences[correct_predictions], bins=30, alpha=0.7, 
                    label='Correct Predictions', color='green', density=True)
        axes[0].hist(confidences[~correct_predictions], bins=30, alpha=0.7, 
                    label='Incorrect Predictions', color='red', density=True)
        axes[0].set_title('Prediction Confidence Distribution')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Confidence by class
        confidence_by_class = []
        for i, label in enumerate(self.config.labels):
            class_mask = (y_true == i)
            if np.any(class_mask):
                class_confidences = confidences[class_mask]
                confidence_by_class.append(class_confidences)
            else:
                confidence_by_class.append([])
        
        axes[1].boxplot(confidence_by_class, labels=self.config.labels)
        axes[1].set_title('Prediction Confidence by Class')
        axes[1].set_xlabel('Activity')
        axes[1].set_ylabel('Confidence Score')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction confidence plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def create_comprehensive_report(self, results: Dict[str, Any], 
                                  save_dir: str = "visualizations") -> None:
        """
        Create a comprehensive visualization report
        
        Args:
            results: Results dictionary from model evaluation
            save_dir: Directory to save visualizations
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Creating comprehensive visualization report in {save_dir}")
        
        y_true = results['true_labels']
        y_pred = results['predictions']
        y_pred_proba = results['prediction_probabilities']
        
        # Plot confusion matrix
        self.plot_confusion_matrix(
            y_true, y_pred, normalize=True,
            save_path=f"{save_dir}/confusion_matrix_normalized.png"
        )
        
        self.plot_confusion_matrix(
            y_true, y_pred, normalize=False,
            save_path=f"{save_dir}/confusion_matrix_counts.png"
        )
        
        # Plot classification report
        self.plot_classification_report(
            y_true, y_pred,
            save_path=f"{save_dir}/classification_report.png"
        )
        
        # Plot prediction confidence
        self.plot_prediction_confidence(
            y_true, y_pred_proba,
            save_path=f"{save_dir}/prediction_confidence.png"
        )
        
        # Plot class distributions
        self.plot_class_distribution(
            y_true, "Test Set",
            save_path=f"{save_dir}/test_class_distribution.png"
        )
        
        logger.info("Comprehensive visualization report completed!")