"""
Model evaluation and analysis utilities for Human Activity Recognition
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, accuracy_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import datetime

from config import Config

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and analysis"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(self.config.n_classes)
        )
        
        results['per_class_metrics'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': support.tolist()
        }
        
        # Average metrics
        results['macro_avg'] = {
            'precision': np.mean(precision),
            'recall': np.mean(recall),
            'f1_score': np.mean(f1)
        }
        
        results['weighted_avg'] = {
            'precision': np.average(precision, weights=support),
            'recall': np.average(recall, weights=support),
            'f1_score': np.average(f1, weights=support)
        }
        
        # Classification report
        results['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.config.labels, output_dict=True
        )
        
        # Prediction confidence analysis
        results['confidence_analysis'] = self._analyze_prediction_confidence(
            y_true, y_pred, y_pred_proba
        )
        
        # Misclassification analysis
        results['misclassification_analysis'] = self._analyze_misclassifications(
            y_true, y_pred
        )
        
        # ROC AUC for multiclass (one-vs-rest)
        try:
            results['roc_auc'] = self._calculate_multiclass_roc_auc(y_true, y_pred_proba)
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
            results['roc_auc'] = None
        
        logger.info("Model evaluation completed")
        return results
    
    def _analyze_prediction_confidence(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction confidence statistics"""
        confidences = np.max(y_pred_proba, axis=1)
        correct_predictions = (y_true == y_pred)
        
        analysis = {
            'overall_confidence': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            },
            'correct_predictions_confidence': {
                'mean': float(np.mean(confidences[correct_predictions])),
                'std': float(np.std(confidences[correct_predictions]))
            },
            'incorrect_predictions_confidence': {
                'mean': float(np.mean(confidences[~correct_predictions])),
                'std': float(np.std(confidences[~correct_predictions]))
            }
        }
        
        # Confidence by class
        confidence_by_class = {}
        for i, label in enumerate(self.config.labels):
            class_mask = (y_true == i)
            if np.any(class_mask):
                class_confidences = confidences[class_mask]
                confidence_by_class[label] = {
                    'mean': float(np.mean(class_confidences)),
                    'std': float(np.std(class_confidences)),
                    'count': int(np.sum(class_mask))
                }
        
        analysis['confidence_by_class'] = confidence_by_class
        
        return analysis
    
    def _analyze_misclassifications(self, y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze misclassification patterns"""
        misclassified_indices = np.where(y_true != y_pred)[0]
        
        analysis = {
            'total_misclassified': int(len(misclassified_indices)),
            'misclassification_rate': float(len(misclassified_indices) / len(y_true)),
            'confusion_pairs': {},
            'most_confused_classes': []
        }
        
        # Analyze confusion pairs
        confusion_pairs = {}
        for true_label, pred_label in zip(y_true[misclassified_indices], 
                                        y_pred[misclassified_indices]):
            pair = f"{self.config.labels[true_label]} -> {self.config.labels[pred_label]}"
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        # Sort by frequency
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        analysis['confusion_pairs'] = dict(sorted_pairs)
        analysis['most_confused_classes'] = sorted_pairs[:5]  # Top 5
        
        # Per-class misclassification rates
        per_class_errors = {}
        for i, label in enumerate(self.config.labels):
            class_mask = (y_true == i)
            if np.any(class_mask):
                class_errors = np.sum((y_true == i) & (y_pred != i))
                total_class_samples = np.sum(class_mask)
                error_rate = class_errors / total_class_samples
                per_class_errors[label] = {
                    'error_count': int(class_errors),
                    'total_samples': int(total_class_samples),
                    'error_rate': float(error_rate)
                }
        
        analysis['per_class_errors'] = per_class_errors
        
        return analysis
    
    def _calculate_multiclass_roc_auc(self, y_true: np.ndarray, 
                                    y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate ROC AUC for multiclass classification"""
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(self.config.n_classes))
        
        # Calculate ROC AUC for each class
        roc_auc = {}
        for i, label in enumerate(self.config.labels):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc[label] = auc(fpr, tpr)
            except Exception:
                roc_auc[label] = 0.0
        
        # Calculate macro and micro averages
        roc_auc['macro_avg'] = np.mean(list(roc_auc.values()))
        
        return roc_auc
    
    def generate_performance_report(self, results: Dict[str, Any], 
                                  model_params: Dict[str, Any]) -> str:
        """
        Generate a comprehensive performance report
        
        Args:
            results: Evaluation results
            model_params: Model parameters
            
        Returns:
            Formatted performance report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("HUMAN ACTIVITY RECOGNITION - PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Model configuration
        report_lines.append("📋 MODEL CONFIGURATION:")
        report_lines.append("-" * 30)
        for key, value in model_params.items():
            report_lines.append(f"  {key}: {value}")
        report_lines.append("")
        
        # Overall performance
        report_lines.append("🎯 OVERALL PERFORMANCE:")
        report_lines.append("-" * 30)
        report_lines.append(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        report_lines.append(f"  Macro Avg Precision: {results['macro_avg']['precision']:.4f}")
        report_lines.append(f"  Macro Avg Recall: {results['macro_avg']['recall']:.4f}")
        report_lines.append(f"  Macro Avg F1-Score: {results['macro_avg']['f1_score']:.4f}")
        report_lines.append("")
        
        # Per-class performance
        report_lines.append("📊 PER-CLASS PERFORMANCE:")
        report_lines.append("-" * 30)
        per_class = results['per_class_metrics']
        for i, label in enumerate(self.config.labels):
            report_lines.append(f"  {label}:")
            report_lines.append(f"    Precision: {per_class['precision'][i]:.3f}")
            report_lines.append(f"    Recall: {per_class['recall'][i]:.3f}")
            report_lines.append(f"    F1-Score: {per_class['f1_score'][i]:.3f}")
            report_lines.append(f"    Support: {per_class['support'][i]}")
            report_lines.append("")
        
        # Confidence analysis
        conf_analysis = results['confidence_analysis']
        report_lines.append("🔍 CONFIDENCE ANALYSIS:")
        report_lines.append("-" * 30)
        report_lines.append(f"  Overall Mean Confidence: {conf_analysis['overall_confidence']['mean']:.3f}")
        report_lines.append(f"  Correct Predictions Confidence: {conf_analysis['correct_predictions_confidence']['mean']:.3f}")
        report_lines.append(f"  Incorrect Predictions Confidence: {conf_analysis['incorrect_predictions_confidence']['mean']:.3f}")
        report_lines.append("")
        
        # Misclassification analysis
        misc_analysis = results['misclassification_analysis']
        report_lines.append("❌ MISCLASSIFICATION ANALYSIS:")
        report_lines.append("-" * 30)
        report_lines.append(f"  Total Misclassified: {misc_analysis['total_misclassified']}")
        report_lines.append(f"  Misclassification Rate: {misc_analysis['misclassification_rate']:.4f}")
        report_lines.append("  Most Common Confusion Pairs:")
        for pair, count in list(misc_analysis['most_confused_classes'])[:3]:
            report_lines.append(f"    {pair}: {count} cases")
        report_lines.append("")
        
        # ROC AUC
        if results.get('roc_auc'):
            report_lines.append("📈 ROC AUC SCORES:")
            report_lines.append("-" * 30)
            for label, score in results['roc_auc'].items():
                if label != 'macro_avg':
                    report_lines.append(f"  {label}: {score:.3f}")
            report_lines.append(f"  Macro Average: {results['roc_auc']['macro_avg']:.3f}")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_results(self, results: Dict[str, Any], model_params: Dict[str, Any], 
                    save_dir: str = "results") -> None:
        """
        Save evaluation results to files
        
        Args:
            results: Evaluation results
            model_params: Model parameters
            save_dir: Directory to save results
        """
        Path(save_dir).mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        # Save detailed results as JSON
        results_file = Path(save_dir) / f"evaluation_results_{timestamp}.json"
        experiment_data = {
            'timestamp': timestamp,
            'model_parameters': model_params,
            'evaluation_results': serializable_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        # Save performance report as text
        report_file = Path(save_dir) / f"performance_report_{timestamp}.txt"
        report = self.generate_performance_report(results, model_params)
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save confusion matrix as CSV
        cm_file = Path(save_dir) / f"confusion_matrix_{timestamp}.csv"
        cm_df = pd.DataFrame(
            results['confusion_matrix'],
            index=self.config.labels,
            columns=self.config.labels
        )
        cm_df.to_csv(cm_file)
        
        logger.info(f"Results saved to {save_dir}")
        logger.info(f"  - Detailed results: {results_file}")
        logger.info(f"  - Performance report: {report_file}")
        logger.info(f"  - Confusion matrix: {cm_file}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def compare_models(self, results_list: List[Dict[str, Any]], 
                      model_names: List[str]) -> pd.DataFrame:
        """
        Compare multiple model evaluation results
        
        Args:
            results_list: List of evaluation results
            model_names: List of model names
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for results, name in zip(results_list, model_names):
            row = {
                'Model': name,
                'Accuracy': results['accuracy'],
                'Macro Precision': results['macro_avg']['precision'],
                'Macro Recall': results['macro_avg']['recall'],
                'Macro F1-Score': results['macro_avg']['f1_score'],
                'Weighted Precision': results['weighted_avg']['precision'],
                'Weighted Recall': results['weighted_avg']['recall'],
                'Weighted F1-Score': results['weighted_avg']['f1_score'],
                'Misclassification Rate': results['misclassification_analysis']['misclassification_rate'],
                'Mean Confidence': results['confidence_analysis']['overall_confidence']['mean']
            }
            
            # Add per-class F1 scores
            for i, label in enumerate(self.config.labels):
                row[f'{label} F1'] = results['per_class_metrics']['f1_score'][i]
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df


class ErrorAnalyzer:
    """Detailed error analysis utilities"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def analyze_difficult_samples(self, X_test: np.ndarray, y_true: np.ndarray, 
                                y_pred: np.ndarray, y_pred_proba: np.ndarray,
                                top_k: int = 10) -> Dict[str, Any]:
        """
        Analyze the most difficult samples (misclassified with high confidence)
        
        Args:
            X_test: Test data
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            top_k: Number of difficult samples to analyze
            
        Returns:
            Analysis of difficult samples
        """
        # Find misclassified samples
        misclassified_mask = (y_true != y_pred)
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            return {"message": "No misclassified samples found"}
        
        # Get confidence scores for misclassified samples
        misclassified_confidences = np.max(y_pred_proba[misclassified_indices], axis=1)
        
        # Sort by confidence (highest confidence errors are most interesting)
        sorted_indices = np.argsort(misclassified_confidences)[::-1]
        top_difficult_indices = misclassified_indices[sorted_indices[:top_k]]
        
        difficult_samples = []
        for idx in top_difficult_indices:
            sample_analysis = {
                'sample_index': int(idx),
                'true_label': self.config.labels[y_true[idx]],
                'predicted_label': self.config.labels[y_pred[idx]],
                'confidence': float(np.max(y_pred_proba[idx])),
                'prediction_probabilities': y_pred_proba[idx].tolist(),
                'signal_statistics': {
                    'mean': float(np.mean(X_test[idx])),
                    'std': float(np.std(X_test[idx])),
                    'min': float(np.min(X_test[idx])),
                    'max': float(np.max(X_test[idx]))
                }
            }
            difficult_samples.append(sample_analysis)
        
        return {
            'total_misclassified': len(misclassified_indices),
            'top_difficult_samples': difficult_samples,
            'analysis_summary': {
                'mean_confidence_of_errors': float(np.mean(misclassified_confidences)),
                'std_confidence_of_errors': float(np.std(misclassified_confidences)),
                'highest_confidence_error': float(np.max(misclassified_confidences))
            }
        }
    
    def analyze_class_specific_errors(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Analyze errors specific to each class
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Class-specific error analysis
        """
        class_analysis = {}
        
        for i, label in enumerate(self.config.labels):
            class_mask = (y_true == i)
            
            if not np.any(class_mask):
                continue
            
            class_true = y_true[class_mask]
            class_pred = y_pred[class_mask]
            class_proba = y_pred_proba[class_mask]
            
            # Calculate metrics for this class
            correct_predictions = (class_true == class_pred)
            accuracy = np.mean(correct_predictions)
            
            # Find what this class is most often confused with
            misclassified_mask = ~correct_predictions
            if np.any(misclassified_mask):
                confused_with = class_pred[misclassified_mask]
                confusion_counts = {}
                for confused_label in confused_with:
                    confused_label_name = self.config.labels[confused_label]
                    confusion_counts[confused_label_name] = confusion_counts.get(confused_label_name, 0) + 1
                
                most_confused_with = max(confusion_counts.items(), key=lambda x: x[1])
            else:
                confusion_counts = {}
                most_confused_with = ("None", 0)
            
            # Confidence analysis for this class
            class_confidences = np.max(class_proba, axis=1)
            correct_confidences = class_confidences[correct_predictions]
            incorrect_confidences = class_confidences[~correct_predictions]
            
            class_analysis[label] = {
                'total_samples': int(np.sum(class_mask)),
                'accuracy': float(accuracy),
                'correct_predictions': int(np.sum(correct_predictions)),
                'incorrect_predictions': int(np.sum(~correct_predictions)),
                'confusion_with_other_classes': confusion_counts,
                'most_confused_with': {
                    'class': most_confused_with[0],
                    'count': most_confused_with[1]
                },
                'confidence_stats': {
                    'overall_mean': float(np.mean(class_confidences)),
                    'overall_std': float(np.std(class_confidences)),
                    'correct_mean': float(np.mean(correct_confidences)) if len(correct_confidences) > 0 else 0.0,
                    'incorrect_mean': float(np.mean(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0.0
                }
            }
        
        return class_analysis
    
    def generate_error_report(self, X_test: np.ndarray, y_true: np.ndarray,
                            y_pred: np.ndarray, y_pred_proba: np.ndarray) -> str:
        """
        Generate a comprehensive error analysis report
        
        Args:
            X_test: Test data
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Formatted error analysis report
        """
        difficult_samples = self.analyze_difficult_samples(X_test, y_true, y_pred, y_pred_proba)
        class_errors = self.analyze_class_specific_errors(y_true, y_pred, y_pred_proba)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ERROR ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overall error statistics
        total_errors = np.sum(y_true != y_pred)
        error_rate = total_errors / len(y_true)
        
        report_lines.append("📊 OVERALL ERROR STATISTICS:")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Errors: {total_errors}")
        report_lines.append(f"Error Rate: {error_rate:.4f} ({error_rate*100:.2f}%)")
        report_lines.append("")
        
        # Class-specific errors
        report_lines.append("📋 CLASS-SPECIFIC ERROR ANALYSIS:")
        report_lines.append("-" * 40)
        
        for class_name, analysis in class_errors.items():
            report_lines.append(f"\n{class_name}:")
            report_lines.append(f"  Total Samples: {analysis['total_samples']}")
            report_lines.append(f"  Accuracy: {analysis['accuracy']:.4f}")
            report_lines.append(f"  Errors: {analysis['incorrect_predictions']}")
            
            if analysis['most_confused_with']['count'] > 0:
                report_lines.append(f"  Most Confused With: {analysis['most_confused_with']['class']} "
                                  f"({analysis['most_confused_with']['count']} times)")
            
            report_lines.append(f"  Confidence (Correct): {analysis['confidence_stats']['correct_mean']:.3f}")
            report_lines.append(f"  Confidence (Incorrect): {analysis['confidence_stats']['incorrect_mean']:.3f}")
        
        # Difficult samples
        if 'top_difficult_samples' in difficult_samples:
            report_lines.append(f"\n🔍 MOST DIFFICULT SAMPLES (High Confidence Errors):")
            report_lines.append("-" * 50)
            
            for i, sample in enumerate(difficult_samples['top_difficult_samples'][:5]):
                report_lines.append(f"\nSample {i+1} (Index: {sample['sample_index']}):")
                report_lines.append(f"  True: {sample['true_label']}")
                report_lines.append(f"  Predicted: {sample['predicted_label']}")
                report_lines.append(f"  Confidence: {sample['confidence']:.4f}")
        
        report_lines.append("\n" + "=" * 80)
        
        return "\n".join(report_lines)