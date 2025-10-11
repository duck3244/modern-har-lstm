"""
Modern Human Activity Recognition using LSTM with TensorFlow 2.0
Main execution script with modular architecture
"""

import tensorflow as tf
import numpy as np
import logging
import argparse
from pathlib import Path
import sys

# Import project modules
from config import Config, ConfigPresets
from data_loader import DataLoader, DataPreprocessor
from model import HARModel, ModelBuilder
from visualizer import Visualizer
from evaluator import ModelEvaluator, ErrorAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('har_training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Human Activity Recognition with LSTM')
    
    parser.add_argument('--config', choices=['default', 'quick', 'high_performance', 'lightweight'],
                       default='default', help='Configuration preset to use')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--lstm_units', type=int, help='Number of LSTM units')
    parser.add_argument('--model_type', choices=['standard', 'simple', 'deep', 'bidirectional'],
                       default='standard', help='Model architecture type')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save visualization plots to files')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and load existing model')
    parser.add_argument('--model_path', type=str, default='best_model.h5',
                       help='Path to save/load model')
    
    return parser.parse_args()


def setup_config(args):
    """Setup configuration based on arguments"""
    # Get base configuration
    if args.config == 'quick':
        config = ConfigPresets.get_quick_test_config()
    elif args.config == 'high_performance':
        config = ConfigPresets.get_high_performance_config()
    elif args.config == 'lightweight':
        config = ConfigPresets.get_lightweight_config()
    else:
        config = Config()
    
    # Override with command line arguments
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.lstm_units:
        config.lstm_units = args.lstm_units
    if args.model_path:
        config.model_save_path = args.model_path
    
    return config


def create_model(config, model_type='standard'):
    """Create model based on specified type"""
    if model_type == 'simple':
        model_keras = ModelBuilder.build_simple_lstm(config)
        har_model = HARModel(config)
        har_model.model = model_keras
        har_model.compile_model()
    elif model_type == 'deep':
        model_keras = ModelBuilder.build_deep_lstm(config, num_layers=3)
        har_model = HARModel(config)
        har_model.model = model_keras
        har_model.compile_model()
    elif model_type == 'bidirectional':
        model_keras = ModelBuilder.build_bidirectional_lstm(config)
        har_model = HARModel(config)
        har_model.model = model_keras
        har_model.compile_model()
    else:  # standard
        har_model = HARModel(config)
        har_model.compile_model()
    
    return har_model


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup configuration
    config = setup_config(args)
    config.create_directories()
    
    logger.info("🚀 Starting Human Activity Recognition Training")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Model type: {args.model_type}")
    
    # Initialize components
    data_loader = DataLoader(config)
    visualizer = Visualizer(config)
    evaluator = ModelEvaluator(config)
    error_analyzer = ErrorAnalyzer(config)
    
    try:
        # Load data
        logger.info("📁 Loading dataset...")
        data_loader.download_dataset()
        X_train, y_train, X_test, y_test = data_loader.load_data()
        
        # Optional: Apply data preprocessing
        logger.info("🔧 Preprocessing data...")
        preprocessor = DataPreprocessor()
        
        # Normalize data (optional - the UCI HAR dataset is already normalized)
        # X_train_norm, norm_params = preprocessor.normalize_data(X_train, method='standard')
        # X_test_norm = preprocessor.apply_normalization(X_test, norm_params)
        
        # Create and setup model
        logger.info(f"🏗️ Building {args.model_type} model...")
        model = create_model(config, args.model_type)
        
        # Training or loading existing model
        if args.skip_training and Path(config.model_save_path).exists():
            logger.info(f"📥 Loading existing model from {config.model_save_path}")
            model.load_model()
        else:
            logger.info("🚀 Starting training...")
            model.train(X_train, y_train)
            logger.info("✅ Training completed!")
        
        # Evaluation
        logger.info("📊 Evaluating model...")
        basic_results = model.evaluate(X_test, y_test)
        
        # Comprehensive evaluation
        comprehensive_results = evaluator.evaluate_model(
            basic_results['true_labels'],
            basic_results['predictions'], 
            basic_results['prediction_probabilities']
        )
        
        # Combine results
        all_results = {**basic_results, **comprehensive_results}
        
        # Generate and save evaluation reports
        model_params = config.get_model_params()
        performance_report = evaluator.generate_performance_report(all_results, model_params)
        
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(performance_report)
        
        # Error analysis
        error_report = error_analyzer.generate_error_report(
            X_test, basic_results['true_labels'],
            basic_results['predictions'], basic_results['prediction_probabilities']
        )
        print("\n" + error_report)
        
        # Save results
        logger.info("💾 Saving results...")
        evaluator.save_results(all_results, model_params)
        
        # Visualizations
        if model.history:
            logger.info("📈 Creating visualizations...")
            
            # Training history
            visualizer.plot_training_history(
                model.history.history,
                save_path="training_history.png" if args.save_visualizations else None
            )
            
            # Confusion matrix
            visualizer.plot_confusion_matrix(
                basic_results['true_labels'],
                basic_results['predictions'],
                normalize=True,
                save_path="confusion_matrix.png" if args.save_visualizations else None
            )
            
            # Classification report
            visualizer.plot_classification_report(
                basic_results['true_labels'],
                basic_results['predictions'],
                save_path="classification_report.png" if args.save_visualizations else None
            )
            
            # Class distribution
            visualizer.plot_class_distribution(
                basic_results['true_labels'],
                subset_name="Test Set",
                save_path="class_distribution.png" if args.save_visualizations else None
            )
            
            # Signal statistics
            visualizer.plot_signal_statistics(
                X_test,
                save_path="signal_statistics.png" if args.save_visualizations else None
            )
            
            # Prediction confidence
            visualizer.plot_prediction_confidence(
                basic_results['true_labels'],
                basic_results['prediction_probabilities'],
                save_path="prediction_confidence.png" if args.save_visualizations else None
            )
            
            # Sample signal visualization
            sample_idx = 0
            sample_label = config.labels[basic_results['true_labels'][sample_idx]]
            visualizer.plot_signal_data(
                X_test[sample_idx],
                activity_label=sample_label,
                save_path="sample_signals.png" if args.save_visualizations else None
            )
        
        # Create comprehensive visualization report if requested
        if args.save_visualizations:
            visualizer.create_comprehensive_report(basic_results, "visualizations")
        
        logger.info("🎉 Analysis completed successfully!")
        
        # Print final summary
        print(f"\n🎯 Final Test Accuracy: {basic_results['test_accuracy']:.4f} ({basic_results['test_accuracy']*100:.2f}%)")
        print(f"📉 Final Test Loss: {basic_results['test_loss']:.4f}")
        print(f"💾 Model saved to: {config.model_save_path}")
        
        # Show top confused class pairs
        confusion_pairs = comprehensive_results['misclassification_analysis']['most_confused_classes']
        if confusion_pairs:
            print(f"\n🔄 Most Common Confusions:")
            for pair, count in confusion_pairs[:3]:
                print(f"   {pair}: {count} cases")
        
    except Exception as e:
        logger.error(f"❌ Error during execution: {e}")
        raise
    
    finally:
        # Cleanup
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()