"""
LSTM model definition and training utilities for Human Activity Recognition
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from .config import Config

logger = logging.getLogger(__name__)


class HARModel:
    """Human Activity Recognition LSTM Model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.history = None
        self.is_compiled = False
    
    def build_model(self) -> tf.keras.Model:
        """
        Build the LSTM model architecture
        
        Returns:
            Compiled Keras model
        """
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(
                shape=(self.config.n_timesteps, self.config.n_features),
                name='input_layer'
            ),
            
            # First LSTM layer (recurrent_dropout=0 to keep cuDNN kernel)
            tf.keras.layers.LSTM(
                self.config.lstm_units,
                return_sequences=True,
                dropout=self.config.dropout_rate,
                kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
                name='lstm_1'
            ),

            # Second LSTM layer
            tf.keras.layers.LSTM(
                self.config.lstm_units,
                dropout=self.config.dropout_rate,
                kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
                name='lstm_2'
            ),
            
            # Dense layer with ReLU activation
            tf.keras.layers.Dense(
                self.config.lstm_units,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
                name='dense_hidden'
            ),
            
            # Dropout for regularization
            tf.keras.layers.Dropout(self.config.dropout_rate, name='dropout'),
            
            # Output layer
            tf.keras.layers.Dense(
                self.config.n_classes, 
                activation='softmax',
                name='output_layer'
            )
        ])
        
        return model
    
    def compile_model(self) -> None:
        """Compile the model with optimizer, loss, and metrics"""
        if self.model is None:
            self.model = self.build_model()
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.is_compiled = True
        logger.info("Model compiled successfully!")
        self._log_model_summary()
    
    def _log_model_summary(self) -> None:
        """Log model architecture summary"""
        logger.info("Model Architecture:")
        logger.info(f"  Input shape: ({self.config.n_timesteps}, {self.config.n_features})")
        logger.info(f"  LSTM units: {self.config.lstm_units}")
        logger.info(f"  Dropout rate: {self.config.dropout_rate}")
        logger.info(f"  L2 regularization: {self.config.l2_reg}")
        logger.info(f"  Output classes: {self.config.n_classes}")
        
        # Count parameters (compatible with TensorFlow 2.13+)
        total_params = self.model.count_params()
        trainable_params = sum([np.prod(w.shape.as_list()) for w in self.model.trainable_weights])
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    def get_callbacks(self) -> list:
        """
        Get training callbacks
        
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Add model checkpoint if save path is specified
        if self.config.model_save_path:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    self.config.model_save_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        return callbacks
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> None:
        """
        Train the model
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
        """
        if not self.is_compiled:
            self.compile_model()
        
        # Prepare validation data
        validation_data = None
        validation_split = 0
        
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            logger.info(f"Using provided validation data: {X_val.shape[0]} samples")
        else:
            validation_split = self.config.validation_split
            logger.info(f"Using validation split: {validation_split:.1%}")
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        logger.info("Starting training...")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Epochs: {self.config.epochs}")
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed!")
        self._log_training_summary()
    
    def _log_training_summary(self) -> None:
        """Log training summary statistics"""
        if self.history is None:
            return
        
        # Get final metrics
        final_loss = self.history.history['loss'][-1]
        final_acc = self.history.history['accuracy'][-1]
        
        logger.info("Training Summary:")
        logger.info(f"  Final training loss: {final_loss:.4f}")
        logger.info(f"  Final training accuracy: {final_acc:.4f}")
        
        if 'val_loss' in self.history.history:
            final_val_loss = self.history.history['val_loss'][-1]
            final_val_acc = self.history.history['val_accuracy'][-1]
            logger.info(f"  Final validation loss: {final_val_loss:.4f}")
            logger.info(f"  Final validation accuracy: {final_val_acc:.4f}")
        
        # Get best metrics
        best_epoch = np.argmin(self.history.history['val_loss']) if 'val_loss' in self.history.history else len(self.history.history['loss']) - 1
        logger.info(f"  Best epoch: {best_epoch + 1}")
        logger.info(f"  Total epochs trained: {len(self.history.history['loss'])}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model and return comprehensive metrics
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate basic metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'true_labels': y_test
        }
        
        logger.info(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        logger.info(f"Test Loss: {test_loss:.4f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data
        
        Args:
            X: Input data
            
        Returns:
            Tuple of (predicted_classes, prediction_probabilities)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        probabilities = self.model.predict(X, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities
    
    def save_model(self, filepath: str = None) -> None:
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model (optional)
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = filepath or self.config.model_save_path
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, filepath: str = None) -> None:
        """
        Load a pre-trained model
        
        Args:
            filepath: Path to load the model from (optional)
        """
        load_path = filepath or self.config.model_save_path
        
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        self.model = tf.keras.models.load_model(load_path)
        self.is_compiled = True
        logger.info(f"Model loaded from {load_path}")
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary as string
        
        Returns:
            String representation of model summary
        """
        if self.model is None:
            return "No model built yet"
        
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)


class ModelBuilder:
    """Factory class for building different model architectures"""
    
    @staticmethod
    def build_simple_lstm(config: Config) -> tf.keras.Model:
        """Build a simple single-layer LSTM model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(config.n_timesteps, config.n_features)),
            tf.keras.layers.LSTM(
                config.lstm_units,
                dropout=config.dropout_rate
            ),
            tf.keras.layers.Dense(config.n_classes, activation='softmax')
        ])
        return model

    @staticmethod
    def build_deep_lstm(config: Config, num_layers: int = 3) -> tf.keras.Model:
        """Build a deep LSTM model with multiple layers"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(config.n_timesteps, config.n_features)))
        
        # Add multiple LSTM layers
        for i in range(num_layers):
            return_sequences = i < num_layers - 1  # Only last layer doesn't return sequences
            model.add(tf.keras.layers.LSTM(
                config.lstm_units,
                return_sequences=return_sequences,
                dropout=config.dropout_rate,
                kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg)
            ))
        
        model.add(tf.keras.layers.Dense(
            config.lstm_units, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg)
        ))
        model.add(tf.keras.layers.Dropout(config.dropout_rate))
        model.add(tf.keras.layers.Dense(config.n_classes, activation='softmax'))
        
        return model
    
    @staticmethod
    def build_bidirectional_lstm(config: Config) -> tf.keras.Model:
        """Build a bidirectional LSTM model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(config.n_timesteps, config.n_features)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                config.lstm_units,
                return_sequences=True,
                dropout=config.dropout_rate
            )),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                config.lstm_units,
                dropout=config.dropout_rate
            )),
            tf.keras.layers.Dense(
                config.lstm_units, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg)
            ),
            tf.keras.layers.Dropout(config.dropout_rate),
            tf.keras.layers.Dense(config.n_classes, activation='softmax')
        ])
        return model