"""
Configuration module for Human Activity Recognition project
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

_BACKEND_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    """Configuration class for the HAR model"""

    # Data parameters
    n_timesteps: int = 128
    n_features: int = 9
    n_classes: int = 6

    # Model parameters
    lstm_units: int = 32
    dropout_rate: float = 0.3
    l2_reg: float = 0.001

    # Training parameters
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2

    # Data paths (relative to CWD = project root when training)
    data_dir: Path = Path("data")
    dataset_dir: Path = Path("data/UCI HAR Dataset")
    # Model/results live inside backend/ regardless of CWD.
    model_save_path: str = str(_BACKEND_DIR / "best_model.keras")
    results_save_path: str = "results"
    
    # Activity labels
    labels: List[str] = None
    signal_types: List[str] = None
    
    def __post_init__(self):
        """Initialize labels and signal types after dataclass creation"""
        if self.labels is None:
            self.labels = [
                "WALKING", 
                "WALKING_UPSTAIRS", 
                "WALKING_DOWNSTAIRS",
                "SITTING", 
                "STANDING", 
                "LAYING"
            ]
        
        if self.signal_types is None:
            self.signal_types = [
                "body_acc_x_", "body_acc_y_", "body_acc_z_",
                "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
                "total_acc_x_", "total_acc_y_", "total_acc_z_"
            ]
    
    def create_directories(self) -> None:
        """Create necessary directories for the project"""
        self.data_dir.mkdir(exist_ok=True)
        Path(self.results_save_path).mkdir(exist_ok=True)
    
    def get_model_params(self) -> dict:
        """Get model parameters as dictionary"""
        return {
            'n_timesteps': self.n_timesteps,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split
        }
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")


# Predefined configurations for different scenarios
class ConfigPresets:
    """Predefined configuration presets for different use cases"""
    
    @staticmethod
    def get_quick_test_config() -> Config:
        """Configuration for quick testing with reduced parameters"""
        config = Config()
        config.epochs = 10
        config.batch_size = 32
        config.lstm_units = 16
        return config
    
    @staticmethod
    def get_high_performance_config() -> Config:
        """Configuration for high performance training"""
        config = Config()
        config.epochs = 200
        config.batch_size = 128
        config.lstm_units = 64
        config.dropout_rate = 0.4
        config.learning_rate = 0.0005
        return config
    
    @staticmethod
    def get_lightweight_config() -> Config:
        """Configuration for lightweight model (for deployment)"""
        config = Config()
        config.lstm_units = 16
        config.dropout_rate = 0.2
        config.batch_size = 32
        return config