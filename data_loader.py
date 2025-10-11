"""
Data loading and preprocessing module for Human Activity Recognition
"""

import numpy as np
import urllib.request
import zipfile
import logging
from typing import Tuple
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loading and preprocessing utilities"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def download_dataset(self) -> None:
        """Download and extract the UCI HAR dataset"""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        zip_path = self.config.data_dir / "UCI HAR Dataset.zip"
        
        # Create data directory
        self.config.data_dir.mkdir(exist_ok=True)
        
        if not self.config.dataset_dir.exists():
            logger.info("Downloading UCI HAR Dataset...")
            try:
                urllib.request.urlretrieve(url, zip_path)
                logger.info("Download completed successfully")
                
                logger.info("Extracting dataset...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.config.data_dir)
                
                # Clean up zip file
                zip_path.unlink()
                logger.info("Dataset extracted successfully!")
                
            except Exception as e:
                logger.error(f"Failed to download dataset: {e}")
                if zip_path.exists():
                    zip_path.unlink()
                raise
        else:
            logger.info("Dataset already exists")
    
    def load_signals(self, subset: str) -> np.ndarray:
        """
        Load signal data for train or test subset
        
        Args:
            subset: Either 'train' or 'test'
            
        Returns:
            Numpy array of shape (n_samples, n_timesteps, n_features)
        """
        signals = []
        
        for signal_type in self.config.signal_types:
            file_path = (self.config.dataset_dir / subset / "Inertial Signals" / 
                        f"{signal_type}{subset}.txt")
            
            try:
                signal_data = np.loadtxt(file_path)
                signals.append(signal_data)
            except FileNotFoundError:
                logger.error(f"Signal file not found: {file_path}")
                raise
            except Exception as e:
                logger.error(f"Error loading signal file {file_path}: {e}")
                raise
        
        # Stack signals: (n_samples, n_timesteps, n_features)
        return np.transpose(np.array(signals), (1, 2, 0))
    
    def load_labels(self, subset: str) -> np.ndarray:
        """
        Load labels for train or test subset
        
        Args:
            subset: Either 'train' or 'test'
            
        Returns:
            Numpy array of labels (0-based indexing)
        """
        file_path = self.config.dataset_dir / subset / f"y_{subset}.txt"
        
        try:
            labels = np.loadtxt(file_path, dtype=int)
            return labels - 1  # Convert to 0-based indexing
        except FileNotFoundError:
            logger.error(f"Label file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading label file {file_path}: {e}")
            raise
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all training and testing data
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        logger.info("Loading training data...")
        X_train = self.load_signals("train")
        y_train = self.load_labels("train")
        
        logger.info("Loading test data...")
        X_test = self.load_signals("test")
        y_test = self.load_labels("test")
        
        # Log data statistics
        self._log_data_statistics(X_train, y_train, X_test, y_test)
        
        return X_train, y_train, X_test, y_test
    
    def _log_data_statistics(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Log basic statistics about the loaded data"""
        logger.info(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"Data shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Data range statistics
        logger.info(f"X_train - Min: {X_train.min():.3f}, Max: {X_train.max():.3f}, "
                   f"Mean: {X_train.mean():.3f}, Std: {X_train.std():.3f}")
        
        # Class distribution
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        
        logger.info("Training class distribution:")
        for label_idx, count in zip(unique_train, counts_train):
            label_name = self.config.labels[label_idx]
            percentage = count / len(y_train) * 100
            logger.info(f"  {label_name}: {count} samples ({percentage:.1f}%)")
        
        logger.info("Test class distribution:")
        for label_idx, count in zip(unique_test, counts_test):
            label_name = self.config.labels[label_idx]
            percentage = count / len(y_test) * 100
            logger.info(f"  {label_name}: {count} samples ({percentage:.1f}%)")
    
    def get_data_info(self) -> dict:
        """
        Get information about the dataset without loading it
        
        Returns:
            Dictionary containing dataset information
        """
        info = {
            'dataset_exists': self.config.dataset_dir.exists(),
            'expected_train_files': [],
            'expected_test_files': [],
            'signal_types': self.config.signal_types,
            'labels': self.config.labels
        }
        
        # Check for expected files
        for subset in ['train', 'test']:
            subset_dir = self.config.dataset_dir / subset
            if subset_dir.exists():
                # Check signal files
                signal_files = []
                for signal_type in self.config.signal_types:
                    file_path = subset_dir / "Inertial Signals" / f"{signal_type}{subset}.txt"
                    signal_files.append({
                        'file': file_path.name,
                        'exists': file_path.exists()
                    })
                
                # Check label file
                label_file = subset_dir / f"y_{subset}.txt"
                label_exists = label_file.exists()
                
                info[f'expected_{subset}_files'] = {
                    'signals': signal_files,
                    'labels': {'file': label_file.name, 'exists': label_exists}
                }
        
        return info


class DataPreprocessor:
    """Additional data preprocessing utilities"""
    
    @staticmethod
    def normalize_data(X: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, dict]:
        """
        Normalize the input data
        
        Args:
            X: Input data array
            method: Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
            Tuple of (normalized_data, normalization_params)
        """
        if method == 'standard':
            mean = np.mean(X, axis=0, keepdims=True)
            std = np.std(X, axis=0, keepdims=True)
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            normalized_X = (X - mean) / std
            params = {'mean': mean, 'std': std, 'method': 'standard'}
            
        elif method == 'minmax':
            min_val = np.min(X, axis=0, keepdims=True)
            max_val = np.max(X, axis=0, keepdims=True)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)  # Avoid division by zero
            normalized_X = (X - min_val) / range_val
            params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
            
        elif method == 'robust':
            median = np.median(X, axis=0, keepdims=True)
            q75 = np.percentile(X, 75, axis=0, keepdims=True)
            q25 = np.percentile(X, 25, axis=0, keepdims=True)
            iqr = q75 - q25
            iqr = np.where(iqr == 0, 1, iqr)  # Avoid division by zero
            normalized_X = (X - median) / iqr
            params = {'median': median, 'q25': q25, 'q75': q75, 'method': 'robust'}
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized_X, params
    
    @staticmethod
    def apply_normalization(X: np.ndarray, params: dict) -> np.ndarray:
        """Apply previously computed normalization parameters to new data"""
        method = params['method']
        
        if method == 'standard':
            return (X - params['mean']) / params['std']
        elif method == 'minmax':
            return (X - params['min']) / (params['max'] - params['min'])
        elif method == 'robust':
            return (X - params['median']) / (params['q75'] - params['q25'])
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def create_sliding_windows(X: np.ndarray, y: np.ndarray, 
                             window_size: int, step_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from time series data
        
        Args:
            X: Input data of shape (n_samples, n_timesteps, n_features)
            y: Labels of shape (n_samples,)
            window_size: Size of sliding window
            step_size: Step size for sliding window
            
        Returns:
            Tuple of (windowed_X, windowed_y)
        """
        windowed_X = []
        windowed_y = []
        
        for i in range(len(X)):
            sample = X[i]
            label = y[i]
            
            # Create windows for this sample
            for start in range(0, len(sample) - window_size + 1, step_size):
                end = start + window_size
                windowed_X.append(sample[start:end])
                windowed_y.append(label)
        
        return np.array(windowed_X), np.array(windowed_y)