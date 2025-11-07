"""
Abstract base class for all machine learning models.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class that defines the interface for all models.
    
    All model implementations should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(self, name: str):
        """
        Initialize the base model.
        
        Args:
            name: Name identifier for the model
        """
        self.name = name
        self.model: Any = None
        self.is_trained: bool = False
        self.training_history: list = []
        
    @abstractmethod
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions as numpy array
        """
        pass
    
    @abstractmethod
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary of metric names and values
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where the model should be saved
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        pass
    
    def _validate_input(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Validate input data.
        
        Args:
            X: Features to validate
            y: Optional targets to validate
            
        Raises:
            ValueError: If input data is invalid
        """
        if X is None or X.empty:
            raise ValueError("Features X cannot be None or empty")
        
        if y is not None:
            if len(X) != len(y):
                raise ValueError(f"X and y must have the same length. Got {len(X)} and {len(y)}")
            
            if y.isna().any():
                raise ValueError("Target y contains NaN values")
        
        if X.isna().any().any():
            logger.warning("Features X contain NaN values")
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}(name='{self.name}', status='{status}')"

