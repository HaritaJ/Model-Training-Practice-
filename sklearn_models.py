"""
Scikit-learn model implementations.
"""
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from base_model import BaseModel
from config import SklearnModelConfig

logger = logging.getLogger(__name__)


class SklearnLinearRegression(BaseModel):
    """
    Scikit-learn Linear Regression model wrapper.
    """
    
    def __init__(self, config: Optional[SklearnModelConfig] = None):
        """
        Initialize the Linear Regression model.
        
        Args:
            config: Configuration object for the model
        """
        super().__init__(name="SklearnLinearRegression")
        self.config = config or SklearnModelConfig()
        self.model = LinearRegression()
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """
        Train the Linear Regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features (not used for sklearn)
            y_val: Optional validation targets (not used for sklearn)
        """
        self._validate_input(X_train, y_train)
        
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_pred)
        logger.info(f"Training MSE: {train_mse:.4f}")
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_pred)
            logger.info(f"Validation MSE: {val_mse:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions as numpy array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self._validate_input(X)
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary of metric names and values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        self._validate_input(X, y)
        
        y_pred = self.predict(X)
        metrics = {
            "mse": mean_squared_error(y, y_pred),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred))
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where the model should be saved
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class SklearnRandomForest(BaseModel):
    """
    Scikit-learn Random Forest Regressor model wrapper.
    """
    
    def __init__(self, config: Optional[SklearnModelConfig] = None):
        """
        Initialize the Random Forest model.
        
        Args:
            config: Configuration object for the model
        """
        super().__init__(name="SklearnRandomForest")
        self.config = config or SklearnModelConfig()
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state
        )
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:

        self._validate_input(X_train, y_train)
        
        logger.info(f"Training {self.name} with {self.config.n_estimators} estimators...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_pred)
        logger.info(f"Training MSE: {train_mse:.4f}")
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_pred)
            logger.info(f"Validation MSE: {val_mse:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions as numpy array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self._validate_input(X)
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary of metric names and values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        self._validate_input(X, y)
        
        y_pred = self.predict(X)
        metrics = {
            "mse": mean_squared_error(y, y_pred),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred))
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save(self, filepath: str) -> None:
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")

