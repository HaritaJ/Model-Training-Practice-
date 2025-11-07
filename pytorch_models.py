"""
PyTorch model implementations.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from base_model import BaseModel
from config import PyTorchModelConfig, MLPConfig
from lr import LinearRegressionModel

logger = logging.getLogger(__name__)


class PyTorchLinearRegression(BaseModel):
    """
    PyTorch Linear Regression model wrapper.
    """
    
    def __init__(self, config: Optional[PyTorchModelConfig] = None):
        """
        Initialize the PyTorch Linear Regression model.
        
        Args:
            config: Configuration object for the model
        """
        super().__init__(name="PyTorchLinearRegression")
        self.config = config or PyTorchModelConfig()
        self.device = torch.device(self.config.device)
        self.model: Optional[nn.Module] = None
        self.criterion = nn.MSELoss()
        self.optimizer: Optional[optim.Optimizer] = None
    
    def _prepare_data(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert pandas DataFrame/Series to PyTorch tensors.
        
        Args:
            X: Features
            y: Optional targets
            
        Returns:
            Tuple of (X_tensor, y_tensor)
        """
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        
        if y is not None:
            y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(self.device)
        else:
            y_tensor = None
        
        return X_tensor, y_tensor
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """
        Train the PyTorch Linear Regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
        """
        self._validate_input(X_train, y_train)
        
        # Initialize model if not already done
        if self.model is None:
            input_dim = X_train.shape[1]
            self.model = LinearRegressionModel(input_dim).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        X_train_tensor, y_train_tensor = self._prepare_data(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val)
        
        logger.info(f"Training {self.name} on {self.device}...")
        logger.info(f"Learning rate: {self.config.learning_rate}, Epochs: {self.config.num_epochs}")
        
        self.training_history = []
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            
            # Forward pass
            y_pred = self.model(X_train_tensor)
            loss = self.criterion(y_pred, y_train_tensor)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Log progress
            if (epoch + 1) % self.config.print_every == 0:
                train_loss = loss.item()
                log_msg = f"Epoch [{epoch+1}/{self.config.num_epochs}], Loss: {train_loss:.4f}"
                
                if X_val is not None and y_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_pred = self.model(X_val_tensor)
                        val_loss = self.criterion(val_pred, y_val_tensor).item()
                    log_msg += f", Val Loss: {val_loss:.4f}"
                    self.training_history.append({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss
                    })
                else:
                    self.training_history.append({
                        "epoch": epoch + 1,
                        "train_loss": train_loss
                    })
                
                logger.info(log_msg)
        
        self.is_trained = True
        logger.info(f"Training completed. Final training loss: {loss.item():.4f}")
    
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
        
        X_tensor, _ = self._prepare_data(X)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy().flatten()
    
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
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "config": self.config,
            "training_history": self.training_history
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Reconstruct model architecture
        # Note: This requires knowing input_dim, which should be saved in checkpoint
        if self.model is None:
            # This is a limitation - we need input_dim to reconstruct
            raise ValueError("Model architecture must be initialized before loading weights")
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint["optimizer_state_dict"] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.training_history = checkpoint.get("training_history", [])
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class PyTorchMLP(BaseModel):
    """
    PyTorch Multi-Layer Perceptron model wrapper.
    """
    
    def __init__(self, config: Optional[MLPConfig] = None):
        """
        Initialize the PyTorch MLP model.
        
        Args:
            config: Configuration object for the model
        """
        super().__init__(name="PyTorchMLP")
        self.config = config or MLPConfig()
        self.device = torch.device(self.config.device)
        self.model: Optional[nn.Module] = None
        self.criterion = nn.MSELoss()
        self.optimizer: Optional[optim.Optimizer] = None
        self.input_dim: Optional[int] = None
    
    def _build_model(self, input_dim: int) -> nn.Module:
        """
        Build the MLP architecture.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            PyTorch model
        """
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in self.config.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.config.activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif self.config.activation.lower() == "tanh":
                layers.append(nn.Tanh())
            elif self.config.activation.lower() == "sigmoid":
                layers.append(nn.Sigmoid())
            
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _prepare_data(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert pandas DataFrame/Series to PyTorch tensors.
        
        Args:
            X: Features
            y: Optional targets
            
        Returns:
            Tuple of (X_tensor, y_tensor)
        """
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        
        if y is not None:
            y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(self.device)
        else:
            y_tensor = None
        
        return X_tensor, y_tensor
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """
        Train the PyTorch MLP model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
        """
        self._validate_input(X_train, y_train)
        
        # Initialize model if not already done
        if self.model is None:
            self.input_dim = X_train.shape[1]
            self.model = self._build_model(self.input_dim).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        X_train_tensor, y_train_tensor = self._prepare_data(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val)
        
        logger.info(f"Training {self.name} on {self.device}...")
        logger.info(f"Architecture: {self.input_dim} -> {self.config.hidden_layers} -> 1")
        logger.info(f"Learning rate: {self.config.learning_rate}, Epochs: {self.config.num_epochs}")
        
        self.training_history = []
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            
            # Forward pass
            y_pred = self.model(X_train_tensor)
            loss = self.criterion(y_pred, y_train_tensor)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Log progress
            if (epoch + 1) % self.config.print_every == 0:
                train_loss = loss.item()
                log_msg = f"Epoch [{epoch+1}/{self.config.num_epochs}], Loss: {train_loss:.4f}"
                
                if X_val is not None and y_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_pred = self.model(X_val_tensor)
                        val_loss = self.criterion(val_pred, y_val_tensor).item()
                    log_msg += f", Val Loss: {val_loss:.4f}"
                    self.training_history.append({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss
                    })
                else:
                    self.training_history.append({
                        "epoch": epoch + 1,
                        "train_loss": train_loss
                    })
                
                logger.info(log_msg)
        
        self.is_trained = True
        logger.info(f"Training completed. Final training loss: {loss.item():.4f}")
    
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
        
        X_tensor, _ = self._prepare_data(X)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy().flatten()
    
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
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "config": self.config,
            "input_dim": self.input_dim,
            "training_history": self.training_history
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.input_dim = checkpoint["input_dim"]
        self.model = self._build_model(self.input_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if checkpoint["optimizer_state_dict"]:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.training_history = checkpoint.get("training_history", [])
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")

