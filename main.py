"""
Main script for training and evaluating machine learning models.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

from utils import setup_logging
from model_factory import ModelFactory
from config import SklearnModelConfig, PyTorchModelConfig, MLPConfig

# Set up logging
setup_logging(log_level="INFO", log_file="logs/training.log")
logger = logging.getLogger(__name__)


def calculate_baseline(y_train: pd.Series, y_test: pd.Series) -> float:
    """
    Calculate baseline MSE using mean of training data.
    
    Args:
        y_train: Training targets
        y_test: Test targets
        
    Returns:
        Baseline MSE
    """
    baseline_pred = np.mean(y_train)
    baseline_mse = mean_squared_error(y_test, [baseline_pred] * len(y_test))
    logger.info(f"Baseline MSE: {baseline_mse:.4f}")
    return baseline_mse


def main():
    """Main function to train and evaluate models."""
    logger.info("=" * 60)
    logger.info("Starting Model Training Pipeline")
    logger.info("=" * 60)
    
    # Load data
    logger.info("Loading diabetes dataset...")
    data = load_diabetes(as_frame=True)
    df = data.frame
    
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    
    # Calculate baseline
    baseline_mse = calculate_baseline(y_train, y_test)
    
    # Define models to train
    models_to_train = [
        {
            "type": "sklearn_lr",
            "name": "Scikit-learn Linear Regression",
            "config": SklearnModelConfig(random_state=42)
        },
        {
            "type": "sklearn_rf",
            "name": "Scikit-learn Random Forest",
            "config": SklearnModelConfig(n_estimators=100, random_state=42)
        },
        {
            "type": "pytorch_lr",
            "name": "PyTorch Linear Regression",
            "config": PyTorchModelConfig(learning_rate=0.01, num_epochs=20000, print_every=1000)
        },
        {
            "type": "pytorch_mlp",
            "name": "PyTorch MLP",
            "config": MLPConfig(
                learning_rate=0.001,
                num_epochs=20000,
                hidden_layers=[64, 32],
                print_every=1000
            )
        }
    ]
    
    results = {}
    
    # Train and evaluate each model
    for model_info in models_to_train:
        logger.info("")
        logger.info("-" * 60)
        logger.info(f"Training {model_info['name']}")
        logger.info("-" * 60)
        
        try:
            # Create model
            model = ModelFactory.create_model(
                model_type=model_info["type"],
                config=model_info["config"]
            )
            
            # Train model
            model.train(X_train, y_train, X_test, y_test)
            
            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            results[model_info["name"]] = metrics
            
        except Exception as e:
            logger.error(f"Error training {model_info['name']}: {str(e)}", exc_info=True)
            results[model_info["name"]] = {"error": str(e)}
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(f"Baseline MSE: {baseline_mse:.4f}")
    logger.info("")
    
    for model_name, metrics in results.items():
        if "error" not in metrics:
            logger.info(f"{model_name}:")
            logger.info(f"  MSE:  {metrics['mse']:.4f}")
            logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            logger.info(f"  MAE:  {metrics['mae']:.4f}")
            logger.info(f"  R²:   {metrics['r2']:.4f}")
            logger.info("")
        else:
            logger.error(f"{model_name}: Error - {metrics['error']}")
    
    logger.info("=" * 60)
    logger.info("Training Pipeline Completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
