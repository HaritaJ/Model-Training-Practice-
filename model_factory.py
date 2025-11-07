"""
Factory pattern for creating model instances.
"""
from typing import Optional
import logging

from base_model import BaseModel
from sklearn_models import SklearnLinearRegression, SklearnRandomForest
from pytorch_models import PyTorchLinearRegression, PyTorchMLP
from config import SklearnModelConfig, PyTorchModelConfig, MLPConfig

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory class for creating model instances.
    
    This follows the Factory Pattern to provide a centralized way
    to create different types of models.
    """
    
    _model_registry = {
        "sklearn_lr": SklearnLinearRegression,
        "sklearn_rf": SklearnRandomForest,
        "pytorch_lr": PyTorchLinearRegression,
        "pytorch_mlp": PyTorchMLP,
    }
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        config: Optional[object] = None
    ) -> BaseModel:
        """
        Create a model instance based on the model type.
        
        Args:
            model_type: Type of model to create. Options:
                - "sklearn_lr": Scikit-learn Linear Regression
                - "sklearn_rf": Scikit-learn Random Forest
                - "pytorch_lr": PyTorch Linear Regression
                - "pytorch_mlp": PyTorch Multi-Layer Perceptron
            config: Optional configuration object for the model
            
        Returns:
            Instance of the requested model
            
        Raises:
            ValueError: If model_type is not recognized
        """
        if model_type not in cls._model_registry:
            available = ", ".join(cls._model_registry.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available types: {available}"
            )
        
        model_class = cls._model_registry[model_type]
        
        if config is None:
            # Use default config based on model type
            if model_type.startswith("sklearn"):
                config = SklearnModelConfig()
            elif model_type == "pytorch_mlp":
                config = MLPConfig()
            else:
                config = PyTorchModelConfig()
        
        logger.info(f"Creating {model_type} model with config: {config}")
        return model_class(config)
    
    @classmethod
    def get_available_models(cls) -> list[str]:
        return list(cls._model_registry.keys())

