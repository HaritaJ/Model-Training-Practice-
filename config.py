"""
Configuration classes for model training and evaluation.
"""
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class SklearnModelConfig:
    """Configuration for sklearn models."""
    random_state: int = 42
    n_estimators: int = 100  # For Random Forest


@dataclass
class PyTorchModelConfig:
    """Configuration for PyTorch models."""
    learning_rate: float = 0.01
    num_epochs: int = 20000
    batch_size: Optional[int] = None
    device: str = "cpu"
    print_every: int = 50  # Print loss every N epochs
    
    def __post_init__(self):
        """Validate and set device."""
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"


@dataclass
class MLPConfig(PyTorchModelConfig):
    """Configuration for Multi-Layer Perceptron."""
    hidden_layers: list[int] = None
    activation: str = "relu"
    dropout: float = 0.0
    
    def __post_init__(self):
        """Initialize default hidden layers if not provided."""
        super().__post_init__()
        if self.hidden_layers is None:
            self.hidden_layers = [64, 32]

