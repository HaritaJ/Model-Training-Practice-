"""
Example usage of the production-ready model classes.
"""
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from utils import setup_logging
from model_factory import ModelFactory
from config import SklearnModelConfig, PyTorchModelConfig, MLPConfig

# Set up logging
setup_logging(log_level="INFO")

# Load data
data = load_diabetes(as_frame=True)
df = data.frame
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Example 1: Using the factory pattern
print("\n=== Example 1: Using Factory Pattern ===")
model = ModelFactory.create_model("sklearn_lr")
model.train(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
print(f"Metrics: {metrics}")

# Example 2: Creating models directly with custom config
print("\n=== Example 2: Direct Instantiation with Custom Config ===")
from sklearn_models import SklearnRandomForest

config = SklearnModelConfig(n_estimators=200, random_state=42)
rf_model = SklearnRandomForest(config=config)
rf_model.train(X_train, y_train, X_test, y_test)
metrics = rf_model.evaluate(X_test, y_test)
print(f"Random Forest Metrics: {metrics}")

# Example 3: PyTorch model with custom config
print("\n=== Example 3: PyTorch Model with Custom Config ===")
pytorch_config = PyTorchModelConfig(
    learning_rate=0.01,
    num_epochs=1000,
    print_every=200
)
pytorch_model = ModelFactory.create_model("pytorch_lr", config=pytorch_config)
pytorch_model.train(X_train, y_train, X_test, y_test)
metrics = pytorch_model.evaluate(X_test, y_test)
print(f"PyTorch Linear Regression Metrics: {metrics}")

# Example 4: Save and load model
print("\n=== Example 4: Model Persistence ===")
model.train(X_train, y_train)
model.save("models/example_model.joblib")

# Load the model
from sklearn_models import SklearnLinearRegression
loaded_model = SklearnLinearRegression()
loaded_model.load("models/example_model.joblib")
predictions = loaded_model.predict(X_test)
print(f"Loaded model predictions shape: {predictions.shape}")

