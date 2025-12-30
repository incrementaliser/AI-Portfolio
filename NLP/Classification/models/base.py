"""
Base model interface for NLP classification models
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseNLPModel(ABC):
    """Base class for all NLP classification models."""
    
    def __init__(self, model_type: str = "ml"):
        """
        Initialize base NLP model.
        
        Args:
            model_type: Type of model (e.g., "ml", "deep_learning")
        """
        self.model_type = model_type
        self.model = None
    
    @abstractmethod
    def fit(self, X, y) -> None:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
        """
        pass
    
    @abstractmethod
    def predict(self, X) -> Any:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X) -> Any:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of parameters
        """
        if self.model is not None and hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}

