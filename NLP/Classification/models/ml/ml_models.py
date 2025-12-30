"""
Machine learning models for NLP classification
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from models.base import BaseNLPModel


class LogisticRegressionModel(BaseNLPModel):
    """Logistic Regression model wrapper."""
    
    def __init__(self, **kwargs):
        """
        Initialize Logistic Regression model.
        
        Args:
            **kwargs: Parameters for LogisticRegression
        """
        super().__init__(model_type="ml")
        self.model = LogisticRegression(**kwargs)
    
    def fit(self, X, y) -> None:
        """
        Train the model.
        
        Args:
            X: Training features (sparse matrix or array)
            y: Training labels
        """
        self.model.fit(X, y)
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities
        """
        return self.model.predict_proba(X)


class MultinomialNBModel(BaseNLPModel):
    """Multinomial Naive Bayes model wrapper."""
    
    def __init__(self, **kwargs):
        """
        Initialize Multinomial Naive Bayes model.
        
        Args:
            **kwargs: Parameters for MultinomialNB
        """
        super().__init__(model_type="ml")
        self.model = MultinomialNB(**kwargs)
    
    def fit(self, X, y) -> None:
        """
        Train the model.
        
        Args:
            X: Training features (sparse matrix or array)
            y: Training labels
        """
        self.model.fit(X, y)
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities
        """
        return self.model.predict_proba(X)


class BernoulliNBModel(BaseNLPModel):
    """Bernoulli Naive Bayes model wrapper."""
    
    def __init__(self, **kwargs):
        """
        Initialize Bernoulli Naive Bayes model.
        
        Args:
            **kwargs: Parameters for BernoulliNB
        """
        super().__init__(model_type="ml")
        self.model = BernoulliNB(**kwargs)
    
    def fit(self, X, y) -> None:
        """
        Train the model.
        
        Args:
            X: Training features (sparse matrix or array)
            y: Training labels
        """
        self.model.fit(X, y)
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities
        """
        return self.model.predict_proba(X)


class GaussianNBModel(BaseNLPModel):
    """Gaussian Naive Bayes model wrapper."""
    
    def __init__(self, **kwargs):
        """
        Initialize Gaussian Naive Bayes model.
        
        Args:
            **kwargs: Parameters for GaussianNB
        """
        super().__init__(model_type="ml")
        self.model = GaussianNB(**kwargs)
    
    def fit(self, X, y) -> None:
        """
        Train the model.
        
        Args:
            X: Training features (dense array required)
            y: Training labels
        """
        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
        self.model.fit(X, y)
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        if hasattr(X, 'toarray'):
            X = X.toarray()
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities
        """
        if hasattr(X, 'toarray'):
            X = X.toarray()
        return self.model.predict_proba(X)


