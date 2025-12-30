"""
Factory for creating NLP classification models
"""
import numpy as np
import random
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 47
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

from models.base import BaseNLPModel
from models.ml.ml_models import (
    LogisticRegressionModel,
    MultinomialNBModel,
    BernoulliNBModel,
    GaussianNBModel
)


class NLPModelFactory:
    """Factory class to create NLP classification models based on configuration."""
    
    @staticmethod
    def create_models(config: Dict) -> Dict[str, BaseNLPModel]:
        """
        Create all models defined in config.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary of model name to model instance
        """
        models = {}
        
        # ML models
        if 'ml' in config.get('models', {}):
            ml_config = config['models']['ml']
            
            if 'logistic_regression' in ml_config:
                params = ml_config['logistic_regression'].copy()
                params['random_state'] = params.get('random_state', RANDOM_SEED)
                models['LogisticRegression'] = LogisticRegressionModel(**params)
            
            if 'naive_bayes_multinomial' in ml_config:
                params = ml_config['naive_bayes_multinomial'].copy()
                models['MultinomialNB'] = MultinomialNBModel(**params)
            
            if 'naive_bayes_bernoulli' in ml_config:
                params = ml_config['naive_bayes_bernoulli'].copy()
                models['BernoulliNB'] = BernoulliNBModel(**params)
            
            if 'naive_bayes_gaussian' in ml_config:
                params = ml_config['naive_bayes_gaussian'].copy()
                models['GaussianNB'] = GaussianNBModel(**params)
        
        return models
    
    @staticmethod
    def get_model_params(model: BaseNLPModel) -> Dict:
        """
        Get model parameters.
        
        Args:
            model: Model instance
            
        Returns:
            Dictionary of parameters
        """
        return model.get_params()


