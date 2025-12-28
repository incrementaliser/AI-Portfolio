"""
Time series model definitions and factory

This module provides backwards compatibility by re-exporting all models
from the modular structure. The actual implementations are now organized
in separate files:
- base.py: BaseTimeSeriesModel
- statistical.py: ARIMA, Prophet, ETS, Theta
- ml_models.py: ML models with lag features
- deep_learning.py: Deep learning models
- factory.py: TimeSeriesModelFactory
"""

from models.base import BaseTimeSeriesModel
from models.statistical.statistical import (
    ARIMAWrapper,
    ProphetWrapper,
    ETSWrapper,
    ThetaWrapper
)
from models.ml.ml_models import MLTimeSeriesWrapper
from models.neural.deep_learning import DartsModelWrapper
from models.factory import TimeSeriesModelFactory

__all__ = [
    'BaseTimeSeriesModel',
    'ARIMAWrapper',
    'ProphetWrapper',
    'ETSWrapper',
    'ThetaWrapper',
    'MLTimeSeriesWrapper',
    'DartsModelWrapper',
    'TimeSeriesModelFactory',
]
