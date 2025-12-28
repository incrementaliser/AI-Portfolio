"""
Factory for creating time series models
"""

from typing import Dict
import warnings
warnings.filterwarnings('ignore')

from models.base import BaseTimeSeriesModel
from models.statistical.statistical import ARIMAWrapper, ProphetWrapper, ETSWrapper, ThetaWrapper
from models.ml.ml_models import (
    MLTimeSeriesWrapper,
    RandomForestRegressor,
    GradientBoostingRegressor,
    Ridge
)
from models.neural.deep_learning import DartsModelWrapper

# Import all dependencies directly
import xgboost as xgb
import lightgbm as lgb
from darts.models import RNNModel, NBEATSModel, TransformerModel, TCNModel


class TimeSeriesModelFactory:
    """Factory class to create time series models based on configuration."""
    
    @staticmethod
    def create_models(config: Dict) -> Dict[str, BaseTimeSeriesModel]:
        """
        Create all models defined in config.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary of model name to model instance
        """
        models = {}
        
        # Statistical models
        if 'statistical' in config.get('models', {}):
            statistical_config = config['models']['statistical']
            
            if 'arima' in statistical_config:
                params = statistical_config['arima']
                order = params.get('order', (1, 1, 1))
                models['ARIMA'] = ARIMAWrapper(order=tuple(order))
            
            if 'prophet' in statistical_config:
                params = statistical_config['prophet']
                models['Prophet'] = ProphetWrapper(**params)
            
            if 'ets' in statistical_config:
                params = statistical_config['ets']
                models['ETS'] = ETSWrapper(**params)
            
            if 'theta' in statistical_config:
                params = statistical_config['theta']
                models['Theta'] = ThetaWrapper(**params)
        
        # ML models
        if 'ml' in config.get('models', {}):
            ml_config = config['models']['ml']
            lookback = config.get('data', {}).get('lookback_window', 10)
            
            if 'random_forest' in ml_config:
                params = ml_config['random_forest']
                model = RandomForestRegressor(**params)
                models['RandomForest'] = MLTimeSeriesWrapper(model, lookback=lookback)
            
            if 'gradient_boosting' in ml_config:
                params = ml_config['gradient_boosting']
                model = GradientBoostingRegressor(**params)
                models['GradientBoosting'] = MLTimeSeriesWrapper(model, lookback=lookback)
            
            if 'xgboost' in ml_config:
                params = ml_config['xgboost']
                model = xgb.XGBRegressor(**params)
                models['XGBoost'] = MLTimeSeriesWrapper(model, lookback=lookback)
            
            if 'lightgbm' in ml_config:
                params = ml_config['lightgbm']
                model = lgb.LGBMRegressor(**params)
                models['LightGBM'] = MLTimeSeriesWrapper(model, lookback=lookback)
            
            if 'ridge' in ml_config:
                params = ml_config['ridge']
                model = Ridge(**params)
                models['Ridge'] = MLTimeSeriesWrapper(model, lookback=lookback)
        
        # Deep learning models
        if 'deep_learning' in config.get('models', {}):
            dl_config = config['models']['deep_learning']
            input_len = config.get('data', {}).get('lookback_window', 24)
            output_len = max(config.get('data', {}).get('forecast_horizons', [1]))
            
            if 'lstm' in dl_config:
                params = dl_config['lstm']
                models['LSTM'] = DartsModelWrapper(
                    RNNModel,
                    input_chunk_length=input_len,
                    output_chunk_length=output_len,
                    model='LSTM',
                    **params
                )
            
            if 'gru' in dl_config:
                params = dl_config['gru']
                models['GRU'] = DartsModelWrapper(
                    RNNModel,
                    input_chunk_length=input_len,
                    output_chunk_length=output_len,
                    model='GRU',
                    **params
                )
            
            if 'nbeats' in dl_config:
                params = dl_config['nbeats']
                models['NBEATS'] = DartsModelWrapper(
                    NBEATSModel,
                    input_chunk_length=input_len,
                    output_chunk_length=output_len,
                    **params
                )
            
            if 'transformer' in dl_config:
                params = dl_config['transformer']
                models['Transformer'] = DartsModelWrapper(
                    TransformerModel,
                    input_chunk_length=input_len,
                    output_chunk_length=output_len,
                    **params
                )
            
            if 'tcn' in dl_config:
                params = dl_config['tcn']
                models['TCN'] = DartsModelWrapper(
                    TCNModel,
                    input_chunk_length=input_len,
                    output_chunk_length=output_len,
                    **params
                )
        
        return models
    
    @staticmethod
    def get_model_params(model: BaseTimeSeriesModel) -> Dict:
        """
        Get model parameters.
        
        Args:
            model: Model instance
            
        Returns:
            Dictionary of parameters
        """
        return model.get_params()
