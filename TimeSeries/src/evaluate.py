"""
Time series model evaluation module
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit
import wandb
import os
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesEvaluator:
    """Evaluator for time series forecasting models."""
    
    def __init__(self, config: Dict):
        """
        Initialize time series evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAE score
        """
        return np.mean(np.abs(y_true - y_pred))
    
    def calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MSE score
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            RMSE score
        """
        return np.sqrt(self.calculate_mse(y_true, y_pred))
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            epsilon: Small value to avoid division by zero
            
        Returns:
            MAPE score (in percentage)
        """
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    def calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            epsilon: Small value to avoid division by zero
            
        Returns:
            SMAPE score (in percentage)
        """
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
        return np.mean(numerator / denominator) * 100
    
    def calculate_mase(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_train: np.ndarray,
        seasonality: int = 1
    ) -> float:
        """
        Calculate Mean Absolute Scaled Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_train: Training data for scaling
            seasonality: Seasonal period (1 for non-seasonal)
            
        Returns:
            MASE score
        """
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Calculate naive forecast MAE on training data
        naive_mae = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
        
        if naive_mae == 0:
            return np.inf
        
        return mae / naive_mae
    
    def calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R² score.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            R² score
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def calculate_forecast_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_train: np.ndarray = None,
        prefix: str = 'test'
    ) -> Dict[str, float]:
        """
        Calculate all forecasting metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_train: Training data (for MASE calculation)
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            f'{prefix}_mae': self.calculate_mae(y_true, y_pred),
            f'{prefix}_mse': self.calculate_mse(y_true, y_pred),
            f'{prefix}_rmse': self.calculate_rmse(y_true, y_pred),
            f'{prefix}_mape': self.calculate_mape(y_true, y_pred),
            f'{prefix}_smape': self.calculate_smape(y_true, y_pred),
            f'{prefix}_r2': self.calculate_r2(y_true, y_pred)
        }
        
        if y_train is not None:
            metrics[f'{prefix}_mase'] = self.calculate_mase(y_true, y_pred, y_train)
        
        return metrics
    
    def time_series_cross_validate(
        self,
        model: Any,
        data: pd.Series,
        n_splits: int = 5,
        forecast_horizon: int = 1
    ) -> Dict[str, float]:
        """
        Perform time series cross-validation with expanding window.
        
        Args:
            model: Time series model instance
            data: Time series data
            n_splits: Number of splits
            forecast_horizon: Number of steps to forecast
            
        Returns:
            Dictionary with mean and std of metrics
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        mae_scores = []
        rmse_scores = []
        mape_scores = []
        
        for train_idx, test_idx in tscv.split(data):
            # Get train and test data
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Limit test data to forecast horizon
            if len(test_data) > forecast_horizon:
                test_data = test_data[:forecast_horizon]
            
            try:
                # Fit model - Prophet and Darts models need Series, others can use values
                if isinstance(train_data, pd.Series):
                    if hasattr(model, '__class__'):
                        class_name = model.__class__.__name__
                        if 'ProphetWrapper' in class_name or 'DartsModelWrapper' in class_name:
                            model.fit(train_data)
                        else:
                            model.fit(train_data.values)
                    else:
                        model.fit(train_data.values)
                else:
                    model.fit(train_data)
                
                # Predict
                predictions = model.predict(steps=len(test_data))
                
                # Calculate metrics
                mae_scores.append(self.calculate_mae(test_data.values, predictions))
                rmse_scores.append(self.calculate_rmse(test_data.values, predictions))
                mape_scores.append(self.calculate_mape(test_data.values, predictions))
            except Exception as e:
                print(f"Warning: CV fold failed with error: {e}")
                continue
        
        if len(mae_scores) == 0:
            return {}
        
        return {
            'cv_mae_mean': np.mean(mae_scores),
            'cv_mae_std': np.std(mae_scores),
            'cv_rmse_mean': np.mean(rmse_scores),
            'cv_rmse_std': np.std(rmse_scores),
            'cv_mape_mean': np.mean(mape_scores),
            'cv_mape_std': np.std(mape_scores)
        }
    
    def backtest(
        self,
        model: Any,
        train_data: pd.Series,
        test_data: pd.Series,
        forecast_horizon: int,
        step_size: int = 1
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backtesting with rolling window.
        
        Args:
            model: Time series model instance
            train_data: Initial training data
            test_data: Test data for backtesting
            forecast_horizon: Number of steps to forecast
            step_size: Number of steps to move forward
            
        Returns:
            Tuple of (predictions_list, actuals_list)
        """
        predictions_list = []
        actuals_list = []
        
        current_train = train_data.copy()
        
        for i in range(0, len(test_data) - forecast_horizon + 1, step_size):
            try:
                # Fit model on current training data
                # Prophet and Darts models need Series with datetime index, others can use values
                if hasattr(model, '__class__'):
                    class_name = model.__class__.__name__
                    if 'ProphetWrapper' in class_name or 'DartsModelWrapper' in class_name:
                        model.fit(current_train)
                    else:
                        model.fit(current_train.values)
                else:
                    model.fit(current_train.values)
                
                # Make prediction
                pred = model.predict(steps=forecast_horizon)
                
                # Get actual values
                actual = test_data.iloc[i:i+forecast_horizon].values
                
                predictions_list.append(pred)
                actuals_list.append(actual)
                
                # Update training data (expanding window)
                current_train = pd.concat([
                    current_train, 
                    test_data.iloc[i:i+step_size]
                ])
            except Exception as e:
                print(f"Warning: Backtest iteration failed: {e}")
                continue
        
        return predictions_list, actuals_list
    
    def evaluate_model(
        self,
        model: Any,
        train_data: pd.Series,
        val_data: pd.Series,
        test_data: pd.Series,
        forecast_horizons: List[int] = None
    ) -> Tuple[Dict, Any, Dict]:
        """
        Comprehensive model evaluation across multiple forecast horizons.
        
        Args:
            model: Time series model instance
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            forecast_horizons: List of forecast horizons to evaluate
            
        Returns:
            Tuple of (metrics_dict, fitted_model, predictions_dict)
        """
        if forecast_horizons is None:
            forecast_horizons = self.config.get('data', {}).get('forecast_horizons', [1])
        
        results = {}
        predictions_dict = {}
        
        # Fit model on training data
        print(f"  Training model on {len(train_data)} samples...")
        try:
            # Prophet and Darts models need Series with datetime index, others can use values
            if hasattr(model, '__class__'):
                class_name = model.__class__.__name__
                if 'ProphetWrapper' in class_name or 'DartsModelWrapper' in class_name:
                    model.fit(train_data)
                else:
                    model.fit(train_data.values)
            else:
                model.fit(train_data.values)
        except Exception as e:
            print(f"  Error fitting model: {e}")
            import traceback
            traceback.print_exc()
            return {}, model, {}
        
        # Evaluate on validation set
        print("  Evaluating on validation set...")
        for horizon in forecast_horizons:
            try:
                val_pred = model.predict(steps=min(horizon, len(val_data)))
                val_actual = val_data.values[:len(val_pred)]
                
                val_metrics = self.calculate_forecast_metrics(
                    val_actual, 
                    val_pred,
                    train_data.values,
                    prefix=f'val_h{horizon}'
                )
                results.update(val_metrics)
            except Exception as e:
                print(f"  Warning: Validation failed for horizon {horizon}: {e}")
        
        # Evaluate on test set
        print("  Evaluating on test set...")
        for horizon in forecast_horizons:
            try:
                # Retrain on train + val for final test evaluation
                combined_train = pd.concat([train_data, val_data])
                # Prophet and Darts models need Series with datetime index, others can use values
                if hasattr(model, '__class__'):
                    class_name = model.__class__.__name__
                    if 'ProphetWrapper' in class_name or 'DartsModelWrapper' in class_name:
                        model.fit(combined_train)
                    else:
                        model.fit(combined_train.values)
                else:
                    model.fit(combined_train.values)
                
                test_pred = model.predict(steps=min(horizon, len(test_data)))
                test_actual = test_data.values[:len(test_pred)]
                
                test_metrics = self.calculate_forecast_metrics(
                    test_actual,
                    test_pred,
                    combined_train.values,
                    prefix=f'test_h{horizon}'
                )
                results.update(test_metrics)
                
                predictions_dict[f'horizon_{horizon}'] = {
                    'predictions': test_pred,
                    'actuals': test_actual
                }
            except Exception as e:
                print(f"  Warning: Test failed for horizon {horizon}: {e}")
        
        # Cross-validation (on training data only)
        print("  Performing cross-validation...")
        cv_splits = self.config.get('evaluation', {}).get('cv_splits', 5)
        try:
            cv_results = self.time_series_cross_validate(
                model, 
                train_data, 
                n_splits=cv_splits,
                forecast_horizon=forecast_horizons[0]
            )
            results.update(cv_results)
        except Exception as e:
            print(f"  Warning: Cross-validation failed: {e}")
        
        return results, model, predictions_dict
    
    def plot_forecast(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        horizon: int,
        save_path: str = None,
        confidence_intervals: Tuple[np.ndarray, np.ndarray] = None
    ) -> plt.Figure:
        """
        Plot forecast vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            horizon: Forecast horizon
            save_path: Path to save figure
            confidence_intervals: Tuple of (lower, upper) bounds
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(y_true))
        ax.plot(x, y_true, 'o-', label='Actual', linewidth=2, markersize=4)
        ax.plot(x, y_pred, 's-', label='Forecast', linewidth=2, markersize=4, alpha=0.7)
        
        if confidence_intervals is not None:
            lower, upper = confidence_intervals
            ax.fill_between(x, lower, upper, alpha=0.2, label='Confidence Interval')
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'{model_name} - Forecast vs Actual (Horizon: {horizon})', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(
                os.path.join(save_path, f'{model_name}_forecast_h{horizon}.png'),
                dpi=150,
                bbox_inches='tight'
            )
        
        return fig
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot residual analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residuals over time
        axes[0, 0].plot(residuals, 'o-', alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals vs predicted
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Predicted Value')
        axes[1, 1].set_ylabel('Residual')
        axes[1, 1].set_title('Residuals vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle(f'{model_name} - Residual Analysis', fontsize=16, y=1.00)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(
                os.path.join(save_path, f'{model_name}_residuals.png'),
                dpi=150,
                bbox_inches='tight'
            )
        
        return fig
    
    def compare_models_by_horizon(
        self,
        results_dict: Dict[str, Dict],
        horizons: List[int],
        save_path: str = None
    ) -> plt.Figure:
        """
        Create comparison plots for multiple models across different horizons.
        
        Args:
            results_dict: Dictionary of model results
            horizons: List of forecast horizons
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        model_names = list(results_dict.keys())
        metrics = ['mae', 'rmse', 'mape', 'smape']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # For each model, plot metric across horizons
            for model_name in model_names:
                metric_values = []
                valid_horizons = []
                
                for horizon in horizons:
                    key = f'test_h{horizon}_{metric}'
                    if key in results_dict[model_name]['metrics']:
                        metric_values.append(results_dict[model_name]['metrics'][key])
                        valid_horizons.append(horizon)
                
                if metric_values:
                    ax.plot(valid_horizons, metric_values, 'o-', label=model_name, linewidth=2, markersize=8)
            
            ax.set_xlabel('Forecast Horizon', fontsize=12)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f'{metric.upper()} vs Forecast Horizon', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(
                os.path.join(save_path, 'model_comparison_by_horizon.png'),
                dpi=150,
                bbox_inches='tight'
            )
        
        return fig
    
    def compare_models(
        self,
        results_dict: Dict[str, Dict],
        horizon: int = 1,
        save_path: str = None
    ) -> plt.Figure:
        """
        Create comparison bar plots for multiple models at a specific horizon.
        
        Args:
            results_dict: Dictionary of model results
            horizon: Forecast horizon to compare
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        model_names = list(results_dict.keys())
        metrics = ['mae', 'rmse', 'mape', 'r2']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            metric_key = f'test_h{horizon}_{metric}'
            scores = []
            valid_models = []
            
            for model_name in model_names:
                if metric_key in results_dict[model_name]['metrics']:
                    scores.append(results_dict[model_name]['metrics'][metric_key])
                    valid_models.append(model_name)
            
            if scores:
                axes[idx].bar(valid_models, scores, alpha=0.7, edgecolor='black')
                axes[idx].set_ylabel(metric.upper(), fontsize=12)
                axes[idx].set_title(f'{metric.upper()} (Horizon: {horizon})', fontsize=14)
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(
                os.path.join(save_path, f'model_comparison_h{horizon}.png'),
                dpi=150,
                bbox_inches='tight'
            )
        
        return fig
    
    def log_to_wandb(
        self,
        model_name: str,
        metrics: Dict,
        model: Any,
        config_params: Dict
    ):
        """
        Log metrics and artifacts to Weights & Biases.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
            model: Model instance
            config_params: Configuration parameters
        """
        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                wandb.log({key: value})
        
        # Log model parameters
        wandb.config.update(config_params)
