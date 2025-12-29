"""
Time series forecasting pipeline orchestration
"""
import wandb
import os
import joblib
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from src.data_loader import TimeSeriesDataLoader
from models.models import TimeSeriesModelFactory
from src.evaluate import TimeSeriesEvaluator
from src.utils import setup_directories, save_results
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 47
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class TimeSeriesPipeline:
    """Main pipeline for time series forecasting model comparison."""
    
    def __init__(self, config: Dict):
        """
        Initialize time series pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        # Resolve all paths relative to script directory
        self._resolve_paths()
        self.data_loader = TimeSeriesDataLoader(config)
        self.evaluator = TimeSeriesEvaluator(config)
        self.results = {}
    
    def _resolve_paths(self):
        """Resolve all paths in config to absolute paths relative to script location."""
        import sys
        from pathlib import Path
        
        # Get the directory where the pipeline module is located
        pipeline_dir = Path(__file__).parent.parent.absolute()
        
        # Resolve all paths
        for path_key in ['data_raw', 'data_processed', 'models_dir', 'figures_dir', 'metrics_dir']:
            if path_key in self.config.get('paths', {}):
                path = self.config['paths'][path_key]
                if not os.path.isabs(path):
                    # Resolve relative to TimeSeries directory
                    self.config['paths'][path_key] = str(pipeline_dir / path)
                else:
                    self.config['paths'][path_key] = os.path.abspath(path)
        
    def run(
        self, 
        csv_path: str = None,
        timestamp_col: str = None,
        target_cols: List[str] = None,
        model_names: List[str] = None,
        use_wandb: bool = True
    ) -> Dict:
        """
        Execute the complete time series forecasting pipeline.
        
        Args:
            csv_path: Path to CSV file with time series data
            timestamp_col: Name of timestamp column
            target_cols: List of target column names
            model_names: List of specific model names to run (e.g., ['xgboost', 'arima'])
            use_wandb: Whether to use Weights & Biases logging
            
        Returns:
            Dictionary containing results for all models
        """
        # Setup directories
        setup_directories(self.config)
        
        print("=" * 80)
        print("Starting Time Series Forecasting Pipeline")
        print("=" * 80)
        
        # [1/5] Load and prepare data
        print("\n[1/5] Loading and preparing time series data...")
        print("-" * 80)
        
        try:
            data = self.data_loader.prepare_data(
                filepath=csv_path,
                timestamp_col=timestamp_col,
                target_cols=target_cols,
                apply_scaling=True
            )
            
            train_data = data['train']
            val_data = data['val']
            test_data = data['test']
            
            # Get target column(s)
            if target_cols is None:
                target_cols = self.config['data'].get('target_cols', ['value'])
            if isinstance(target_cols, str):
                target_cols = [target_cols]
            
            # For univariate forecasting, extract the first target
            target_col = target_cols[0]
            
            print(f"Dataset loaded successfully!")
            print(f"  Total samples: {len(train_data) + len(val_data) + len(test_data)}")
            print(f"  Train size: {len(train_data)}")
            print(f"  Validation size: {len(val_data)}")
            print(f"  Test size: {len(test_data)}")
            print(f"  Target column: {target_col}")
            print(f"  Date range: {train_data.index[0]} to {test_data.index[-1]}")
            
            # Extract target series
            train_series = train_data[target_col]
            val_series = val_data[target_col]
            test_series = test_data[target_col]
            
            # Save processed data
            self.data_loader.save_processed_data(
                data,
                self.config['paths']['data_processed']
            )
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please ensure:")
            print("  1. CSV file exists at the specified path")
            print("  2. Timestamp column and target columns are correct")
            print("  3. Data is properly formatted")
            return {}
        
        # [2/5] Create models
        print("\n[2/5] Creating time series models...")
        print("-" * 80)
        
        # Create models - factory will only create models enabled in config
        models = TimeSeriesModelFactory.create_models(self.config)
        
        # Filter to only requested models if model_names specified
        if model_names is not None:
            # Map model names to their display names
            model_name_mapping = {
                'arima': 'ARIMA',
                'prophet': 'Prophet',
                'ets': 'ETS',
                'theta': 'Theta',
                'random_forest': 'RandomForest',
                'gradient_boosting': 'GradientBoosting',
                'xgboost': 'XGBoost',
                'lightgbm': 'LightGBM',
                'ridge': 'Ridge',
                'lstm': 'LSTM',
                'gru': 'GRU',
                'nbeats': 'NBEATS',
                'transformer': 'Transformer',
                'tcn': 'TCN'
            }
            
            filtered_models = {}
            for model_name in model_names:
                display_name = model_name_mapping.get(model_name, model_name.capitalize())
                if display_name in models:
                    filtered_models[display_name] = models[display_name]
                else:
                    print(f"  Warning: Model '{model_name}' not found in created models")
            
            models = filtered_models
        
        if not models:
            print("No models to train! Check your configuration.")
            return {}
        
        print(f"Models to train: {list(models.keys())}")
        print(f"Model types: {set([m.model_type for m in models.values()])}")
        
        # Get forecast horizons
        forecast_horizons = self.config['data'].get('forecast_horizons', [1])
        print(f"Forecast horizons: {forecast_horizons}")
        
        # [3/5] Train and evaluate each model
        print("\n[3/5] Training and evaluating models...")
        print("-" * 80)
        
        for idx, (model_name, model) in enumerate(models.items(), 1):
            print(f"\n[Model {idx}/{len(models)}] {model_name}")
            print("=" * 60)
            print(f"  Model type: {model.model_type}")
            
            # Check if model already exists
            model_path = os.path.join(
                self.config['paths']['models_dir'],
                f"{model_name}.pkl"
            )
            model_path = os.path.abspath(model_path)
            
            metrics = {}
            predictions_dict = {}
            trained_model = None
            model_loaded = False
            
            # Try to load existing model
            if os.path.exists(model_path):
                print(f"  Loading existing model from: {model_path}")
                try:
                    trained_model = joblib.load(model_path)
                    model_loaded = True
                    print("  Model loaded successfully")
                    
                    # Try to load metrics from saved results instead of re-evaluating
                    # (Re-evaluation can fail for models like Prophet that can't be refit)
                    metrics_file = os.path.join(
                        self.config['paths']['metrics_dir'],
                        'results_summary.json'
                    )
                    metrics_file = os.path.abspath(metrics_file)
                    
                    if os.path.exists(metrics_file):
                        try:
                            import json
                            with open(metrics_file, 'r') as f:
                                saved_results = json.load(f)
                            if model_name in saved_results:
                                metrics = saved_results[model_name].get('metrics', {})
                                
                                # Check if test metrics exist for all horizons
                                forecast_horizons = self.config['data'].get('forecast_horizons', [1])
                                has_all_test_metrics = all(
                                    any(f'test_h{h}_{m}' in metrics for m in ['mae', 'rmse'])
                                    for h in forecast_horizons
                                )
                                
                                if has_all_test_metrics:
                                    print("  Loaded metrics from saved results (including test metrics)")
                                    # Metrics are loaded, but we still need predictions for visualization
                                    # Re-evaluate to get predictions
                                    print("  Generating predictions for visualization...")
                                    try:
                                        if hasattr(trained_model, '__class__') and 'ProphetWrapper' in trained_model.__class__.__name__:
                                            print("  Creating fresh Prophet instance for prediction generation...")
                                            from models.statistical.statistical import ProphetWrapper
                                            if hasattr(trained_model, 'get_params'):
                                                prophet_params = trained_model.get_params()
                                            else:
                                                prophet_params = getattr(trained_model, 'prophet_params', {})
                                            fresh_model = ProphetWrapper(**prophet_params)
                                            _, _, predictions_dict = self.evaluator.evaluate_model(
                                                fresh_model,
                                                train_series,
                                                val_series,
                                                test_series,
                                                forecast_horizons=forecast_horizons
                                            )
                                        else:
                                            _, _, predictions_dict = self.evaluator.evaluate_model(
                                                trained_model,
                                                train_series,
                                                val_series,
                                                test_series,
                                                forecast_horizons=forecast_horizons
                                            )
                                    except Exception as pred_error:
                                        print(f"  Warning: Could not generate predictions: {pred_error}")
                                        predictions_dict = {}
                                else:
                                    print("  Loaded metrics from saved results, but test metrics are missing")
                                    print("  Will re-evaluate to get test metrics...")
                                    metrics = {}  # Clear metrics to force re-evaluation
                        except Exception as e:
                            print(f"  Could not load saved metrics: {e}")
                            metrics = {}
                    
                    # If no saved metrics or test metrics are missing, try to re-evaluate
                    if not metrics:
                        print("  Re-evaluating loaded model...")
                        try:
                            # For Prophet, we need to create a fresh instance since it can't be refit
                            if hasattr(trained_model, '__class__') and 'ProphetWrapper' in trained_model.__class__.__name__:
                                print("  Creating fresh Prophet instance for evaluation...")
                                from models.statistical.statistical import ProphetWrapper
                                if hasattr(trained_model, 'get_params'):
                                    prophet_params = trained_model.get_params()
                                else:
                                    prophet_params = getattr(trained_model, 'prophet_params', {})
                                fresh_model = ProphetWrapper(**prophet_params)
                                metrics, trained_model, predictions_dict = self.evaluator.evaluate_model(
                                    fresh_model,
                                    train_series,
                                    val_series,
                                    test_series,
                                    forecast_horizons=forecast_horizons
                                )
                            else:
                                metrics, trained_model, predictions_dict = self.evaluator.evaluate_model(
                                    trained_model,
                                    train_series,
                                    val_series,
                                    test_series,
                                    forecast_horizons=forecast_horizons
                                )
                        except Exception as eval_error:
                            print(f"  Warning: Re-evaluation failed: {eval_error}")
                            import traceback
                            traceback.print_exc()
                            print("  Using saved metrics if available, or skipping this model")
                            metrics = {}
                            predictions_dict = {}
                    
                    if not metrics:
                        print(f"  Warning: No metrics available for loaded model, will train new one")
                        model_loaded = False
                        trained_model = None
                except Exception as e:
                    print(f"  Warning: Could not load model, will train new one: {e}")
                    import traceback
                    traceback.print_exc()
                    model_loaded = False
                    trained_model = None
            
            # Train new model if not loaded or loading failed
            if not model_loaded:
                if use_wandb:
                    # Initialize wandb run for this model
                    try:
                        run = wandb.init(
                            project=self.config['project']['name'],
                            entity=self.config['project'].get('entity'),
                            name=f"{model_name}",
                            config=TimeSeriesModelFactory.get_model_params(model),
                            reinit=True
                        )
                    except Exception as e:
                        print(f"  Warning: Could not initialize wandb: {e}")
                        use_wandb = False
                
                try:
                    # Evaluate model
                    metrics, trained_model, predictions_dict = self.evaluator.evaluate_model(
                        model,
                        train_series,
                        val_series,
                        test_series,
                        forecast_horizons=forecast_horizons
                    )
                    
                    if not metrics:
                        print(f"  Skipping {model_name} due to evaluation errors")
                        if use_wandb:
                            wandb.finish()
                        continue
                    
                    # Save newly trained model
                    try:
                        joblib.dump(trained_model, model_path)
                        print(f"  Model saved to: {model_path}")
                    except Exception as e:
                        print(f"  Warning: Could not save model: {e}")
                    
                    # Log to wandb for newly trained models
                    if use_wandb:
                        try:
                            self.evaluator.log_to_wandb(
                                model_name,
                                metrics,
                                trained_model,
                                TimeSeriesModelFactory.get_model_params(model)
                            )
                            
                            # Plot and log forecast for each horizon
                            for horizon in forecast_horizons[:2]:  # Log first 2 horizons
                                if f'horizon_{horizon}' in predictions_dict:
                                    pred_data = predictions_dict[f'horizon_{horizon}']
                                    fig = self.evaluator.plot_forecast(
                                        pred_data['actuals'],
                                        pred_data['predictions'],
                                        model_name,
                                        horizon,
                                        self.config['paths']['figures_dir']
                                    )
                                    wandb.log({f"{model_name}_forecast_h{horizon}": wandb.Image(fig)})
                                    fig.clf()
                            
                            # Plot and log residuals for first horizon
                            first_pred_data = predictions_dict[f'horizon_{forecast_horizons[0]}']
                            fig = self.evaluator.plot_residuals(
                                first_pred_data['actuals'],
                                first_pred_data['predictions'],
                                model_name,
                                self.config['paths']['figures_dir']
                            )
                            wandb.log({f"{model_name}_residuals": wandb.Image(fig)})
                            fig.clf()
                            
                        except Exception as e:
                            print(f"  Warning: Could not log to wandb: {e}")
                        
                        wandb.finish()
                        
                except Exception as e:
                    print(f"  Error training {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    if use_wandb:
                        wandb.finish()
                    continue
            
            # Store results (for both loaded and newly trained models)
            if metrics:
                self.results[model_name] = {
                    'model': trained_model,
                    'metrics': metrics,
                    'predictions': predictions_dict,
                    'model_type': model.model_type if hasattr(model, 'model_type') else (trained_model.model_type if hasattr(trained_model, 'model_type') else 'unknown')
                }
                
                # Debug: print prediction keys
                print(f"  Stored predictions for {model_name}: {list(predictions_dict.keys())}")
                for key, pred in predictions_dict.items():
                    if isinstance(pred, dict):
                        train_pred_len = len(pred.get('train_predictions', []))
                        test_pred_len = len(pred.get('test_predictions', []))
                        print(f"    {key}: train_pred={train_pred_len}, test_pred={test_pred_len}")
                
                # Print key metrics for first horizon
                first_horizon = forecast_horizons[0]
                print(f"\n  Results for horizon {first_horizon}:")
                for metric in ['mae', 'rmse', 'mape', 'r2']:
                    key = f'test_h{first_horizon}_{metric}'
                    if key in metrics:
                        print(f"    {metric.upper()}: {metrics[key]:.4f}")
                
                # Print CV results if available
                if 'cv_mae_mean' in metrics:
                    print(f"\n  Cross-validation (MAE): {metrics['cv_mae_mean']:.4f} "
                          f"(±{metrics['cv_mae_std']:.4f})")
        
        if not self.results:
            print("\nNo models were successfully trained!")
            return {}
        
        # [4/5] Compare models
        print("\n[4/5] Comparing models...")
        print("-" * 80)
        
        # Load results from all previously trained models for comparison
        self._load_previous_results(train_series, val_series, test_series, forecast_horizons)
        
        # Create train/test comparison plot with all horizons
        try:
            print(f"  Creating train/test comparison plot for all horizons...")
            comparison_fig = self.evaluator.plot_train_test_comparison(
                self.results,
                train_series,
                test_series,
                save_path=self.config['paths']['figures_dir'],
                horizons=forecast_horizons
            )
            comparison_fig.clf()
            plt.close(comparison_fig)
        except Exception as e:
            print(f"  Warning: Could not create train/test comparison plot: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # Compare models across all horizons
            comparison_fig = self.evaluator.compare_models(
                self.results,
                horizons=forecast_horizons,
                save_path=self.config['paths']['figures_dir']
            )
            
            if use_wandb:
                try:
                    run = wandb.init(
                        project=self.config['project']['name'],
                        entity=self.config['project'].get('entity'),
                        name="model_comparison",
                        reinit=True
                    )
                    wandb.log({"model_comparison": wandb.Image(comparison_fig)})
                except:
                    pass
            
            comparison_fig.clf()
            
            # Compare models across horizons
            if len(forecast_horizons) > 1:
                horizon_comparison_fig = self.evaluator.compare_models_by_horizon(
                    self.results,
                    forecast_horizons,
                    save_path=self.config['paths']['figures_dir']
                )
                
                if use_wandb:
                    try:
                        wandb.log({"horizon_comparison": wandb.Image(horizon_comparison_fig)})
                    except:
                        pass
                
                horizon_comparison_fig.clf()
            
            if use_wandb:
                try:
                    wandb.finish()
                except:
                    pass
                    
        except Exception as e:
            print(f"Warning: Could not create comparison plots: {e}")
        
        # [5/5] Save results and determine best model
        print("\n[5/5] Saving results...")
        print("-" * 80)
        
        # Ensure metrics_dir is absolute before saving
        metrics_dir = self.config['paths']['metrics_dir']
        if not os.path.isabs(metrics_dir):
            metrics_dir = os.path.abspath(metrics_dir)
            self.config['paths']['metrics_dir'] = metrics_dir
        
        print(f"  Saving results to: {metrics_dir}")
        save_results(self.results, metrics_dir, append=True)
        
        # Find best model based on first horizon MAE
        best_model_name = self._get_best_model(
            metric=f'test_h{forecast_horizons[0]}_mae',
            minimize=True
        )
        
        print("\n" + "=" * 80)
        print(f"PIPELINE COMPLETED!")
        print("=" * 80)
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Model Type: {self.results[best_model_name]['model_type']}")
        
        for horizon in forecast_horizons:
            mae_key = f'test_h{horizon}_mae'
            rmse_key = f'test_h{horizon}_rmse'
            if mae_key in self.results[best_model_name]['metrics']:
                mae = self.results[best_model_name]['metrics'][mae_key]
                rmse = self.results[best_model_name]['metrics'][rmse_key]
                print(f"\nHorizon {horizon}:")
                print(f"  MAE:  {mae:.4f}")
                print(f"  RMSE: {rmse:.4f}")
        
        print("\n" + "=" * 80)
        
        return self.results
    
    def _load_previous_results(
        self,
        train_series: pd.Series,
        val_series: pd.Series,
        test_series: pd.Series,
        forecast_horizons: List[int]
    ):
        """
        Load results from previously trained models that are not in the current experiment.
        
        This ensures that comparison plots include all trained models, not just
        models from the current experiment.
        
        Args:
            train_series: Training data series
            val_series: Validation data series
            test_series: Test data series
            forecast_horizons: List of forecast horizons
        """
        import json
        
        # Check for saved results
        metrics_file = os.path.join(
            self.config['paths']['metrics_dir'],
            'results_summary.json'
        )
        metrics_file = os.path.abspath(metrics_file)
        
        if not os.path.exists(metrics_file):
            return
        
        try:
            with open(metrics_file, 'r') as f:
                saved_results = json.load(f)
        except Exception as e:
            print(f"  Could not load saved results: {e}")
            return
        
        # Find models that have saved results but are not in current results
        models_to_load = [
            model_name for model_name in saved_results.keys()
            if model_name not in self.results
        ]
        
        if not models_to_load:
            return
        
        print(f"  Loading {len(models_to_load)} previously trained model(s) for comparison: {models_to_load}")
        
        for model_name in models_to_load:
            model_path = os.path.join(
                self.config['paths']['models_dir'],
                f"{model_name}.pkl"
            )
            model_path = os.path.abspath(model_path)
            
            if not os.path.exists(model_path):
                print(f"    Skipping {model_name}: model file not found")
                continue
            
            try:
                # Load the saved model
                trained_model = joblib.load(model_path)
                saved_metrics = saved_results[model_name].get('metrics', {})
                
                # Generate predictions for visualization
                predictions_dict = {}
                combined_train = pd.concat([train_series, val_series])
                
                for horizon in forecast_horizons:
                    try:
                        # Check model type and handle appropriately
                        if hasattr(trained_model, '__class__'):
                            class_name = trained_model.__class__.__name__
                            if 'ProphetWrapper' in class_name:
                                # Create fresh Prophet instance
                                from models.statistical.statistical import ProphetWrapper
                                if hasattr(trained_model, 'get_params'):
                                    prophet_params = trained_model.get_params()
                                else:
                                    prophet_params = getattr(trained_model, 'prophet_params', {})
                                fresh_model = ProphetWrapper(**prophet_params)
                                fresh_model.fit(combined_train)
                                test_pred = fresh_model.predict(steps=min(horizon, len(test_series)))
                            elif 'DartsModelWrapper' in class_name:
                                # Create fresh Darts instance
                                from models.neural.deep_learning import DartsModelWrapper
                                if hasattr(trained_model, 'model_class') and hasattr(trained_model, 'input_chunk_length'):
                                    fresh_model = DartsModelWrapper(
                                        model_class=trained_model.model_class,
                                        input_chunk_length=trained_model.input_chunk_length,
                                        output_chunk_length=trained_model.output_chunk_length,
                                        **getattr(trained_model, 'model_params', {})
                                    )
                                    fresh_model.fit(combined_train)
                                    test_pred = fresh_model.predict(steps=min(horizon, len(test_series)))
                                else:
                                    trained_model.fit(combined_train)
                                    test_pred = trained_model.predict(steps=min(horizon, len(test_series)))
                            else:
                                # ML models - refit with combined data
                                trained_model.fit(combined_train.values)
                                test_pred = trained_model.predict(steps=min(horizon, len(test_series)))
                        else:
                            trained_model.fit(combined_train.values)
                            test_pred = trained_model.predict(steps=min(horizon, len(test_series)))
                        
                        test_actual = test_series.values[:len(test_pred)]
                        
                        predictions_dict[f'horizon_{horizon}'] = {
                            'test_predictions': test_pred,
                            'test_actuals': test_actual,
                            'train_predictions': np.array([]),
                            'train_actuals': train_series.values
                        }
                    except Exception as e:
                        print(f"    Warning: Could not generate predictions for {model_name} horizon {horizon}: {e}")
                        continue
                
                # Add to results if predictions were generated
                if predictions_dict:
                    self.results[model_name] = {
                        'model': trained_model,
                        'metrics': saved_metrics,
                        'predictions': predictions_dict,
                        'model_type': getattr(trained_model, 'model_type', 'unknown')
                    }
                    print(f"    Loaded {model_name} with predictions for horizons: {list(predictions_dict.keys())}")
                else:
                    print(f"    Warning: Could not generate any predictions for {model_name}")
                    
            except Exception as e:
                print(f"    Error loading {model_name}: {e}")
                continue
    
    def _get_best_model(self, metric: str = 'test_h1_mae', minimize: bool = True) -> str:
        """
        Find the best performing model based on a metric.
        
        Args:
            metric: Metric name to compare
            minimize: Whether lower values are better
            
        Returns:
            Name of best model
        """
        if not self.results:
            return None
        
        valid_models = {
            name: data['metrics'].get(metric, float('inf') if minimize else float('-inf'))
            for name, data in self.results.items()
            if metric in data['metrics']
        }
        
        if not valid_models:
            return list(self.results.keys())[0]
        
        if minimize:
            best_model = min(valid_models.keys(), key=lambda k: valid_models[k])
        else:
            best_model = max(valid_models.keys(), key=lambda k: valid_models[k])
        
        return best_model
    
    def load_model(self, model_name: str) -> any:
        """
        Load a saved model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model
        """
        model_path = os.path.join(
            self.config['paths']['models_dir'],
            f"{model_name}.pkl"
        )
        return joblib.load(model_path)
    
    def predict_with_model(
        self,
        model_name: str,
        steps: int,
        data: pd.Series = None
    ) -> np.ndarray:
        """
        Make predictions with a saved model.
        
        Args:
            model_name: Name of the model to use
            steps: Number of steps to forecast
            data: Optional data to use for prediction (if None, uses test data)
            
        Returns:
            Predictions
        """
        model = self.load_model(model_name)
        
        if data is not None:
            model.fit(data.values)
        
        return model.predict(steps=steps)
