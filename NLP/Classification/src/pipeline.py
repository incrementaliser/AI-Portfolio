"""
NLP classification pipeline orchestration
"""
import mlflow
import mlflow.sklearn
import os
import joblib
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from src.data_loader import NLPDataLoader
from models.models import NLPModelFactory
from src.evaluate import NLPEvaluator
from src.utils import setup_directories, save_results
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 47
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class NLPClassificationPipeline:
    """Main pipeline for NLP classification model comparison."""
    
    def __init__(self, config: Dict):
        """
        Initialize NLP classification pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        # Resolve all paths relative to script directory
        self._resolve_paths()
        self.data_loader = NLPDataLoader(config)
        self.evaluator = NLPEvaluator(config)
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
                    # Resolve relative to NLP/Classification directory
                    self.config['paths'][path_key] = str(pipeline_dir / path)
                else:
                    self.config['paths'][path_key] = os.path.abspath(path)
    
    def run(
        self,
        data_path: str = None,
        model_names: List[str] = None,
        use_mlflow: bool = False
    ) -> Dict:
        """
        Execute the complete NLP classification pipeline.
        
        Args:
            data_path: Path pattern for review files
            model_names: List of specific model names to run (e.g., ['logistic_regression', 'naive_bayes_multinomial'])
            use_mlflow: Whether to use MLflow logging
            
        Returns:
            Dictionary containing results for all models
        """
        # Setup directories
        setup_directories(self.config)
        
        print("=" * 80)
        print("Starting NLP Classification Pipeline")
        print("=" * 80)
        
        # [1/5] Load and prepare data
        print("\n[1/5] Loading and preparing data...")
        print("-" * 80)
        
        try:
            data = self.data_loader.prepare_data(data_path=data_path, preprocess=True)
            
            X_train = data['X_train']
            X_val = data['X_val']
            X_test = data['X_test']
            y_train = data['y_train']
            y_val = data['y_val']
            y_test = data['y_test']
            
            print(f"Dataset loaded successfully!")
            print(f"  Train size: {len(y_train)}")
            print(f"  Validation size: {len(y_val)}")
            print(f"  Test size: {len(y_test)}")
            print(f"  Feature dimension: {X_train.shape[1]}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please ensure:")
            print("  1. Review files exist at the specified path")
            print("  2. Data path is correct")
            print("  3. Data is properly formatted")
            import traceback
            traceback.print_exc()
            return {}
        
        # [2/5] Create models
        print("\n[2/5] Creating classification models...")
        print("-" * 80)
        
        # Create models - factory will only create models enabled in config
        models = NLPModelFactory.create_models(self.config)
        
        # Filter to only requested models if model_names specified
        if model_names is not None:
            # Map model names to their display names
            model_name_mapping = {
                'logistic_regression': 'LogisticRegression',
                'naive_bayes_multinomial': 'MultinomialNB',
                'naive_bayes_bernoulli': 'BernoulliNB',
                'naive_bayes_gaussian': 'GaussianNB'
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
                    
                    # Re-evaluate loaded model
                    print("  Re-evaluating loaded model...")
                    metrics, trained_model, predictions_dict = self.evaluator.evaluate_model(
                        trained_model,
                        X_train, X_val, X_test,
                        y_train, y_val, y_test
                    )
                except Exception as e:
                    print(f"  Warning: Could not load model, will train new one: {e}")
                    model_loaded = False
                    trained_model = None
            
            # Train new model if not loaded or loading failed
            if not model_loaded:
                if use_mlflow:
                    # Initialize MLflow run for this model
                    try:
                        # Set tracking URI if specified in config
                        tracking_uri = self.config.get('mlflow', {}).get('tracking_uri', None)
                        if tracking_uri:
                            mlflow.set_tracking_uri(tracking_uri)
                        
                        # Set experiment name
                        experiment_name = self.config['project']['name']
                        try:
                            mlflow.set_experiment(experiment_name)
                        except Exception:
                            # Experiment might already exist, which is fine
                            pass
                        
                        # Start MLflow run
                        mlflow.start_run(run_name=f"{model_name}")
                        
                        # Log model parameters
                        model_params = NLPModelFactory.get_model_params(model)
                        mlflow.log_params(model_params)
                    except Exception as e:
                        print(f"  Warning: Could not initialize MLflow: {e}")
                        use_mlflow = False
                
                try:
                    # Evaluate model
                    metrics, trained_model, predictions_dict = self.evaluator.evaluate_model(
                        model,
                        X_train, X_val, X_test,
                        y_train, y_val, y_test
                    )
                    
                    if not metrics:
                        print(f"  Skipping {model_name} due to evaluation errors")
                        if use_mlflow:
                            mlflow.end_run()
                        continue
                    
                    # Save newly trained model
                    try:
                        joblib.dump(trained_model, model_path)
                        print(f"  Model saved to: {model_path}")
                    except Exception as e:
                        print(f"  Warning: Could not save model: {e}")
                    
                    # Log to MLflow for newly trained models
                    if use_mlflow:
                        try:
                            # Log metrics
                            for key, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    mlflow.log_metric(key, value)
                            
                            # Log model artifact
                            # Note: registered_model_name requires MLflow Model Registry (optional)
                            try:
                                mlflow.sklearn.log_model(
                                    trained_model,
                                    artifact_path="model",
                                    registered_model_name=f"{model_name}"
                                )
                            except Exception:
                                # If model registry is not available, just log the model without registration
                                mlflow.sklearn.log_model(
                                    trained_model,
                                    artifact_path="model"
                                )
                            
                            # Log confusion matrix
                            if 'test' in predictions_dict:
                                test_pred = predictions_dict['test']['predictions']
                                test_actual = predictions_dict['test']['actuals']
                                fig = self.evaluator.plot_confusion_matrix(
                                    test_actual, test_pred, model_name
                                )
                                # Save figure temporarily for MLflow
                                cm_path = os.path.join(self.config['paths']['figures_dir'], 
                                                      f"{model_name}_confusion_matrix_temp.png")
                                fig.savefig(cm_path)
                                mlflow.log_artifact(cm_path, "figures")
                                os.remove(cm_path)  # Clean up temp file
                                fig.clf()
                            
                            # Log ROC curve if probabilities available
                            if 'test' in predictions_dict and predictions_dict['test']['probabilities'] is not None:
                                test_proba = predictions_dict['test']['probabilities']
                                test_actual = predictions_dict['test']['actuals']
                                fig = self.evaluator.plot_roc_curve(
                                    test_actual, test_proba, model_name
                                )
                                # Save figure temporarily for MLflow
                                roc_path = os.path.join(self.config['paths']['figures_dir'],
                                                       f"{model_name}_roc_curve_temp.png")
                                fig.savefig(roc_path)
                                mlflow.log_artifact(roc_path, "figures")
                                os.remove(roc_path)  # Clean up temp file
                                fig.clf()
                            
                        except Exception as e:
                            print(f"  Warning: Could not log to MLflow: {e}")
                            import traceback
                            traceback.print_exc()
                        
                        mlflow.end_run()
                        
                except Exception as e:
                    print(f"  Error training {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    if use_mlflow:
                        mlflow.end_run()
                    continue
            
            # Store results (for both loaded and newly trained models)
            if metrics:
                self.results[model_name] = {
                    'model': trained_model,
                    'metrics': metrics,
                    'predictions': predictions_dict,
                    'model_type': model.model_type if hasattr(model, 'model_type') else (trained_model.model_type if hasattr(trained_model, 'model_type') else 'unknown')
                }
                
                # Print key metrics
                print(f"\n  Results:")
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                    key = f'test_{metric}'
                    if key in metrics:
                        print(f"    {metric.upper()}: {metrics[key]:.4f}")
                
                # Print CV results if available
                if 'cv_f1_mean' in metrics:
                    print(f"\n  Cross-validation (F1): {metrics['cv_f1_mean']:.4f} "
                          f"(±{metrics['cv_f1_std']:.4f})")
        
        if not self.results:
            print("\nNo models were successfully trained!")
            return {}
        
        # [4/5] Compare models
        print("\n[4/5] Comparing models...")
        print("-" * 80)
        
        try:
            # Create comparison plot
            comparison_fig = self.evaluator.compare_models(
                self.results,
                save_path=self.config['paths']['figures_dir']
            )
            comparison_fig.clf()
            plt.close(comparison_fig)
            
            # Create confusion matrices and ROC curves for each model
            for model_name, result_data in self.results.items():
                if 'test' in result_data['predictions']:
                    pred_data = result_data['predictions']['test']
                    test_pred = pred_data['predictions']
                    test_actual = pred_data['actuals']
                    
                    # Confusion matrix
                    fig = self.evaluator.plot_confusion_matrix(
                        test_actual, test_pred, model_name,
                        save_path=self.config['paths']['figures_dir']
                    )
                    fig.clf()
                    plt.close(fig)
                    
                    # ROC curve if probabilities available
                    if pred_data['probabilities'] is not None:
                        fig = self.evaluator.plot_roc_curve(
                            test_actual, pred_data['probabilities'], model_name,
                            save_path=self.config['paths']['figures_dir']
                        )
                        fig.clf()
                        plt.close(fig)
                    
        except Exception as e:
            print(f"Warning: Could not create comparison plots: {e}")
            import traceback
            traceback.print_exc()
        
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
        
        # Find best model based on F1 score
        best_model_name = self._get_best_model(
            metric='test_f1',
            maximize=True
        )
        
        print("\n" + "=" * 80)
        print(f"PIPELINE COMPLETED!")
        print("=" * 80)
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Model Type: {self.results[best_model_name]['model_type']}")
        
        best_metrics = self.results[best_model_name]['metrics']
        print(f"\nBest Model Metrics:")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            key = f'test_{metric}'
            if key in best_metrics:
                print(f"  {metric.upper()}: {best_metrics[key]:.4f}")
        
        print("\n" + "=" * 80)
        
        return self.results
    
    def _get_best_model(self, metric: str = 'test_f1', maximize: bool = True) -> str:
        """
        Find the best performing model based on a metric.
        
        Args:
            metric: Metric name to compare
            maximize: Whether higher values are better
            
        Returns:
            Name of best model
        """
        if not self.results:
            return None
        
        valid_models = {
            name: data['metrics'].get(metric, float('-inf') if maximize else float('inf'))
            for name, data in self.results.items()
            if metric in data['metrics']
        }
        
        if not valid_models:
            return list(self.results.keys())[0]
        
        if maximize:
            best_model = max(valid_models.keys(), key=lambda k: valid_models[k])
        else:
            best_model = min(valid_models.keys(), key=lambda k: valid_models[k])
        
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
        texts: List[str]
    ) -> np.ndarray:
        """
        Make predictions with a saved model.
        
        Args:
            model_name: Name of the model to use
            texts: List of texts to predict on
            
        Returns:
            Predictions
        """
        model = self.load_model(model_name)
        
        # Preprocess texts
        processed_texts = self.data_loader.preprocess_texts(texts)
        
        # Vectorize texts
        X = self.data_loader.vectorize_texts(processed_texts, fit=False)
        
        return model.predict(X)


