"""
NLP classification pipeline orchestration.

Supports:
- Traditional ML models (using TF-IDF features)
- Transformer-based LM models (using raw text, fine-tuning)
- Prompt-based LLM models (using raw text, zero-shot/few-shot/CoT)
"""
import mlflow
import mlflow.sklearn
import os
import json
import joblib
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from src.data_loader import NLPDataLoader
from models.factory import NLPModelFactory
from src.evaluate import NLPEvaluator
from src.utils import setup_directories, save_results
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 47
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class NLPClassificationPipeline:
    """
    Main pipeline for NLP classification model comparison.
    
    Supports:
    - Traditional ML models (Logistic Regression, Naive Bayes)
    - Transformer-based LM models (BERT, DistilBERT)
    - Prompt-based LLM models (Zero-shot, Few-shot, Chain-of-Thought)
    """
    
    # Model name mapping from config keys to display names
    MODEL_NAME_MAPPING = {
        # ML models
        'logistic_regression': 'LogisticRegression',
        'naive_bayes_multinomial': 'MultinomialNB',
        'naive_bayes_bernoulli': 'BernoulliNB',
        'naive_bayes_gaussian': 'GaussianNB',
        # LM models (fine-tuning)
        'bert_base': 'BERT',
        'distilbert': 'DistilBERT',
        'roberta_base': 'RoBERTa',
        # Prompting models
        'zero_shot_mistral': 'ZeroShot_Mistral',
        'few_shot_mistral': 'FewShot_Mistral',
        'cot_mistral': 'CoT_Mistral',
        'zero_shot_phi3': 'ZeroShot_Phi3',
        'few_shot_phi3': 'FewShot_Phi3',
        'cot_phi3': 'CoT_Phi3',
        'zero_shot': 'ZeroShot',
        'few_shot': 'FewShot',
        'chain_of_thought': 'ChainOfThought',
    }
    
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
        self._data_cache = None  # Cache for loaded data
    
    def _resolve_paths(self) -> None:
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
    
    def _is_lm_model(self, model: Any) -> bool:
        """
        Check if a model requires raw text input (LM or prompting model).
        
        Args:
            model: Model instance
            
        Returns:
            True if the model requires raw text (LM or prompting model)
        """
        if not hasattr(model, 'model_type'):
            return False
        return model.model_type.startswith('lm') or model.model_type.startswith('prompting')
    
    def _is_prompting_model(self, model: Any) -> bool:
        """
        Check if a model is a prompting model (zero-shot, few-shot, CoT).
        
        Args:
            model: Model instance
            
        Returns:
            True if the model is a prompting model
        """
        return hasattr(model, 'model_type') and model.model_type.startswith('prompting')
    
    def _train_lm_model(
        self,
        model: Any,
        train_texts: List[str],
        y_train: np.ndarray,
        val_texts: List[str],
        y_val: np.ndarray,
        test_texts: List[str],
        y_test: np.ndarray
    ) -> tuple:
        """
        Train and evaluate a language model or prompting model.
        
        For transformer models (BERT, etc.): Fine-tunes the model
        For prompting models: Selects examples (few-shot) or no training (zero-shot)
        
        Args:
            model: LM or prompting model instance
            train_texts: Training texts (raw)
            y_train: Training labels
            val_texts: Validation texts (raw)
            y_val: Validation labels
            test_texts: Test texts (raw)
            y_test: Test labels
            
        Returns:
            Tuple of (metrics_dict, trained_model, predictions_dict)
        """
        is_prompting = self._is_prompting_model(model)
        
        if is_prompting:
            print("  Setting up prompting model...")
        else:
            print("  Training transformer model (this may take a while)...")
        
        # Train the model with validation data
        model.fit(train_texts, y_train, val_texts, y_val)
        
        # Get predictions
        print("  Generating predictions...")
        predictions_dict = {}
        
        # Validation predictions
        val_pred = model.predict(val_texts)
        val_pred_proba = model.predict_proba(val_texts)
        predictions_dict['val'] = {
            'predictions': val_pred,
            'probabilities': val_pred_proba,
            'actuals': y_val
        }
        
        # Test predictions
        test_pred = model.predict(test_texts)
        test_pred_proba = model.predict_proba(test_texts)
        predictions_dict['test'] = {
            'predictions': test_pred,
            'probabilities': test_pred_proba,
            'actuals': y_test
        }
        
        # Calculate metrics
        print("  Calculating metrics...")
        metrics = {}
        
        # Validation metrics
        val_metrics = self.evaluator.calculate_all_metrics(
            y_val, val_pred, val_pred_proba, prefix='val'
        )
        metrics.update(val_metrics)
        
        # Test metrics
        test_metrics = self.evaluator.calculate_all_metrics(
            y_test, test_pred, test_pred_proba, prefix='test'
        )
        metrics.update(test_metrics)
        
        return metrics, model, predictions_dict
    
    def _save_lm_model(self, model: Any, model_name: str, model_path: str) -> None:
        """
        Save an LM model to disk.
        
        Args:
            model: LM model instance
            model_name: Display name of the model
            model_path: Path to save the model (used as base for directory)
        """
        # LM models save to a directory, not a single file
        lm_model_dir = model_path.replace('.pkl', '')
        model.save(lm_model_dir)
        
        # Also save a marker file for compatibility
        with open(model_path, 'w') as f:
            json.dump({'type': 'lm', 'path': lm_model_dir}, f)
    
    def _load_lm_model(self, model: Any, model_path: str) -> Any:
        """
        Load an LM model from disk.
        
        Args:
            model: LM model instance (for type reference)
            model_path: Path to the model marker file
            
        Returns:
            Loaded model instance
        """
        # Read the marker file to get the actual model directory
        with open(model_path, 'r') as f:
            marker = json.load(f)
        
        lm_model_dir = marker.get('path', model_path.replace('.pkl', ''))
        model.load(lm_model_dir)
        return model
    
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
            model_names: List of specific model names to run
                        (e.g., ['logistic_regression', 'bert_base'])
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
            self._data_cache = data  # Cache for LM models
            
            X_train = data['X_train']
            X_val = data['X_val']
            X_test = data['X_test']
            y_train = data['y_train']
            y_val = data['y_val']
            y_test = data['y_test']
            
            # Also get raw texts for LM models
            train_texts = data['train_df']['review'].tolist()
            val_texts = data['val_df']['review'].tolist()
            test_texts = data['test_df']['review'].tolist()
            
            print(f"Dataset loaded successfully!")
            print(f"  Train size: {len(y_train)}")
            print(f"  Validation size: {len(y_val)}")
            print(f"  Test size: {len(y_test)}")
            print(f"  Feature dimension (TF-IDF): {X_train.shape[1]}")
            
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
            filtered_models = {}
            for model_name in model_names:
                display_name = self.MODEL_NAME_MAPPING.get(model_name, model_name)
                if display_name in models:
                    filtered_models[display_name] = models[display_name]
                else:
                    print(f"  Warning: Model '{model_name}' not found in created models")
            
            models = filtered_models
        
        if not models:
            print("No models to train! Check your configuration.")
            return {}
        
        # Separate ML and LM models
        ml_models = {k: v for k, v in models.items() if not self._is_lm_model(v)}
        lm_models = {k: v for k, v in models.items() if self._is_lm_model(v)}
        
        print(f"Models to train: {list(models.keys())}")
        if ml_models:
            print(f"  ML models: {list(ml_models.keys())}")
        if lm_models:
            print(f"  LM models: {list(lm_models.keys())}")
        
        # [3/5] Train and evaluate each model
        print("\n[3/5] Training and evaluating models...")
        print("-" * 80)
        
        all_models = list(models.items())
        for idx, (model_name, model) in enumerate(all_models, 1):
            print(f"\n[Model {idx}/{len(all_models)}] {model_name}")
            print("=" * 60)
            print(f"  Model type: {model.model_type}")
            
            is_lm = self._is_lm_model(model)
            
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
            
            # Try to load existing model and its results
            if os.path.exists(model_path):
                print(f"  Found existing model at: {model_path}")
                try:
                    if is_lm:
                        # Load LM model
                        trained_model = self._load_lm_model(model, model_path)
                    else:
                        # Load ML model
                        trained_model = joblib.load(model_path)
                    
                    model_loaded = True
                    print("  ✓ Model loaded successfully")
                    
                    # Check if we have saved predictions
                    metrics_file = os.path.join(self.config['paths']['metrics_dir'], 'results_summary.json')
                    has_saved_predictions = False
                    
                    if os.path.exists(metrics_file):
                        with open(metrics_file, 'r') as f:
                            saved_results = json.load(f)
                        if model_name in saved_results and 'predictions' in saved_results[model_name]:
                            print("  ✓ Found saved predictions, skipping re-evaluation")
                            metrics = saved_results[model_name].get('metrics', {})
                            # Convert predictions back to dict format
                            predictions_dict = {}
                            for split_name, pred_data in saved_results[model_name]['predictions'].items():
                                predictions_dict[split_name] = {
                                    'predictions': np.array(pred_data['predictions']),
                                    'actuals': np.array(pred_data['actuals']),
                                    'probabilities': np.array(pred_data['probabilities']) if pred_data.get('probabilities') is not None else None
                                }
                            has_saved_predictions = True
                    
                    # Re-evaluate only if no saved predictions
                    if not has_saved_predictions:
                        print("  No saved predictions found, re-evaluating...")
                        if is_lm:
                            # Re-evaluate LM model
                            metrics, trained_model, predictions_dict = self._train_lm_model(
                                trained_model,
                                train_texts, y_train,
                                val_texts, y_val,
                                test_texts, y_test
                            )
                        else:
                            # Re-evaluate ML model
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
                    if is_lm:
                        # Train LM model with raw texts
                        metrics, trained_model, predictions_dict = self._train_lm_model(
                            model,
                            train_texts, y_train,
                            val_texts, y_val,
                            test_texts, y_test
                        )
                    else:
                        # Train ML model with TF-IDF features
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
                        if is_lm:
                            self._save_lm_model(trained_model, model_name, model_path)
                        else:
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
                            
                            if not is_lm:
                                # Log sklearn model artifact
                                try:
                                    mlflow.sklearn.log_model(
                                        trained_model,
                                        artifact_path="model",
                                        registered_model_name=f"{model_name}"
                                    )
                                except Exception:
                                    mlflow.sklearn.log_model(
                                        trained_model,
                                        artifact_path="model"
                                    )
                            
                            # Log figures
                            if 'test' in predictions_dict:
                                test_pred = predictions_dict['test']['predictions']
                                test_actual = predictions_dict['test']['actuals']
                                fig = self.evaluator.plot_confusion_matrix(
                                    test_actual, test_pred, model_name
                                )
                                cm_path = os.path.join(self.config['paths']['figures_dir'], 
                                                      f"{model_name}_confusion_matrix_temp.png")
                                fig.savefig(cm_path)
                                mlflow.log_artifact(cm_path, "figures")
                                os.remove(cm_path)
                                fig.clf()
                            
                            # Log ROC curve if probabilities available
                            if 'test' in predictions_dict and predictions_dict['test']['probabilities'] is not None:
                                test_proba = predictions_dict['test']['probabilities']
                                test_actual = predictions_dict['test']['actuals']
                                fig = self.evaluator.plot_roc_curve(
                                    test_actual, test_proba, model_name
                                )
                                roc_path = os.path.join(self.config['paths']['figures_dir'],
                                                       f"{model_name}_roc_curve_temp.png")
                                fig.savefig(roc_path)
                                mlflow.log_artifact(roc_path, "figures")
                                os.remove(roc_path)
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
                
                # Print CV results if available (only for ML models)
                if 'cv_f1_mean' in metrics:
                    print(f"\n  Cross-validation (F1): {metrics['cv_f1_mean']:.4f} "
                          f"(±{metrics['cv_f1_std']:.4f})")
        
        if not self.results:
            print("\nNo models were successfully trained!")
            return {}
        
        # [4/5] Compare models
        print("\n[4/5] Comparing models...")
        print("-" * 80)
        
        print("  Individual model results saved.")
        print("  Combined visualizations will be generated after all experiments complete.")
        
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
    
    def load_model(self, model_name: str) -> Any:
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
        
        # Check if it's an LM model by reading the marker file
        if os.path.exists(model_path):
            try:
                with open(model_path, 'r') as f:
                    content = f.read()
                    if content.startswith('{'):
                        marker = json.loads(content)
                        if marker.get('type') == 'lm':
                            # It's an LM model, need to create the right type
                            from models.LMs import BERTClassifier, DistilBERTClassifier, RoBERTaClassifier
                            model_classes = {
                                'BERT': BERTClassifier,
                                'DistilBERT': DistilBERTClassifier,
                                'RoBERTa': RoBERTaClassifier,
                            }
                            if model_name in model_classes:
                                model = model_classes[model_name]()
                                model.load(marker['path'])
                                return model
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # Not an LM model, load as joblib
        
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
        
        # Check if it's an LM model
        if hasattr(model, 'model_type') and model.model_type.startswith('lm'):
            # LM models use raw text directly
            return model.predict(texts)
        else:
            # ML models need preprocessing and vectorization
            processed_texts = self.data_loader.preprocess_texts(texts)
            X = self.data_loader.vectorize_texts(processed_texts, fit=False)
            return model.predict(X)
