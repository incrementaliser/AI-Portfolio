"""
NLP classification model evaluation module
"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 47
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class NLPEvaluator:
    """Evaluator for NLP classification models."""
    
    def __init__(self, config: Dict):
        """
        Initialize NLP evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score
        """
        return accuracy_score(y_true, y_pred)
    
    def calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
        """
        Calculate precision score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy
            
        Returns:
            Precision score
        """
        return precision_score(y_true, y_pred, average=average, zero_division=0)
    
    def calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
        """
        Calculate recall score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy
            
        Returns:
            Recall score
        """
        return recall_score(y_true, y_pred, average=average, zero_division=0)
    
    def calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
        """
        Calculate F1 score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy
            
        Returns:
            F1 score
        """
        return f1_score(y_true, y_pred, average=average, zero_division=0)
    
    def calculate_roc_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Calculate ROC AUC score.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            ROC AUC score
        """
        try:
            return roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            return 0.0
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
        prefix: str = 'test'
    ) -> Dict[str, float]:
        """
        Calculate all classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            f'{prefix}_accuracy': self.calculate_accuracy(y_true, y_pred),
            f'{prefix}_precision': self.calculate_precision(y_true, y_pred),
            f'{prefix}_recall': self.calculate_recall(y_true, y_pred),
            f'{prefix}_f1': self.calculate_f1(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics[f'{prefix}_roc_auc'] = self.calculate_roc_auc(y_true, y_pred_proba)
        
        return metrics
    
    def cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv_splits: int = 5,
        scoring: str = 'f1'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            model: Model instance
            X: Features
            y: Labels
            cv_splits: Number of CV splits
            scoring: Scoring metric
            
        Returns:
            Dictionary with mean and std of scores
        """
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            f'cv_{scoring}_mean': np.mean(scores),
            f'cv_{scoring}_std': np.std(scores)
        }
    
    def evaluate_model(
        self,
        model: Any,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Dict, Any, Dict]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Model instance
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            y_train: Training labels
            y_val: Validation labels
            y_test: Test labels
            
        Returns:
            Tuple of (metrics_dict, fitted_model, predictions_dict)
        """
        results = {}
        predictions_dict = {}
        
        # Train model
        print(f"  Training model on {len(y_train)} samples...")
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"  Error fitting model: {e}")
            import traceback
            traceback.print_exc()
            return {}, model, {}
        
        # Evaluate on validation set
        print("  Evaluating on validation set...")
        try:
            val_pred = model.predict(X_val)
            val_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
            
            val_metrics = self.calculate_all_metrics(
                y_val, val_pred, val_pred_proba, prefix='val'
            )
            results.update(val_metrics)
            
            predictions_dict['val'] = {
                'predictions': val_pred,
                'probabilities': val_pred_proba,
                'actuals': y_val
            }
        except Exception as e:
            print(f"  Warning: Validation evaluation failed: {e}")
        
        # Evaluate on test set
        print("  Evaluating on test set...")
        try:
            test_pred = model.predict(X_test)
            test_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            test_metrics = self.calculate_all_metrics(
                y_test, test_pred, test_pred_proba, prefix='test'
            )
            results.update(test_metrics)
            
            predictions_dict['test'] = {
                'predictions': test_pred,
                'probabilities': test_pred_proba,
                'actuals': y_test
            }
        except Exception as e:
            print(f"  Warning: Test evaluation failed: {e}")
        
        # Cross-validation
        print("  Performing cross-validation...")
        cv_splits = self.config.get('evaluation', {}).get('cv_splits', 5)
        try:
            cv_results = self.cross_validate(model, X_train, y_train, cv_splits=cv_splits)
            results.update(cv_results)
        except Exception as e:
            print(f"  Warning: Cross-validation failed: {e}")
        
        return results, model, predictions_dict
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'],
               title=f'{model_name} - Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(
                os.path.join(save_path, f'{model_name}_confusion_matrix.png'),
                dpi=150,
                bbox_inches='tight'
            )
        
        return fig
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{model_name} - ROC Curve', fontsize=14)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(
                os.path.join(save_path, f'{model_name}_roc_curve.png'),
                dpi=150,
                bbox_inches='tight'
            )
        
        return fig
    
    def plot_all_confusion_matrices(
        self,
        results_dict: Dict[str, Dict],
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot confusion matrices for all models in a single figure with subplots.
        
        Args:
            results_dict: Dictionary of model results with predictions
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure with all confusion matrices
        """
        model_names = []
        confusion_matrices = []
        
        # Collect confusion matrices for all models
        for model_name, result_data in results_dict.items():
            if 'test' in result_data.get('predictions', {}):
                pred_data = result_data['predictions']['test']
                y_true = pred_data['actuals']
                y_pred = pred_data['predictions']
                
                cm = confusion_matrix(y_true, y_pred)
                confusion_matrices.append(cm)
                model_names.append(model_name)
        
        if not confusion_matrices:
            # Return empty figure if no data
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No confusion matrices available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create subplots - arrange in a grid with max 4 columns
        n_models = len(confusion_matrices)
        n_cols = min(4, n_models)  # Max 4 columns (rows of 4)
        n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        # Plot each confusion matrix
        for idx, (model_name, cm) in enumerate(zip(model_names, confusion_matrices)):
            ax = axes[idx]
            
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=12, fontweight='bold')
            
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   title=f'{model_name}',
                   ylabel='True Label',
                   xlabel='Predicted Label')
            ax.tick_params(labelsize=10)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(
                os.path.join(save_path, 'all_confusion_matrices.png'),
                dpi=150,
                bbox_inches='tight'
            )
        
        return fig
    
    def plot_all_roc_curves(
        self,
        results_dict: Dict[str, Dict],
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot ROC curves for all models in a single figure.
        
        Args:
            results_dict: Dictionary of model results with predictions
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure with all ROC curves
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color palette for different models
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
        
        # Plot ROC curve for each model
        for idx, (model_name, result_data) in enumerate(results_dict.items()):
            if 'test' in result_data.get('predictions', {}):
                pred_data = result_data['predictions']['test']
                y_true = pred_data['actuals']
                y_pred_proba = pred_data.get('probabilities')
                
                if y_pred_proba is not None:
                    try:
                        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                        
                        ax.plot(fpr, tpr, 
                               color=colors[idx], 
                               lw=2, 
                               label=f'{model_name} (AUC = {roc_auc:.3f})',
                               alpha=0.8)
                    except Exception as e:
                        print(f"  Warning: Could not plot ROC curve for {model_name}: {e}")
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
               label='Random Classifier', alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(
                os.path.join(save_path, 'all_roc_curves.png'),
                dpi=150,
                bbox_inches='tight'
            )
        
        return fig
    
    def compare_models(
        self,
        results_dict: Dict[str, Dict],
        save_path: str = None
    ) -> plt.Figure:
        """
        Create comparison bar plots for multiple models.
        
        Args:
            results_dict: Dictionary of model results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        model_names = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            values = []
            labels = []
            for model_name in model_names:
                key = f'test_{metric}'
                if key in results_dict[model_name]['metrics']:
                    values.append(results_dict[model_name]['metrics'][key])
                    labels.append(model_name)
            
            if values:
                bars = ax.bar(labels, values, alpha=0.7, edgecolor='black')
                ax.set_ylabel(metric.upper(), fontsize=12)
                ax.set_title(f'{metric.upper()}', fontsize=14)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(
                os.path.join(save_path, 'model_comparison.png'),
                dpi=150,
                bbox_inches='tight'
            )
        
        return fig

