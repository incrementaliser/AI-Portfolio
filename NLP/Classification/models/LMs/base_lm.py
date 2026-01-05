"""
Base class for Language Model classifiers.

Provides common functionality for transformer-based models including
device management, training loops, and the standard fit/predict interface.
"""
import os
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import base class from parent
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base import BaseNLPModel


class BaseLMModel(BaseNLPModel):
    """
    Base class for transformer-based language model classifiers.
    
    Extends BaseNLPModel to provide common functionality for BERT-like models
    including tokenization, GPU management, and training loops.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        max_length: int = 256,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        fp16: bool = True,
        early_stopping_patience: int = 2,
        random_state: int = 42,
        fine_tune: bool = False,
        **kwargs
    ):
        """
        Initialize the base language model classifier.
        
        Args:
            model_name: Hugging Face model identifier (e.g., 'bert-base-uncased')
            num_labels: Number of classification labels
            max_length: Maximum sequence length for tokenization
            batch_size: Training and inference batch size
            learning_rate: Learning rate for AdamW optimizer
            epochs: Number of training epochs (only used if fine_tune=True)
            warmup_ratio: Fraction of training steps for learning rate warmup
            weight_decay: Weight decay for regularization
            fp16: Whether to use mixed precision training (requires GPU)
            early_stopping_patience: Number of epochs to wait before early stopping
            random_state: Random seed for reproducibility
            fine_tune: Whether to fine-tune the model on training data (default: False)
                       If False, uses pre-trained model without training
            **kwargs: Additional keyword arguments
        """
        super().__init__(model_type="lm")
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size  # Training batch size
        # Inference batch size (can be larger than training batch size)
        self.inference_batch_size = kwargs.get('inference_batch_size', batch_size * 4)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.fp16 = fp16
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.fine_tune = fine_tune
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            self.fp16 = False  # FP16 requires GPU
        
        # Model components (initialized in _load_model)
        self.tokenizer = None
        self.model = None
        self._is_trained = False
        
    def _load_model(self) -> None:
        """Load the pretrained model and tokenizer from Hugging Face."""
        print(f"  Loading model: {self.model_name}")
        print(f"  Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        self.model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency (only during training)
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Optimize model for inference using torch.compile (PyTorch 2.0+)
        # This can provide 20-30% speedup on compatible hardware
        try:
            if hasattr(torch, 'compile') and self.device.type == "cuda":
                print("  Compiling model for faster inference...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
        except Exception as e:
            # torch.compile not available or failed, continue without it
            pass
    
    def _tokenize_texts(
        self,
        texts: List[str],
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a list of texts using the model's tokenizer.
        
        Args:
            texts: List of input texts
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            
        Returns:
            Dictionary containing input_ids and attention_mask tensors
        """
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def _create_dataloader(
        self,
        texts: List[str],
        labels: Optional[np.ndarray] = None,
        shuffle: bool = False,
        use_inference_batch_size: bool = False
    ) -> DataLoader:
        """
        Create a DataLoader from texts and optional labels.
        
        Args:
            texts: List of input texts
            labels: Optional array of labels
            shuffle: Whether to shuffle the data
            use_inference_batch_size: If True, use inference batch size (larger, faster)
            
        Returns:
            PyTorch DataLoader
        """
        from models.LMs.lm_data import TextClassificationDataset
        
        dataset = TextClassificationDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        batch_size = self.inference_batch_size if use_inference_batch_size else self.batch_size
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Avoid multiprocessing issues on Windows
            pin_memory=True if self.device.type == "cuda" else False
        )
    
    def fit(
        self,
        X: Any,
        y: np.ndarray,
        X_val: Optional[Any] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        Load and optionally fine-tune the model on the given data.
        
        Args:
            X: Training texts (list of strings or DataFrame with 'review' column)
            y: Training labels
            X_val: Optional validation texts
            y_val: Optional validation labels
            
        Note:
            If fine_tune=False (default), only loads the pre-trained model without training.
            The classification head will be randomly initialized and predictions may be poor.
            Set fine_tune=True to enable fine-tuning on your dataset.
        """
        # Load model if not already loaded
        if self.model is None:
            self._load_model()
        
        # Skip training if fine_tune is False
        if not self.fine_tune:
            print("  Using pre-trained model without fine-tuning (fine_tune=False)")
            print("  Note: Classification head is randomly initialized. Enable fine_tune=True for better performance.")
            self._is_trained = True
            return
        
        
        # Fine-tuning mode: train the model
        print("  Fine-tuning model on training data...")
        
        # Extract texts if X is a DataFrame
        if hasattr(X, 'tolist'):
            texts = X.tolist()
        elif hasattr(X, 'values'):
            texts = X.values.tolist() if hasattr(X.values, 'tolist') else list(X.values)
        else:
            texts = list(X)
        
        # Convert labels to numpy array
        labels = np.array(y) if not isinstance(y, np.ndarray) else y
        
        # Create training dataloader
        train_loader = self._create_dataloader(texts, labels, shuffle=True)
        
        # Create validation dataloader if provided
        val_loader = None
        if X_val is not None and y_val is not None:
            if hasattr(X_val, 'tolist'):
                val_texts = X_val.tolist()
            elif hasattr(X_val, 'values'):
                val_texts = X_val.values.tolist() if hasattr(X_val.values, 'tolist') else list(X_val.values)
            else:
                val_texts = list(X_val)
            val_labels = np.array(y_val) if not isinstance(y_val, np.ndarray) else y_val
            val_loader = self._create_dataloader(val_texts, val_labels, shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if self.fp16 else None
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n  Fine-tuning for {self.epochs} epochs...")
        print(f"  Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(
                train_loader,
                desc=f"  Epoch {epoch + 1}/{self.epochs}",
                leave=True
            )
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                total_loss += loss.item()
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                print(f"  Epoch {epoch + 1} - Validation loss: {val_loss:.4f}")
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self._best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(f"  Early stopping triggered after {epoch + 1} epochs")
                        # Restore best model
                        if hasattr(self, '_best_model_state'):
                            self.model.load_state_dict(self._best_model_state)
                            self.model.to(self.device)
                        break
        
        self._is_trained = True
        print("  Fine-tuning completed!")
    
    def _validate(self, val_loader: DataLoader) -> float:
        """
        Calculate validation loss.
        
        Args:
            val_loader: Validation DataLoader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()
        
        return total_loss / len(val_loader)
    
    def predict(self, X: Any) -> np.ndarray:
        """
        Make predictions on the given data.
        
        Args:
            X: Input texts (list of strings, array, or sparse matrix)
            
        Returns:
            Array of predicted labels
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call fit() first.")
        
        # Handle sparse matrices (from TF-IDF) - this shouldn't happen for LM models
        # but we handle it gracefully
        if hasattr(X, 'toarray'):
            raise ValueError(
                "LM models require raw text input, not sparse matrices. "
                "Ensure the pipeline passes raw texts to LM models."
            )
        
        # Extract texts
        if hasattr(X, 'tolist'):
            texts = X.tolist()
        elif hasattr(X, 'values'):
            texts = X.values.tolist() if hasattr(X.values, 'tolist') else list(X.values)
        else:
            texts = list(X)
        
        # Create dataloader for inference (use larger batch size for faster inference)
        dataloader = self._create_dataloader(texts, labels=None, shuffle=False, use_inference_batch_size=True)
        
        self.model.eval()
        all_predictions = []
        
        total_samples = len(texts)
        total_batches = len(dataloader)
        
        with torch.no_grad():
            progress_bar = tqdm(
                dataloader,
                desc="  Predicting",
                total=total_batches,
                unit="batch",
                leave=True
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                
                # Update progress bar with sample count
                samples_processed = min((batch_idx + 1) * self.inference_batch_size, total_samples)
                progress_bar.set_postfix({
                    'samples': f'{samples_processed}/{total_samples}'
                })
        
        return np.array(all_predictions)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities for the given data.
        
        Args:
            X: Input texts (list of strings or array)
            
        Returns:
            Array of class probabilities with shape (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call fit() first.")
        
        # Handle sparse matrices
        if hasattr(X, 'toarray'):
            raise ValueError(
                "LM models require raw text input, not sparse matrices. "
                "Ensure the pipeline passes raw texts to LM models."
            )
        
        # Extract texts
        if hasattr(X, 'tolist'):
            texts = X.tolist()
        elif hasattr(X, 'values'):
            texts = X.values.tolist() if hasattr(X.values, 'tolist') else list(X.values)
        else:
            texts = list(X)
        
        # Create dataloader for inference (use larger batch size for faster inference)
        dataloader = self._create_dataloader(texts, labels=None, shuffle=False, use_inference_batch_size=True)
        
        self.model.eval()
        all_probabilities = []
        
        total_samples = len(texts)
        total_batches = len(dataloader)
        
        with torch.no_grad():
            progress_bar = tqdm(
                dataloader,
                desc="  Predicting probabilities",
                total=total_batches,
                unit="batch",
                leave=True
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update progress bar with sample count
                samples_processed = min((batch_idx + 1) * self.inference_batch_size, total_samples)
                progress_bar.set_postfix({
                    'samples': f'{samples_processed}/{total_samples}'
                })
        
        return np.array(all_probabilities)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'warmup_ratio': self.warmup_ratio,
            'weight_decay': self.weight_decay,
            'fp16': self.fp16,
            'early_stopping_patience': self.early_stopping_patience,
            'random_state': self.random_state,
            'fine_tune': self.fine_tune,
            'inference_batch_size': self.inference_batch_size,
            'device': str(self.device)
        }
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Directory path to save the model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save additional config
        import json
        config = self.get_params()
        config['_is_trained'] = self._is_trained
        with open(os.path.join(path, 'training_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"  Model saved to: {path}")
    
    def load(self, path: str) -> None:
        """
        Load a model from disk.
        
        Args:
            path: Directory path containing the saved model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        
        # Load additional config
        import json
        config_path = os.path.join(path, 'training_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self._is_trained = config.get('_is_trained', True)
        else:
            self._is_trained = True
        
        print(f"  Model loaded from: {path}")



