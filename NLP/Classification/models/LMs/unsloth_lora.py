"""
Unsloth-based LoRA fine-tuning for LLMs.

Uses Unsloth for 2-5x faster LoRA/QLoRA fine-tuning with reduced memory usage.
Recommended for Phase 3 (LLM fine-tuning) as per suggestions.txt.

Unsloth optimizes:
- Memory usage (up to 60% reduction)
- Training speed (2-5x faster)
- LoRA/QLoRA implementations

Note: Unsloth requires specific installation and GPU support.
Install with: pip install unsloth
"""
import os
import torch
import numpy as np
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Check for Unsloth availability
UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    pass

# TRL for training
TRL_AVAILABLE = False
try:
    from trl import SFTTrainer
    from transformers import TrainingArguments
    TRL_AVAILABLE = True
except ImportError:
    pass

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base import BaseNLPModel


class UnslothLoRAClassifier(BaseNLPModel):
    """
    LLM classifier using Unsloth for efficient LoRA fine-tuning.
    
    Features:
    - 2-5x faster training than standard HuggingFace
    - Up to 60% less memory usage
    - Supports 4-bit quantization (QLoRA)
    - Optimized LoRA implementation
    
    Example usage:
        classifier = UnslothLoRAClassifier(
            model_name="unsloth/Llama-3.2-1B-Instruct",
            lora_r=16,
            load_in_4bit=True
        )
        classifier.fit(train_texts, train_labels)
        predictions = classifier.predict(test_texts)
    
    Recommended models (small, fast):
    - unsloth/Llama-3.2-1B-Instruct (~1B params)
    - unsloth/Phi-3.5-mini-instruct (~3.8B params)
    - unsloth/Mistral-7B-Instruct-v0.3 (~7B params)
    """
    
    # Supported models optimized by Unsloth
    SUPPORTED_MODELS = [
        "unsloth/Llama-3.2-1B-Instruct",
        "unsloth/Llama-3.2-3B-Instruct",
        "unsloth/Phi-3.5-mini-instruct",
        "unsloth/Mistral-7B-Instruct-v0.3",
        "unsloth/gemma-2-2b-it",
        "unsloth/Qwen2.5-1.5B-Instruct",
    ]
    
    def __init__(
        self,
        model_name: str = "unsloth/Llama-3.2-1B-Instruct",
        max_length: int = 512,
        load_in_4bit: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: List[str] = None,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        epochs: int = 1,
        warmup_ratio: float = 0.03,
        max_steps: int = -1,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize Unsloth LoRA classifier.
        
        Args:
            model_name: Unsloth-optimized model identifier
            max_length: Maximum sequence length
            load_in_4bit: Use 4-bit quantization (QLoRA)
            lora_r: LoRA rank (higher = more capacity, more memory)
            lora_alpha: LoRA alpha scaling parameter
            lora_dropout: Dropout for LoRA layers
            target_modules: Modules to apply LoRA to (None = auto-detect)
            batch_size: Per-device batch size
            gradient_accumulation_steps: Steps to accumulate gradients
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            warmup_ratio: Warmup ratio for scheduler
            max_steps: Max training steps (-1 for epochs-based)
            random_state: Random seed
            **kwargs: Additional arguments
        """
        super().__init__(model_type="lm_unsloth_lora")
        
        if not UNSLOTH_AVAILABLE:
            raise ImportError(
                "Unsloth is required for UnslothLoRAClassifier. "
                "Install with: pip install unsloth\n"
                "See https://github.com/unslothai/unsloth for installation instructions."
            )
        
        if not TRL_AVAILABLE:
            raise ImportError(
                "TRL (Transformer Reinforcement Learning) is required. "
                "Install with: pip install trl"
            )
        
        self.model_name = model_name
        self.max_length = max_length
        self.load_in_4bit = load_in_4bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.max_steps = max_steps
        self.random_state = random_state
        
        # Set seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Model components
        self.model = None
        self.tokenizer = None
        self._is_trained = False
        
        # Classification prompt template
        self.prompt_template = """Below is a product review. Classify its sentiment as 'positive' or 'negative'.

### Review:
{text}

### Sentiment:
{label}"""
    
    def _load_model(self) -> None:
        """Load model with Unsloth optimizations and LoRA."""
        print(f"  Loading Unsloth-optimized model: {self.model_name}")
        print(f"  4-bit quantization: {self.load_in_4bit}")
        print(f"  LoRA rank: {self.lora_r}")
        
        # Load with Unsloth's FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_length,
            dtype=None,  # Auto-detect best dtype
            load_in_4bit=self.load_in_4bit
        )
        
        # Add LoRA adapters
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_r,
            target_modules=self.target_modules,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth optimization
            random_state=self.random_state,
        )
        
        print(f"  Model loaded with LoRA adapters")
        print(f"  Trainable parameters: {self._count_trainable_params():,}")
    
    def _count_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _format_training_data(
        self,
        texts: List[str],
        labels: np.ndarray
    ) -> List[Dict[str, str]]:
        """
        Format data for instruction fine-tuning.
        
        Args:
            texts: Input texts
            labels: Binary labels (0=negative, 1=positive)
            
        Returns:
            List of formatted training examples
        """
        label_map = {0: "negative", 1: "positive"}
        
        formatted = []
        for text, label in zip(texts, labels):
            formatted.append({
                "text": self.prompt_template.format(
                    text=text[:1000],  # Truncate very long texts
                    label=label_map[int(label)]
                )
            })
        
        return formatted
    
    def _format_inference_prompt(self, text: str) -> str:
        """Format a single text for inference (without label)."""
        return self.prompt_template.split("{label}")[0].format(text=text[:1000])
    
    def fit(
        self,
        X: Any,
        y: np.ndarray,
        X_val: Optional[Any] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        Fine-tune the model using Unsloth's optimized LoRA training.
        
        Args:
            X: Training texts
            y: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
        """
        if self.model is None:
            self._load_model()
        
        # Process inputs
        texts = self._extract_texts(X)
        labels = np.array(y) if not isinstance(y, np.ndarray) else y
        
        # Format training data
        train_data = self._format_training_data(texts, labels)
        
        # Create dataset
        from datasets import Dataset
        train_dataset = Dataset.from_list(train_data)
        
        # Validation dataset
        eval_dataset = None
        if X_val is not None and y_val is not None:
            val_texts = self._extract_texts(X_val)
            val_labels = np.array(y_val) if not isinstance(y_val, np.ndarray) else y_val
            val_data = self._format_training_data(val_texts, val_labels)
            eval_dataset = Dataset.from_list(val_data)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./unsloth_output",
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_ratio=self.warmup_ratio,
            num_train_epochs=self.epochs,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",  # Memory-efficient optimizer
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=self.random_state,
            report_to="none",  # Disable wandb/tensorboard
            save_strategy="no",  # Don't save checkpoints
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_length,
            args=training_args,
        )
        
        print(f"\n  Training with Unsloth (2-5x faster than standard)...")
        print(f"  Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        
        # Train
        trainer.train()
        
        self._is_trained = True
        print("  Training completed!")
    
    def _extract_texts(self, X: Any) -> List[str]:
        """Extract text list from various input formats."""
        if hasattr(X, 'tolist'):
            return X.tolist()
        elif hasattr(X, 'values'):
            return X.values.tolist() if hasattr(X.values, 'tolist') else list(X.values)
        return list(X)
    
    def predict(self, X: Any) -> np.ndarray:
        """
        Predict sentiment labels.
        
        Args:
            X: Input texts
            
        Returns:
            Array of predicted labels (0=negative, 1=positive)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        texts = self._extract_texts(X)
        predictions = []
        
        # Enable inference mode for faster generation
        FastLanguageModel.for_inference(self.model)
        
        print(f"  Generating predictions for {len(texts)} samples...")
        
        for text in texts:
            prompt = self._format_inference_prompt(text)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip().lower()
            
            # Parse response
            if 'positive' in response:
                predictions.append(1)
            else:
                predictions.append(0)
        
        return np.array(predictions)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities.
        
        Note: Returns pseudo-probabilities since this is a generative model.
        
        Args:
            X: Input texts
            
        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        predictions = self.predict(X)
        
        # Create pseudo-probabilities (high confidence for predicted class)
        probabilities = np.zeros((len(predictions), 2))
        for i, pred in enumerate(predictions):
            if pred == 1:
                probabilities[i] = [0.1, 0.9]
            else:
                probabilities[i] = [0.9, 0.1]
        
        return probabilities
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'load_in_4bit': self.load_in_4bit,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'random_state': self.random_state,
        }
    
    def save(self, path: str) -> None:
        """
        Save the LoRA adapters.
        
        Args:
            path: Directory to save adapters
        """
        os.makedirs(path, exist_ok=True)
        
        # Save LoRA adapters (much smaller than full model)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save config
        import json
        config = self.get_params()
        config['_is_trained'] = self._is_trained
        with open(os.path.join(path, 'lora_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"  LoRA adapters saved to: {path}")
    
    def load(self, path: str) -> None:
        """
        Load saved LoRA adapters.
        
        Args:
            path: Directory containing saved adapters
        """
        import json
        
        # Load config
        config_path = os.path.join(path, 'lora_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.model_name = config.get('model_name', self.model_name)
            self._is_trained = config.get('_is_trained', True)
        
        # Load base model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=path,  # Load from saved path
            max_seq_length=self.max_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit
        )
        
        print(f"  LoRA model loaded from: {path}")


# Convenience aliases for different base models
class LlamaLoRA(UnslothLoRAClassifier):
    """Llama 3.2 1B with LoRA fine-tuning using Unsloth."""
    
    def __init__(self, **kwargs):
        """Initialize Llama with Unsloth LoRA."""
        kwargs.setdefault('model_name', 'unsloth/Llama-3.2-1B-Instruct')
        super().__init__(**kwargs)


class PhiLoRA(UnslothLoRAClassifier):
    """Phi-3.5 Mini with LoRA fine-tuning using Unsloth."""
    
    def __init__(self, **kwargs):
        """Initialize Phi with Unsloth LoRA."""
        kwargs.setdefault('model_name', 'unsloth/Phi-3.5-mini-instruct')
        super().__init__(**kwargs)


class MistralLoRA(UnslothLoRAClassifier):
    """Mistral 7B with LoRA fine-tuning using Unsloth."""
    
    def __init__(self, **kwargs):
        """Initialize Mistral with Unsloth LoRA."""
        kwargs.setdefault('model_name', 'unsloth/Mistral-7B-Instruct-v0.3')
        super().__init__(**kwargs)


