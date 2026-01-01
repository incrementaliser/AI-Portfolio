"""
DSpy-based prompt optimization for sentiment classification.

DSpy (Declarative Self-improving Python) treats prompts as learnable parameters
that can be systematically optimized. This is useful for:
- Automatic prompt engineering
- Multi-step reasoning chains
- Optimizing prompts on training data

Based on recommendations from suggestions.txt for Phase 2 (LLM zero/few-shot).

Installation: pip install dspy-ai
"""
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Check for DSpy availability
DSPY_AVAILABLE = False
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    pass

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.base import BaseNLPModel


class DSPySentimentSignature(dspy.Signature):
    """
    DSpy signature for sentiment classification.
    
    Defines the input-output specification for the task.
    """
    review = dspy.InputField(desc="A product review text")
    sentiment = dspy.OutputField(desc="The sentiment: 'positive' or 'negative'")


class DSPyCoTSentiment(dspy.Signature):
    """DSpy signature for chain-of-thought sentiment classification."""
    review = dspy.InputField(desc="A product review text to analyze")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning about the sentiment")
    sentiment = dspy.OutputField(desc="Final sentiment: 'positive' or 'negative'")


class DSPyClassifier(BaseNLPModel):
    """
    Sentiment classifier using DSpy for prompt optimization.
    
    DSpy allows:
    - Defining task signatures (input/output specs)
    - Automatic prompt optimization using training data
    - Multi-step reasoning with Chain of Thought
    - Systematic comparison of prompting strategies
    
    Example usage:
        classifier = DSPyClassifier(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            technique="cot",  # Chain of thought
            optimize=True     # Optimize prompts on training data
        )
        classifier.fit(train_texts, train_labels)
        predictions = classifier.predict(test_texts)
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        api_key: Optional[str] = None,
        technique: str = "predict",  # "predict", "cot", or "pot"
        optimize: bool = True,
        optimizer: str = "bootstrap",  # "bootstrap", "mipro", or "none"
        num_candidates: int = 10,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
        metric: str = "accuracy",
        temperature: float = 0.0,
        max_tokens: int = 100,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize DSpy classifier.
        
        Args:
            model_name: HuggingFace model identifier for inference
            api_key: HuggingFace API token (or use HF_TOKEN env var)
            technique: Prompting technique to use:
                - "predict": Simple prediction
                - "cot": Chain-of-thought reasoning
                - "pot": Program-of-thought (code generation)
            optimize: Whether to optimize prompts on training data
            optimizer: Optimization strategy:
                - "bootstrap": BootstrapFewShot (fast, recommended start)
                - "mipro": MIPRO optimizer (slower, better quality)
                - "none": No optimization
            num_candidates: Number of prompt candidates to try
            max_bootstrapped_demos: Max examples for bootstrapping
            max_labeled_demos: Max labeled examples in prompt
            metric: Optimization metric ("accuracy", "f1")
            temperature: Generation temperature
            max_tokens: Max tokens to generate
            random_state: Random seed
            **kwargs: Additional arguments
        """
        super().__init__(model_type="prompting_dspy")
        
        if not DSPY_AVAILABLE:
            raise ImportError(
                "DSpy is required for DSPyClassifier. "
                "Install with: pip install dspy-ai"
            )
        
        self.model_name = model_name
        self.api_key = api_key or os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
        self.technique = technique
        self.optimize = optimize
        self.optimizer_type = optimizer
        self.num_candidates = num_candidates
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.metric_name = metric
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.random_state = random_state
        
        # Set seeds
        np.random.seed(random_state)
        
        # DSpy components
        self._lm = None
        self._module = None
        self._compiled_module = None
        self._trainset = None
        self._is_trained = False
    
    def _setup_lm(self) -> None:
        """Initialize the DSpy language model."""
        print(f"  Setting up DSpy with model: {self.model_name}")
        
        # Configure DSpy to use HuggingFace models via API
        self._lm = dspy.HFClientTGI(
            model=self.model_name,
            port=None,  # Use default
            url="https://api-inference.huggingface.co/models/" + self.model_name,
            token=self.api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        # Set as default LM
        dspy.settings.configure(lm=self._lm)
        
        print(f"  DSpy configured with {self.model_name}")
    
    def _create_module(self) -> dspy.Module:
        """
        Create the DSpy module based on technique.
        
        Returns:
            DSpy module for sentiment classification
        """
        if self.technique == "predict":
            return dspy.Predict(DSPySentimentSignature)
        elif self.technique == "cot":
            return dspy.ChainOfThought(DSPySentimentSignature)
        elif self.technique == "pot":
            # Program of Thought - uses code for reasoning
            return dspy.ProgramOfThought(DSPySentimentSignature)
        else:
            raise ValueError(f"Unknown technique: {self.technique}")
    
    def _create_trainset(
        self,
        texts: List[str],
        labels: np.ndarray
    ) -> List[dspy.Example]:
        """
        Create DSpy training examples.
        
        Args:
            texts: Input texts
            labels: Binary labels
            
        Returns:
            List of DSpy Example objects
        """
        label_map = {0: "negative", 1: "positive"}
        
        examples = []
        for text, label in zip(texts, labels):
            example = dspy.Example(
                review=text[:1000],
                sentiment=label_map[int(label)]
            ).with_inputs("review")
            examples.append(example)
        
        return examples
    
    def _metric_fn(self, example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
        """
        Evaluation metric for optimization.
        
        Args:
            example: Ground truth example
            pred: Model prediction
            trace: Optional trace for debugging
            
        Returns:
            True if prediction is correct
        """
        predicted = pred.sentiment.lower().strip()
        actual = example.sentiment.lower().strip()
        
        # Handle variations
        if "positive" in predicted:
            predicted = "positive"
        elif "negative" in predicted:
            predicted = "negative"
        
        return predicted == actual
    
    def fit(
        self,
        X: Any,
        y: np.ndarray,
        X_val: Optional[Any] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        Fit the DSpy classifier, optionally optimizing prompts.
        
        Args:
            X: Training texts
            y: Training labels
            X_val: Validation texts (used for optimization if provided)
            y_val: Validation labels
        """
        # Setup LM
        if self._lm is None:
            self._setup_lm()
        
        # Create module
        self._module = self._create_module()
        
        # Process inputs
        texts = self._extract_texts(X)
        labels = np.array(y) if not isinstance(y, np.ndarray) else y
        
        # Create training set
        self._trainset = self._create_trainset(texts, labels)
        
        # Limit trainset size for efficiency
        if len(self._trainset) > self.max_labeled_demos:
            indices = np.random.choice(
                len(self._trainset),
                self.max_labeled_demos,
                replace=False
            )
            self._trainset = [self._trainset[i] for i in indices]
        
        # Create validation set if provided
        valset = None
        if X_val is not None and y_val is not None:
            val_texts = self._extract_texts(X_val)
            val_labels = np.array(y_val) if not isinstance(y_val, np.ndarray) else y_val
            valset = self._create_trainset(val_texts, val_labels)
            # Limit valset
            if len(valset) > 50:
                indices = np.random.choice(len(valset), 50, replace=False)
                valset = [valset[i] for i in indices]
        
        # Optimize if requested
        if self.optimize and self.optimizer_type != "none":
            print(f"\n  Optimizing prompts with {self.optimizer_type}...")
            
            if self.optimizer_type == "bootstrap":
                optimizer = dspy.BootstrapFewShot(
                    metric=self._metric_fn,
                    max_bootstrapped_demos=self.max_bootstrapped_demos,
                    max_labeled_demos=self.max_labeled_demos,
                )
            elif self.optimizer_type == "mipro":
                optimizer = dspy.MIPROv2(
                    metric=self._metric_fn,
                    num_candidates=self.num_candidates,
                    init_temperature=1.0
                )
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
            
            # Compile (optimize) the module
            eval_set = valset if valset else self._trainset
            self._compiled_module = optimizer.compile(
                self._module,
                trainset=self._trainset,
                valset=eval_set
            )
            
            print("  Prompt optimization completed!")
        else:
            self._compiled_module = self._module
            print("  Using unoptimized prompts (no training data used)")
        
        self._is_trained = True
    
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
        if self._compiled_module is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        texts = self._extract_texts(X)
        predictions = []
        
        print(f"  Predicting {len(texts)} samples with DSpy...")
        
        for text in texts:
            try:
                result = self._compiled_module(review=text[:1000])
                sentiment = result.sentiment.lower().strip()
                
                # Parse response
                if "positive" in sentiment:
                    predictions.append(1)
                elif "negative" in sentiment:
                    predictions.append(0)
                else:
                    # Default to negative if unclear
                    predictions.append(0)
            except Exception as e:
                print(f"    Warning: Prediction failed for text: {e}")
                predictions.append(0)
        
        return np.array(predictions)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities.
        
        Note: Returns pseudo-probabilities since DSpy doesn't provide true probs.
        
        Args:
            X: Input texts
            
        Returns:
            Array of shape (n_samples, 2)
        """
        predictions = self.predict(X)
        
        # Create pseudo-probabilities
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
            'technique': self.technique,
            'optimize': self.optimize,
            'optimizer': self.optimizer_type,
            'num_candidates': self.num_candidates,
            'max_bootstrapped_demos': self.max_bootstrapped_demos,
            'max_labeled_demos': self.max_labeled_demos,
            'temperature': self.temperature,
            'random_state': self.random_state,
        }
    
    def save(self, path: str) -> None:
        """
        Save the optimized DSpy module.
        
        Args:
            path: Directory to save module
        """
        os.makedirs(path, exist_ok=True)
        
        # Save compiled module
        if self._compiled_module is not None:
            module_path = os.path.join(path, 'dspy_module.json')
            self._compiled_module.save(module_path)
        
        # Save config
        import json
        config = self.get_params()
        config['_is_trained'] = self._is_trained
        with open(os.path.join(path, 'dspy_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"  DSpy module saved to: {path}")
    
    def load(self, path: str) -> None:
        """
        Load a saved DSpy module.
        
        Args:
            path: Directory containing saved module
        """
        import json
        
        # Load config
        config_path = os.path.join(path, 'dspy_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self._is_trained = config.get('_is_trained', True)
        
        # Setup LM if needed
        if self._lm is None:
            self._setup_lm()
        
        # Load module
        module_path = os.path.join(path, 'dspy_module.json')
        if os.path.exists(module_path):
            self._module = self._create_module()
            self._compiled_module = self._module
            self._compiled_module.load(module_path)
        
        print(f"  DSpy module loaded from: {path}")


class DSPyReActClassifier(BaseNLPModel):
    """
    DSpy classifier using ReAct (Reasoning + Acting) pattern.
    
    ReAct combines reasoning traces with actions, useful for
    more complex classification scenarios.
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        api_key: Optional[str] = None,
        max_iterations: int = 3,
        **kwargs
    ):
        """
        Initialize ReAct classifier.
        
        Args:
            model_name: Model identifier
            api_key: HuggingFace API token
            max_iterations: Max reasoning iterations
            **kwargs: Additional arguments
        """
        super().__init__(model_type="prompting_dspy_react")
        
        if not DSPY_AVAILABLE:
            raise ImportError("DSpy is required. Install with: pip install dspy-ai")
        
        self.model_name = model_name
        self.api_key = api_key or os.environ.get('HF_TOKEN')
        self.max_iterations = max_iterations
        self.kwargs = kwargs
        
        self._lm = None
        self._module = None
        self._is_trained = False
    
    def fit(self, X: Any, y: np.ndarray, **kwargs) -> None:
        """Initialize the ReAct module (no actual training)."""
        # Setup LM
        self._lm = dspy.HFClientTGI(
            model=self.model_name,
            url="https://api-inference.huggingface.co/models/" + self.model_name,
            token=self.api_key
        )
        dspy.settings.configure(lm=self._lm)
        
        # Create ReAct module
        self._module = dspy.ReAct(
            DSPySentimentSignature,
            max_iters=self.max_iterations
        )
        
        self._is_trained = True
    
    def predict(self, X: Any) -> np.ndarray:
        """Predict using ReAct reasoning."""
        if self._module is None:
            raise ValueError("Model not initialized. Call fit() first.")
        
        texts = X if isinstance(X, list) else list(X)
        predictions = []
        
        for text in texts:
            try:
                result = self._module(review=text[:1000])
                sentiment = result.sentiment.lower()
                predictions.append(1 if "positive" in sentiment else 0)
            except Exception:
                predictions.append(0)
        
        return np.array(predictions)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict probabilities."""
        preds = self.predict(X)
        probs = np.zeros((len(preds), 2))
        probs[preds == 0, 0] = 0.9
        probs[preds == 0, 1] = 0.1
        probs[preds == 1, 0] = 0.1
        probs[preds == 1, 1] = 0.9
        return probs
    
    def get_params(self) -> Dict[str, Any]:
        """Get parameters."""
        return {
            'model_name': self.model_name,
            'max_iterations': self.max_iterations,
        }


# Convenience aliases
DSPyPredict = DSPyClassifier  # With technique="predict"
DSPyCoT = lambda **kwargs: DSPyClassifier(technique="cot", **kwargs)
DSPyOptimized = lambda **kwargs: DSPyClassifier(optimize=True, **kwargs)


