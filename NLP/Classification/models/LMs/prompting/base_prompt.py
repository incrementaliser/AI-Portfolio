"""
Base class for prompt-based sentiment classifiers.

Provides the common interface and functionality for all prompting techniques.
"""
import numpy as np
from abc import abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import base class
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.base import BaseNLPModel

from models.LMs.prompting.inference import BaseInferenceBackend, get_backend
from models.LMs.prompting.langchain_utils import sentiment_parser


class BasePromptClassifier(BaseNLPModel):
    """
    Base class for prompt-based sentiment classifiers.
    
    Extends BaseNLPModel to provide prompting-specific functionality
    while maintaining compatibility with the existing pipeline.
    """
    
    # Label mapping
    LABEL_TO_INT = {'negative': 0, 'positive': 1}
    INT_TO_LABEL = {0: 'negative', 1: 'positive'}
    
    def __init__(
        self,
        backend: str = "hf_api",
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        technique: str = "zero_shot",
        max_new_tokens: int = 50,
        temperature: float = 0.1,
        batch_size: int = 1,
        show_progress: bool = True,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the prompt classifier.
        
        Args:
            backend: Inference backend ('hf_api' or 'local')
            model_name: Model identifier for the backend
            technique: Prompting technique name (for identification)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            batch_size: Batch size for processing (currently processes sequentially)
            show_progress: Whether to show progress bar during inference
            random_state: Random seed for reproducibility
            **kwargs: Additional backend configuration
        """
        super().__init__(model_type=f"prompting_{technique}")
        
        self.backend_type = backend
        self.model_name = model_name
        self.technique = technique
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.random_state = random_state
        self.backend_kwargs = kwargs
        
        # Initialize backend lazily
        self._backend: Optional[BaseInferenceBackend] = None
        
        # Storage for few-shot examples (used by FewShotPromptClassifier)
        self._examples: List[Dict[str, str]] = []
        
        np.random.seed(random_state)
    
    def _get_backend(self) -> BaseInferenceBackend:
        """Get or create the inference backend."""
        if self._backend is None:
            self._backend = get_backend(
                backend_type=self.backend_type,
                model_name=self.model_name,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                **self.backend_kwargs
            )
        return self._backend
    
    @abstractmethod
    def _build_prompt(self, text: str) -> str:
        """
        Build the prompt for a given input text.
        
        Args:
            text: Input review text
            
        Returns:
            Formatted prompt string
        """
        pass
    
    def _parse_response(self, response: str) -> str:
        """
        Parse the model response to extract sentiment label.
        
        Args:
            response: Raw model output
            
        Returns:
            Normalized sentiment label ('positive' or 'negative')
        """
        return sentiment_parser.parse(response)
    
    def _classify_single(self, text: str) -> Tuple[str, float]:
        """
        Classify a single text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (sentiment_label, confidence)
        """
        backend = self._get_backend()
        prompt = self._build_prompt(text)
        
        try:
            response = backend.generate(prompt)
            label = self._parse_response(response)
            # Confidence is 1.0 since we don't have actual probabilities
            return label, 1.0
        except Exception as e:
            print(f"  Warning: Classification failed for text: {e}")
            # Default to negative on failure
            return 'negative', 0.5
    
    def fit(self, X: Any, y: np.ndarray, **kwargs) -> None:
        """
        Fit the model (no-op for zero-shot, stores examples for few-shot).
        
        For prompt-based models, fit() doesn't train anything.
        Subclasses may override to select few-shot examples.
        
        Args:
            X: Training texts (list or array)
            y: Training labels
            **kwargs: Additional arguments
        """
        # Base implementation does nothing
        # Subclasses (FewShotPromptClassifier) will override to store examples
        pass
    
    def predict(self, X: Any) -> np.ndarray:
        """
        Predict sentiment labels for input texts.
        
        Args:
            X: Input texts (list, array, or sparse matrix)
            
        Returns:
            Array of integer labels (0=negative, 1=positive)
        """
        # Handle sparse matrices (shouldn't happen, but be safe)
        if hasattr(X, 'toarray'):
            raise ValueError(
                "Prompting models require raw text input, not sparse matrices."
            )
        
        # Convert to list
        if hasattr(X, 'tolist'):
            texts = X.tolist()
        elif hasattr(X, 'values'):
            texts = list(X.values)
        else:
            texts = list(X)
        
        predictions = []
        
        # Process with optional progress bar
        iterator = tqdm(texts, desc="  Classifying", disable=not self.show_progress)
        
        for text in iterator:
            label, _ = self._classify_single(text)
            predictions.append(self.LABEL_TO_INT[label])
        
        return np.array(predictions)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities.
        
        Note: Prompt-based models don't provide true probabilities.
        Returns pseudo-probabilities based on the predicted class.
        
        Args:
            X: Input texts
            
        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        # Handle sparse matrices
        if hasattr(X, 'toarray'):
            raise ValueError(
                "Prompting models require raw text input, not sparse matrices."
            )
        
        # Convert to list
        if hasattr(X, 'tolist'):
            texts = X.tolist()
        elif hasattr(X, 'values'):
            texts = list(X.values)
        else:
            texts = list(X)
        
        probabilities = []
        
        iterator = tqdm(texts, desc="  Classifying", disable=not self.show_progress)
        
        for text in iterator:
            label, confidence = self._classify_single(text)
            
            # Create pseudo-probabilities
            if label == 'positive':
                proba = [1 - confidence, confidence]
            else:
                proba = [confidence, 1 - confidence]
            
            probabilities.append(proba)
        
        return np.array(probabilities)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'backend': self.backend_type,
            'model_name': self.model_name,
            'technique': self.technique,
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'batch_size': self.batch_size,
            'random_state': self.random_state,
        }



