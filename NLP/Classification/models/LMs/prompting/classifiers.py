"""
Concrete prompt-based classifier implementations.

Provides Zero-shot, Few-shot, and Chain-of-Thought classifiers
using LangChain templates and various inference backends.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import random

from models.LMs.prompting.base_prompt import BasePromptClassifier
from models.LMs.prompting.langchain_utils import (
    get_zero_shot_template,
    get_zero_shot_template_instruct,
    get_few_shot_template,
    get_few_shot_template_instruct,
    get_chain_of_thought_template,
    get_chain_of_thought_template_instruct,
    get_default_examples,
    select_examples,
    cot_parser
)


class ZeroShotPromptClassifier(BasePromptClassifier):
    """
    Zero-shot prompt classifier.
    
    Classifies sentiment using only the task description and input,
    without any examples. Best for quick experiments and when
    labeled data is unavailable.
    """
    
    def __init__(
        self,
        backend: str = "hf_api",
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_instruct_format: bool = True,
        max_new_tokens: int = 20,
        temperature: float = 0.1,
        show_progress: bool = True,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize zero-shot classifier.
        
        Args:
            backend: Inference backend ('hf_api' or 'local')
            model_name: Model identifier
            use_instruct_format: Use instruction-tuned model format
            max_new_tokens: Maximum tokens to generate (short for classification)
            temperature: Sampling temperature (low for determinism)
            show_progress: Show progress bar during inference
            random_state: Random seed
            **kwargs: Additional backend configuration
        """
        super().__init__(
            backend=backend,
            model_name=model_name,
            technique="zero_shot",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            show_progress=show_progress,
            random_state=random_state,
            **kwargs
        )
        
        self.use_instruct_format = use_instruct_format
        
        # Initialize prompt template
        if use_instruct_format:
            self._template = get_zero_shot_template_instruct()
        else:
            self._template = get_zero_shot_template()
    
    def _build_prompt(self, text: str) -> str:
        """Build zero-shot classification prompt."""
        return self._template.format(review=text)


class FewShotPromptClassifier(BasePromptClassifier):
    """
    Few-shot prompt classifier.
    
    Classifies sentiment using a few labeled examples before the query.
    Better accuracy than zero-shot, especially for domain-specific tasks.
    """
    
    def __init__(
        self,
        backend: str = "hf_api",
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_instruct_format: bool = True,
        num_examples: int = 3,
        use_training_examples: bool = True,
        max_new_tokens: int = 20,
        temperature: float = 0.1,
        show_progress: bool = True,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize few-shot classifier.
        
        Args:
            backend: Inference backend ('hf_api' or 'local')
            model_name: Model identifier
            use_instruct_format: Use instruction-tuned model format
            num_examples: Number of examples to include in prompt
            use_training_examples: Select examples from training data in fit()
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            show_progress: Show progress bar during inference
            random_state: Random seed
            **kwargs: Additional backend configuration
        """
        super().__init__(
            backend=backend,
            model_name=model_name,
            technique="few_shot",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            show_progress=show_progress,
            random_state=random_state,
            **kwargs
        )
        
        self.use_instruct_format = use_instruct_format
        self.num_examples = num_examples
        self.use_training_examples = use_training_examples
        
        # Initialize with default examples
        self._examples = get_default_examples()
        self._selected_examples = select_examples(
            self._examples, 
            num_examples=num_examples,
            balanced=True
        )
        self._template = None
        self._update_template()
    
    def _update_template(self):
        """Update the prompt template with current examples."""
        if self.use_instruct_format:
            self._template = get_few_shot_template_instruct(self._selected_examples)
        else:
            self._template = get_few_shot_template(self._selected_examples)
    
    def fit(self, X: Any, y: np.ndarray, **kwargs) -> None:
        """
        Select few-shot examples from training data.
        
        Args:
            X: Training texts
            y: Training labels (0=negative, 1=positive)
            **kwargs: Additional arguments
        """
        if not self.use_training_examples:
            return
        
        # Convert inputs to lists
        if hasattr(X, 'tolist'):
            texts = X.tolist()
        elif hasattr(X, 'values'):
            texts = list(X.values)
        else:
            texts = list(X)
        
        labels = np.array(y) if not isinstance(y, np.ndarray) else y
        
        # Build examples from training data
        self._examples = []
        for text, label in zip(texts, labels):
            sentiment = 'positive' if label == 1 else 'negative'
            self._examples.append({
                'review': text[:500],  # Truncate long reviews
                'sentiment': sentiment
            })
        
        # Select balanced examples
        random.seed(self.random_state)
        self._selected_examples = select_examples(
            self._examples,
            num_examples=self.num_examples,
            balanced=True
        )
        
        # Update template with new examples
        self._update_template()
        
        print(f"  Selected {len(self._selected_examples)} few-shot examples from training data")
    
    def _build_prompt(self, text: str) -> str:
        """Build few-shot classification prompt."""
        return self._template.format(review=text)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters including few-shot specific ones."""
        params = super().get_params()
        params.update({
            'num_examples': self.num_examples,
            'use_training_examples': self.use_training_examples,
            'use_instruct_format': self.use_instruct_format,
        })
        return params


class ChainOfThoughtClassifier(BasePromptClassifier):
    """
    Chain-of-thought prompt classifier.
    
    Guides the model through step-by-step reasoning before classification.
    Often produces more accurate results and provides interpretable reasoning.
    """
    
    def __init__(
        self,
        backend: str = "hf_api",
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_instruct_format: bool = True,
        max_new_tokens: int = 300,  # Longer for reasoning
        temperature: float = 0.3,  # Slightly higher for varied reasoning
        show_progress: bool = True,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize chain-of-thought classifier.
        
        Args:
            backend: Inference backend ('hf_api' or 'local')
            model_name: Model identifier
            use_instruct_format: Use instruction-tuned model format
            max_new_tokens: Maximum tokens (longer for reasoning chains)
            temperature: Sampling temperature (slightly higher for reasoning)
            show_progress: Show progress bar during inference
            random_state: Random seed
            **kwargs: Additional backend configuration
        """
        super().__init__(
            backend=backend,
            model_name=model_name,
            technique="chain_of_thought",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            show_progress=show_progress,
            random_state=random_state,
            **kwargs
        )
        
        self.use_instruct_format = use_instruct_format
        
        # Initialize prompt template
        if use_instruct_format:
            self._template = get_chain_of_thought_template_instruct()
        else:
            self._template = get_chain_of_thought_template()
        
        # Store reasoning for analysis
        self._last_reasoning: Optional[str] = None
    
    def _build_prompt(self, text: str) -> str:
        """Build chain-of-thought classification prompt."""
        return self._template.format(review=text)
    
    def _parse_response(self, response: str) -> str:
        """
        Parse CoT response to extract sentiment.
        
        Args:
            response: Model output with reasoning
            
        Returns:
            Sentiment label
        """
        result = cot_parser.parse(response)
        self._last_reasoning = result['reasoning']
        return result['sentiment']
    
    def _classify_single(self, text: str) -> Tuple[str, float]:
        """
        Classify a single text with chain-of-thought.
        
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
            return label, 1.0
        except Exception as e:
            print(f"  Warning: CoT classification failed: {e}")
            return 'negative', 0.5
    
    def get_last_reasoning(self) -> Optional[str]:
        """
        Get the reasoning from the last classification.
        
        Returns:
            Reasoning text or None if not available
        """
        return self._last_reasoning
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = super().get_params()
        params['use_instruct_format'] = self.use_instruct_format
        return params


# Convenience aliases
ZeroShot = ZeroShotPromptClassifier
FewShot = FewShotPromptClassifier
CoT = ChainOfThoughtClassifier



