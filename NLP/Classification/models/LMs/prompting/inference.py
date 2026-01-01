"""
Inference backends for LLM prompting.

Provides HuggingFace API and local inference backends with a common interface.
"""
import os
import time
from abc import ABC, abstractmethod
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class BaseInferenceBackend(ABC):
    """Abstract base class for inference backends."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion for a prompt.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the backend is available and properly configured.
        
        Returns:
            True if backend can be used
        """
        pass


class HFAPIBackend(BaseInferenceBackend):
    """
    HuggingFace Inference API backend.
    
    Uses the free tier of HuggingFace's serverless inference API.
    Requires a HuggingFace API token.
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        token: Optional[str] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.1,
        rate_limit_delay: float = 1.0,
        timeout: int = 30,
        retry_attempts: int = 3
    ):
        """
        Initialize HuggingFace API backend.
        
        Args:
            model_name: HuggingFace model identifier
            token: HuggingFace API token (or set HF_TOKEN env var)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            rate_limit_delay: Seconds to wait between API calls
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts on failure
        """
        self.model_name = model_name
        self.token = token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        
        self._client = None
        self._last_call_time = 0
    
    def _get_client(self):
        """Lazily initialize the HuggingFace InferenceClient."""
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(token=self.token)
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required for HF API backend. "
                    "Install with: pip install huggingface_hub"
                )
        return self._client
    
    def is_available(self) -> bool:
        """Check if HF API is available."""
        try:
            from huggingface_hub import InferenceClient
            # Token is optional for some models but recommended
            return True
        except ImportError:
            return False
    
    def _rate_limit(self):
        """Apply rate limiting between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_call_time = time.time()
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using HuggingFace Inference API.
        
        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters
            
        Returns:
            Generated text
        """
        client = self._get_client()
        
        # Apply rate limiting
        self._rate_limit()
        
        # Merge default params with overrides
        params = {
            'max_new_tokens': kwargs.get('max_new_tokens', self.max_new_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'do_sample': kwargs.get('temperature', self.temperature) > 0,
        }
        
        # Retry logic
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                result = client.text_generation(
                    prompt,
                    model=self.model_name,
                    **params
                )
                return result
            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    # Wait before retry with exponential backoff
                    wait_time = (2 ** attempt) * self.rate_limit_delay
                    print(f"  API call failed, retrying in {wait_time:.1f}s... ({e})")
                    time.sleep(wait_time)
        
        raise RuntimeError(f"HF API failed after {self.retry_attempts} attempts: {last_error}")


class LocalHFBackend(BaseInferenceBackend):
    """
    Local HuggingFace Transformers backend.
    
    Runs models locally using the transformers library.
    Requires GPU for reasonable performance with larger models.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: str = "auto",
        torch_dtype: str = "float16",
        max_new_tokens: int = 100,
        temperature: float = 0.1,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize local HuggingFace backend.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('auto', 'cuda', 'cpu')
            torch_dtype: Data type ('float16', 'bfloat16', 'float32')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        self._pipeline = None
    
    def _get_pipeline(self):
        """Lazily initialize the text generation pipeline."""
        if self._pipeline is None:
            try:
                import torch
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                
                print(f"  Loading local model: {self.model_name}")
                
                # Determine dtype
                dtype_map = {
                    'float16': torch.float16,
                    'bfloat16': torch.bfloat16,
                    'float32': torch.float32,
                }
                dtype = dtype_map.get(self.torch_dtype, torch.float16)
                
                # Determine device
                if self.device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    device = self.device
                
                # Load with quantization if requested
                model_kwargs = {'torch_dtype': dtype}
                if self.load_in_8bit:
                    model_kwargs['load_in_8bit'] = True
                    model_kwargs.pop('torch_dtype', None)
                elif self.load_in_4bit:
                    model_kwargs['load_in_4bit'] = True
                    model_kwargs.pop('torch_dtype', None)
                
                # Create pipeline
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device_map=device if device != "cpu" else None,
                    model_kwargs=model_kwargs,
                    trust_remote_code=True
                )
                
                print(f"  Model loaded on {device}")
                
            except ImportError as e:
                raise ImportError(
                    f"transformers and torch are required for local backend. "
                    f"Install with: pip install transformers torch. Error: {e}"
                )
        
        return self._pipeline
    
    def is_available(self) -> bool:
        """Check if local inference is available."""
        try:
            import torch
            from transformers import pipeline
            return True
        except ImportError:
            return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using local model.
        
        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters
            
        Returns:
            Generated text (excluding the prompt)
        """
        pipe = self._get_pipeline()
        
        params = {
            'max_new_tokens': kwargs.get('max_new_tokens', self.max_new_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'do_sample': kwargs.get('temperature', self.temperature) > 0,
            'return_full_text': False,  # Only return generated text
            'pad_token_id': pipe.tokenizer.eos_token_id,
        }
        
        result = pipe(prompt, **params)
        
        # Extract generated text
        if result and len(result) > 0:
            return result[0]['generated_text']
        
        return ""


def get_backend(
    backend_type: str,
    model_name: str,
    **kwargs
) -> BaseInferenceBackend:
    """
    Factory function to create an inference backend.
    
    Args:
        backend_type: Type of backend ('hf_api' or 'local')
        model_name: Model identifier
        **kwargs: Additional backend configuration
        
    Returns:
        Configured inference backend
    """
    if backend_type == "hf_api":
        return HFAPIBackend(model_name=model_name, **kwargs)
    elif backend_type == "local":
        return LocalHFBackend(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. Use 'hf_api' or 'local'.")



