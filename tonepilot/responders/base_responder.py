"""
Base interface for text response generators.

This module defines the base interface that all responders must implement,
ensuring consistent behavior across different implementations.
"""

from abc import ABC, abstractmethod
from typing import Tuple

class BaseResponder(ABC):
    """
    Abstract base class for text response generators.
    
    All responder implementations must inherit from this class and implement
    its abstract methods to ensure consistent behavior.
    """
    
    def __init__(self) -> None:
        """
        Initialize the responder.
        """
        try:
            # Initialize model-specific components
            self._initialize_model()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize responder: {str(e)}")
    
    @abstractmethod
    def _initialize_model(self) -> None:
        """
        Initialize the specific LLM model.
        Must be implemented by child classes.
        
        Raises:
            RuntimeError: If model initialization fails
        """
        pass
        
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int) -> Tuple[str, str]:
        """
        Generate response using the specific LLM model.
        Must be implemented by child classes.
        
        Args:
            prompt (str): The full prompt to send to the model
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            Tuple[str, str]: The generated response text and the final prompt used
            
        Raises:
            RuntimeError: If generation fails
        """
        pass 