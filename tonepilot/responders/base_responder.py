"""
Base interface for text response generators.

This module defines the base interface that all responders must implement,
ensuring consistent behavior across different implementations.
"""

import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

@dataclass
class ResponseData:
    """Data class to hold response information."""
    text: str
    prompt: str
    personality_prompt: str
    input_text: str
    final_prompt: str  # The actual prompt sent to the model after any modifications
    response_length: int  # Target response length in tokens

class BaseResponder(ABC):
    """
    Abstract base class for text response generators.
    
    All responder implementations must inherit from this class and implement
    its abstract methods to ensure consistent behavior.
    """
    
    def __init__(self, library_path: Optional[str] = None) -> None:
        """
        Initialize the responder.
        
        Args:
            library_path (str, optional): Path to personality library YAML file
        """
        try:
            # Load personality library
            library_path = library_path or Path(__file__).parent.parent / "config" / "personality_library.yaml"
            with open(library_path, 'r') as f:
                self.library = yaml.safe_load(f)
                
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
    def _generate_response(self, prompt: str, max_tokens: int) -> Tuple[str, str]:
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

    def blend(self, input_text: str, response_tags: Dict[str, bool], 
              weights: Dict[str, float]) -> ResponseData:
        """
        Blend personality prompts and generate a response.
        
        Args:
            input_text (str): The user's input text
            response_tags (dict): Boolean flags for selected response tones
            weights (dict): Confidence scores for each response tone
            
        Returns:
            ResponseData: Object containing the generated response text and prompt information
            
        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If generation fails
        """
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("input_text must be a non-empty string")
            
        try:
            # Build personality prompt
            prompt = []
            for tag, weight in weights.items():
                if tag in self.library:
                    style = self.library[tag]
                    # Repeat based on weight (1-3 times)
                    repeat = max(1, min(3, round(weight * 3)))
                    prompt.extend([style.strip()] * repeat)
            
            personality_prompt = " ".join(prompt)
            
            # Determine response length
            length = self._determine_response_length(input_text, weights)
            
            # Generate full prompt with input text
            full_prompt = f"{personality_prompt}\n\nUser: {input_text}\nAssistant:"
            
            # Generate response using model-specific implementation and get final prompt
            response_text, final_prompt = self._generate_response(full_prompt, length)
            
            # Return response data with all information
            return ResponseData(
                text=response_text,
                prompt=full_prompt,
                personality_prompt=personality_prompt,
                input_text=input_text,
                final_prompt=final_prompt,
                response_length=length
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {str(e)}")

    def _determine_response_length(self, input_text: str, 
                               weights: Dict[str, float]) -> int:
        """
        Determine appropriate response length based on input and personality.
        
        Args:
            input_text (str): The user's input text
            weights (dict): Confidence scores for response tones
            
        Returns:
            int: Target response length in tokens
        """
        base_length = len(input_text.split())
        personality_factor = sum(weights.values())
        # Allow for longer responses (up to 500 tokens) and increase the multiplier
        return int(min(500, max(50, base_length * personality_factor * 12))) 