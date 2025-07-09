"""
HuggingFace-based text response generator.

This module provides a responder implementation that uses HuggingFace's
transformers for text generation.
"""

from typing import Optional, Tuple
from transformers import pipeline
from .base_responder import BaseResponder

class HuggingFaceResponder(BaseResponder):
    """
    Text response generator using HuggingFace models.
    
    This class uses HuggingFace's transformers library to generate
    contextually appropriate responses with specified emotional tones.
    """
    
    def _initialize_model(self) -> None:
        """
        Initialize the HuggingFace model.
            
        Raises:
            RuntimeError: If model initialization fails
        """
        try:
            self.model = pipeline(
                "text-generation",
                model="tiiuae/falcon-7b-instruct",
                device=-1,  # Use CPU
                max_new_tokens=100,
                torch_dtype="auto"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFace model: {str(e)}")

    def _generate_response(self, prompt: str, max_tokens: int) -> Tuple[str, str]:
        """
        Generate response using HuggingFace model.
        
        Args:
            prompt (str): The full prompt to send to the model
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            Tuple[str, str]: The generated response text and the final prompt used
            
        Raises:
            RuntimeError: If generation fails
        """
        try:
            # For HuggingFace, we need to ensure the prompt ends with a space
            # to prevent token merging issues
            final_prompt = prompt.rstrip() + " "
            
            response = self.model(final_prompt, max_new_tokens=max_tokens)
            full_text = response[0]['generated_text']
            
            # Extract only the new content after our prompt
            response_text = full_text[len(final_prompt):].strip()
            if response_text.startswith("AI:"):
                response_text = response_text[3:].strip()
            
            return response_text, final_prompt
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {str(e)}")