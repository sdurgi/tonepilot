"""
Google Gemini-based text response generator.

This module provides a responder implementation that uses Google's Gemini
model for text generation.
"""

import os
from typing import Tuple
import google.generativeai as genai
from .base_responder import BaseResponder

class GeminiResponder(BaseResponder):
    """
    Text response generator using Google's Gemini model.
    
    This class uses the Gemini API to generate contextually appropriate
    responses with specified emotional tones.
    """
    
    def _initialize_model(self) -> None:
        """
        Initialize the Gemini model (lazy initialization - only configure when needed).
        """
        # Don't initialize here - we'll do it lazily in generate_response
        self.model = None

    def _ensure_model_initialized(self) -> None:
        """
        Ensure the Gemini model is initialized and configured.
        This is called only when we actually need to generate a response.
            
        Raises:
            ValueError: If API key not set
            RuntimeError: If initialization fails
        """
        if self.model is not None:
            return
            
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
                
            genai.configure(api_key=api_key)
                
            self.model = genai.GenerativeModel("gemini-1.5-pro")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini model: {str(e)}")

    def generate_response(self, prompt: str, max_tokens: int) -> Tuple[str, str]:
        """
        Generate response using Gemini model.
        
        Args:
            prompt (str): The full prompt to send to the model
            max_tokens (int): Suggested number of tokens to generate (not a hard limit)
            
        Returns:
            Tuple[str, str]: The generated response text and the final prompt used
            
        Raises:
            RuntimeError: If generation fails
        """
        # Initialize model only when we actually need to generate
        self._ensure_model_initialized()
        
        try:
            # Gemini-specific generation parameters
            # Use max_tokens as a suggestion in the prompt rather than a hard limit
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                # Allow model to exceed the suggested length if needed
                "max_output_tokens": max_tokens * 2  # Double the suggested length as maximum
            }
            
            # Use the prompt as-is since length suggestion is already included
            final_prompt = prompt.replace("\n\n", "\n").strip()
            
            response = self.model.generate_content(
                final_prompt,
                generation_config=generation_config
            )
            
            # Extract only the new content after "Assistant:"
            full_text = response.text.strip()
            assistant_marker = "Assistant:"
            if assistant_marker in full_text:
                response_text = full_text.split(assistant_marker, 1)[1].strip()
            else:
                response_text = full_text
            
            return response_text, final_prompt
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {str(e)}")