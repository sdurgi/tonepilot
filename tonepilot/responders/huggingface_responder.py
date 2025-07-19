"""
HuggingFace-based text response generator.

This module provides a responder implementation that uses HuggingFace's
transformers for text generation.
"""

from typing import Tuple
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
                model="microsoft/DialoGPT-medium",
                device=-1,  # Use CPU
                max_new_tokens=100,
                torch_dtype="auto",
                pad_token_id=50256  # Set pad token for DialoGPT
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFace model: {str(e)}")

    def generate_response(self, prompt: str, max_tokens: int) -> Tuple[str, str]:
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
            # Create a cleaner conversational format for DialoGPT
            conversation_prompt = prompt.replace("Assistant: (Aim to respond in about", "Human: (Please respond in about").replace(" words)", " words)")
            final_prompt = f"{conversation_prompt}\nBot:"
            
            response = self.model(
                final_prompt, 
                max_new_tokens=min(max_tokens, 100),
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
            
            full_text = response[0]['generated_text']
            
            # Extract only the new content after our prompt
            response_text = full_text[len(final_prompt):].strip()
            
            # Clean up the response
            if response_text.startswith("Human:") or response_text.startswith("Bot:"):
                response_text = response_text.split(":", 1)[1].strip()
            
            # Stop at conversation turns or end tokens
            response_text = response_text.split('\nHuman:')[0].strip()
            response_text = response_text.split('\nBot:')[0].strip()
            response_text = response_text.split('<|endoftext|>')[0].strip()
            
            # If response is too short or nonsensical, provide a fallback
            if len(response_text.split()) < 3:
                response_text = "I understand this is a challenging situation. Take some time to reflect on what you want to achieve in this conversation with your friend."
            
            return response_text, final_prompt
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {str(e)}")