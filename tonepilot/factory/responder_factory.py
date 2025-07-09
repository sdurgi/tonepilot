"""
Factory module for creating text response generators.

This module provides a factory function to create the appropriate responder instance
based on the specified mode and configuration.
"""

from typing import Optional
from tonepilot.responders.base_responder import BaseResponder
from tonepilot.responders.huggingface_responder import HuggingFaceResponder
from tonepilot.responders.gemini_responder import GeminiResponder

def get_responder(mode: str = 'hf', config_path: Optional[str] = None) -> BaseResponder:
    """
    Factory function to get the appropriate responder based on mode.
    
    This function creates and returns a responder instance that can generate
    contextually appropriate responses with specified emotional tones.
    
    Args:
        mode (str): The mode to use for response generation
            - 'hf': Uses HuggingFace models (default)
            - 'gemini': Uses Google's Gemini model
        config_path (str, optional): Path to configuration file for customization
            The config file should be in YAML format with appropriate settings
            
    Returns:
        BaseResponder: A responder instance implementing the blend(text, tags, weights) method
        
    Raises:
        ValueError: If an unsupported mode is provided
        FileNotFoundError: If config_path is invalid
        RuntimeError: If responder initialization fails
    """
    if not isinstance(mode, str):
        raise ValueError("Mode must be a string")
        
    mode = mode.lower()
    
    try:
        if mode == 'hf':
            return HuggingFaceResponder(library_path=config_path)
        elif mode == 'gemini':
            return GeminiResponder(library_path=config_path)
        else:
            raise ValueError(f"Unsupported responder mode: {mode}")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise RuntimeError(f"Failed to create responder: {str(e)}") 