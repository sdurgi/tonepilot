"""
Factory module for creating text tone taggers.

This module provides a factory function to create the appropriate tagger instance
based on the specified mode and configuration.
"""

from typing import Optional
from tonepilot.taggers.zero_shot_tagger import HuggingFaceTagger
from tonepilot.taggers.gemini_tagger import GeminiTagger

def get_tagger(mode: str = 'hf', config_path: Optional[str] = None):
    """
    Factory function to get the appropriate tagger based on mode.
    
    This function creates and returns a tagger instance that can detect emotional
    tones in text. Supports both HuggingFace and Gemini-based classification.
    
    Args:
        mode (str): The mode to use for tagging
            - 'hf': Uses HuggingFace zero-shot classification (default)
            - 'gemini': Uses Google's Gemini model
        config_path (str, optional): Path to configuration file for customization
            The config file should be in YAML format with appropriate settings
            
    Returns:
        A tagger instance implementing the classify(text) method
        
    Raises:
        ValueError: If an unsupported mode is provided
        FileNotFoundError: If config_path is invalid
        RuntimeError: If tagger initialization fails
    """
    if not isinstance(mode, str):
        raise ValueError("Mode must be a string")
        
    mode = mode.lower()
    
    try:
        if mode == 'hf':
            return HuggingFaceTagger(config_path)
        elif mode == 'gemini':
            return GeminiTagger(config_path)
        else:
            raise ValueError(f"Unsupported tagger mode: {mode}")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise RuntimeError(f"Failed to create tagger: {str(e)}") 