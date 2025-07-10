"""
Factory for creating emotion taggers.
"""
from typing import Optional
from ..taggers.base_tagger import BaseTagger
from ..taggers.hf_tagger import HFTagger
from ..taggers.gemini_tagger import GeminiTagger

def get_tagger(config_path: Optional[str] = None, mode: Optional[str] = None) -> BaseTagger:
    """
    Creates and returns an emotion tagger based on the specified mode.
    
    Args:
        config_path (str, optional): Path to config file for HF tagger containing custom labels.
            If None, uses default labels.
        mode (str, optional): The mode to use ('hf' for HuggingFace or 'gemini' for Gemini).
            If None, defaults to 'hf'.
            
    Returns:
        BaseTagger: An emotion tagger instance
        
    Raises:
        ValueError: If an invalid mode is specified
    """
    # Default to HF if no mode specified
    if mode is None or mode.lower() == 'hf':
        return HFTagger(config_path)
    elif mode.lower() == 'gemini':
        return GeminiTagger()
    else:
        raise ValueError(f"Invalid tagger mode: {mode}. Must be 'hf' or 'gemini'") 