"""
TonePilot Core Engine
This module provides the main engine for the TonePilot system, which processes text through
a pipeline of tone detection, mapping, and response generation.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union
from dotenv import load_dotenv
from tonepilot.factory.tagger_factory import get_tagger
from tonepilot.factory.responder_factory import get_responder
from tonepilot.tonemapper.tonemapper import ToneMapper

load_dotenv()  # Load environment variables from .env

class TonePilotEngine:
    """
    The main engine for TonePilot that coordinates tagging, tone mapping, and response generation.
    """
    
    def __init__(self, mode: str = 'hf') -> None:
        """Initialize the TonePilot engine."""
        try:
            # Initialize components
            self.tagger = get_tagger(mode)
            self.responder = get_responder(mode)
            self.mapper = ToneMapper()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TonePilot engine: {str(e)}")

    def run(self, input_text: str) -> Dict[str, Union[str, Dict[str, float]]]:
        """Process input text through the TonePilot pipeline."""
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("Input text must be a non-empty string")
            
        try:
            # Detect input tags
            input_tags = self.tagger.classify(input_text)
            
            # Map input tags to response tags
            response_tags, weights = self.mapper.map_tags(input_tags)
            
            # Generate response with prompts
            response_data = self.responder.blend(input_text, response_tags, weights)

            return {
                "input_text": input_text,
                "input_tags": input_tags,
                "response_tags": response_tags,
                "response_weights": weights,
                "response_text": response_data.text,
                "final_prompt": response_data.final_prompt,
                "response_length": response_data.response_length
            }
        except Exception as e:
            raise RuntimeError(f"Failed to process text: {str(e)}")
