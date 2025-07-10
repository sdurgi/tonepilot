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
from tonepilot.responders.prompt_builder import PromptBuilder

load_dotenv()  # Load environment variables from .env

class TonePilotEngine:
    """
    The main engine for TonePilot that coordinates tagging, tone mapping, and response generation.
    """
    
    def __init__(self, mode: str = 'hf', respond: bool = False) -> None:
        """
        Initialize the TonePilot engine.
        
        Args:
            mode (str): The mode to use for response generation ('hf' or 'gemini')
            respond (bool): Whether to generate responses or just return prompts
        """
        try:
            # Initialize components
            self.tagger = get_tagger()  # Always use HF tagger
            self.mapper = ToneMapper()  # Use BERT classifier
            self.prompt_builder = PromptBuilder()  # For prompt blending
            self.mode = mode
            self.respond = respond
            self._responder = None  # Lazy initialization
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TonePilot engine: {str(e)}")

    @property
    def responder(self):
        """Lazy initialization of responder - only create when needed."""
        if self._responder is None:
            self._responder = get_responder(self.mode)
        return self._responder

    def run(self, input_text: str) -> Dict[str, Union[str, Dict[str, float]]]:
        """Process input text through the TonePilot pipeline."""
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("Input text must be a non-empty string")
            
        try:
            # Detect input tags
            input_tags = self.tagger.classify(input_text)
            
            # Map input tags to response tags
            response_tags, weights = self.mapper.map_tags(input_tags)
            
            # Create blended prompt using PromptBuilder
            prompt_data = self.prompt_builder.blend(input_text, response_tags, weights)
            
            # Base result
            result = {
                "input_text": input_text,
                "input_tags": input_tags,
                "response_tags": response_tags,
                "response_weights": weights,
                "final_prompt": prompt_data.final_prompt,
                "response_length": prompt_data.response_length
            }
            
            if self.respond:
                # Generate actual response using responder
                response_text, final_prompt = self.responder.generate_response(
                    prompt_data.prompt, 
                    prompt_data.response_length
                )
                result.update({
                    "final_prompt": final_prompt,
                    "response_text": response_text,
                })
            else:
                # Just return the prompt without generating response
                result.update({
                    "response_text": "",
                })
                
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to process text: {str(e)}")
