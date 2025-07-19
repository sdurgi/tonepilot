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
from tonepilot.tonemapper.hf_api_tonemapper import HFApiToneMapper
from tonepilot.responders.prompt_builder import PromptBuilder

load_dotenv()  # Load environment variables from .env

class TonePilotEngine:
    """
    The main engine for TonePilot that coordinates tagging, tone mapping, and response generation.
    """
    
    def __init__(self, mode: str = 'hf', respond: bool = False, mapper_mode: str = 'local') -> None:
        """
        Initialize the TonePilot engine.
        
        Args:
            mode (str): The mode to use for response generation ('hf' or 'gemini')
            respond (bool): Whether to generate responses or just return prompts
            mapper_mode (str): Mode for tone mapping ('local' or 'hf_api')
                - 'local': Use local BERT model (default)
                - 'hf_api': Use Hugging Face API (requires HF_TOKEN and HF_MODEL_ID)
        """
        try:
            # Initialize components
            self.tagger = get_tagger()  # Always use HF tagger
            self.mapper_mode = mapper_mode
            self.mapper = self._initialize_mapper(mapper_mode)
            self.prompt_builder = PromptBuilder()  # For prompt blending
            self.mode = mode
            self.respond = respond
            self._responder = None  # Lazy initialization
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TonePilot engine: {str(e)}")

    def _initialize_mapper(self, mapper_mode: str):
        """Initialize the appropriate tone mapper based on mode."""
        if mapper_mode == 'local':
            return ToneMapper()  # Use local BERT classifier
        elif mapper_mode == 'hf_api':
            # Use HF API-based mapper
            model_id = os.getenv("HF_MODEL_ID")
            api_token = os.getenv("HF_TOKEN")
            
            if not model_id or not api_token:
                print("⚠️  Missing HF_MODEL_ID or HF_TOKEN environment variables")
                print("Falling back to local mapper...")
                return ToneMapper()
            
            try:
                return HFApiToneMapper(model_id=model_id, api_token=api_token)
            except Exception as e:
                print(f"⚠️  Failed to initialize HF API mapper: {e}")
                print("Falling back to local mapper...")
                return ToneMapper()
        else:
            raise ValueError(f"Unsupported mapper_mode: {mapper_mode}. Use 'local' or 'hf_api'")

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
            
            # Safety check: ensure we have input tags
            if not input_tags:
                print("⚠️  Core engine: No input tags detected, adding defaults")
                input_tags = {"curious": 0.200, "thoughtful": 0.180}
            
            # Map input tags to response tags
            response_tags, weights = self.mapper.map_tags(input_tags)
            
            # Safety check: ensure we have response tags and weights
            if not response_tags or not weights:
                print("⚠️  Core engine: No response tags/weights found, adding defaults")
                response_tags = {"supportive": True, "thoughtful": True}
                weights = {"supportive": 0.150, "thoughtful": 0.130}
            
            # Create blended prompt using PromptBuilder
            prompt_data = self.prompt_builder.blend(input_text, response_tags, weights)
            
            # Base result
            result = {
                "input_text": input_text,
                "input_tags": input_tags,
                "response_tags": response_tags,
                "response_weights": weights,
                "final_prompt": prompt_data.final_prompt,
                "response_length": prompt_data.response_length,
                "mapper_mode": self.mapper_mode
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
