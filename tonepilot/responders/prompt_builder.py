"""
Prompt builder for blending personality traits into prompts.

This module provides functionality to blend multiple personality traits
into coherent prompts for text generation.
"""

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

@dataclass
class PromptData:
    """Data class to hold prompt information."""
    prompt: str
    personality_prompt: str
    input_text: str
    final_prompt: str
    response_length: int

class PromptBuilder:
    """
    Handles blending of personality prompts for text generation.
    
    This class is responsible for loading personality libraries and
    creating blended prompts based on emotional tone weights.
    """
    
    def __init__(self, library_path: str = None) -> None:
        """
        Initialize the prompt builder.
        
        Args:
            library_path (str, optional): Path to personality library YAML file
        """
        try:
            # Load personality library
            library_path = library_path or Path(__file__).parent.parent / "config" / "personality_library.yaml"
            with open(library_path, 'r') as f:
                self.library = yaml.safe_load(f)
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize prompt builder: {str(e)}")

    def blend(self, input_text: str, response_tags: Dict[str, bool], 
              weights: Dict[str, float]) -> PromptData:
        """
        Blend personality prompts based on emotional tone weights.
        
        Args:
            input_text (str): The user's input text
            response_tags (dict): Boolean flags for selected response tones
            weights (dict): Confidence scores for each response tone
            
        Returns:
            PromptData: Object containing the blended prompt information
            
        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If blending fails
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
            
            # Generate full prompt with input text and length suggestion
            full_prompt = f"{personality_prompt}\n\nUser: {input_text}\nAssistant: (Aim to respond in about {length} words)"
            
            # Return prompt data
            return PromptData(
                prompt=full_prompt,
                personality_prompt=personality_prompt,
                input_text=input_text,
                final_prompt=full_prompt,
                response_length=length
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to blend prompts: {str(e)}")

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