"""
Prompt builder for blending personality traits into prompts.

This module provides functionality to blend multiple personality traits
into coherent prompts for text generation.
"""

import json
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
            library_path (str, optional): Path to personality parameters JSON file
        """
        try:
            # Load personality parameters from JSON
            library_path = library_path or Path(__file__).parent.parent / "config" / "tonepilot_response_personality_parameters.json"
            with open(library_path, 'r') as f:
                personality_list = json.load(f)
            
            # Convert list to dictionary for easier lookup
            self.personality_params = {item['tag']: item for item in personality_list}
                
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
            # Build personality prompt using tone instructions from JSON
            prompt_parts = []
            
            # If no weights provided, use defaults
            if not weights:
                print("⚠️  No personality weights provided, using defaults")
                weights = {
                    "supportive": 0.150,
                    "thoughtful": 0.130,
                    "helpful": 0.120
                }
            
            for tag, weight in weights.items():
                if tag in self.personality_params:
                    params = self.personality_params[tag]
                    tone_instruction = params['tone_instruction']
                    
                    # Repeat based on weight (1-3 times)
                    repeat = max(1, min(3, round(weight * 3)))
                    prompt_parts.extend([tone_instruction.strip()] * repeat)
                else:
                    print(f"⚠️  Tag '{tag}' not found in personality parameters, skipping")
            
            # If no valid prompt parts found, use default supportive tone
            if not prompt_parts:
                print("⚠️  No valid personality parameters found, using default tone")
                prompt_parts = ["Provide encouragement and positive reinforcement. Help the user feel seen and not alone."]
            
            personality_prompt = " ".join(prompt_parts)
            
            # Calculate response length using the existing method
            response_length = self._determine_response_length(input_text, weights)
            
            # Generate full prompt with input text and length suggestion
            full_prompt = f"{personality_prompt}\n\nUser: {input_text}\nAssistant: (Aim to respond in about {response_length} words)"
            
            # Return prompt data
            return PromptData(
                prompt=full_prompt,
                personality_prompt=personality_prompt,
                input_text=input_text,
                final_prompt=full_prompt,
                response_length=response_length
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to blend prompts: {str(e)}")

    def _determine_response_length(self, input_text: str, 
                               weights: Dict[str, float]) -> int:
        """
        Calculate response length using JSON parameters and the formula: base + input(scale).
        
        Args:
            input_text (str): The user's input text
            weights (dict): Confidence scores for response tones
            
        Returns:
            int: Target response length in words
        """
        input_length = len(input_text.split())
        total_base = 0
        total_scale = 0
        
        # Accumulate base and scale from personality parameters
        for tag, weight in weights.items():
            if tag in self.personality_params:
                params = self.personality_params[tag]
                total_base += params['base'] * weight
                total_scale += params['scale'] * weight
        
        # If no valid parameters found, use defaults
        if total_base == 0 and total_scale == 0:
            total_base = 30  # Default base length
            total_scale = 0.8  # Default scale factor
        
        # Apply formula: base + input(scale)
        response_length = int(total_base + (input_length * total_scale))
        
        # Ensure reasonable bounds (between 20 and 200 words)
        return max(20, min(200, response_length)) 