"""
Emotion detection using Google's Gemini model.

This module provides a tagger that uses Gemini to detect emotional tones in text
through zero-shot prompting.
"""

import os
from typing import Dict, List, Optional
import google.generativeai as genai
import yaml

class GeminiTagger:
    """
    A Gemini-based classifier for detecting emotional tones in text.
    
    This class uses Google's Gemini model to perform zero-shot classification
    of text into predefined emotional categories through careful prompting.
    """
    
    DEFAULT_LABELS = [
        "curious", "angry", "sad", "excited", "confused",
        "hopeful", "tired", "scared", "playful", "assertive"
    ]
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the Gemini tagger.
        
        Args:
            config_path (str, optional): Path to YAML config file containing custom labels
                The config should have a 'labels' key with a list of emotional labels
                
        Raises:
            ValueError: If config file is invalid or API key is missing
            RuntimeError: If initialization fails
        """
        try:
            # Configure Gemini
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
                
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-pro")
            
            # Load custom labels from config if provided
            self.labels = self._load_labels(config_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini tagger: {str(e)}")
    
    def _load_labels(self, config_path: Optional[str]) -> List[str]:
        """Load emotion labels from config file or use defaults."""
        if not config_path:
            return self.DEFAULT_LABELS
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict) or 'labels' not in config:
                raise ValueError("Config must contain a 'labels' key")
                
            labels = config['labels']
            if not isinstance(labels, list) or not all(isinstance(l, str) for l in labels):
                raise ValueError("Labels must be a list of strings")
                
            return labels
            
        except Exception as e:
            raise ValueError(f"Failed to load config: {str(e)}")

    def classify(self, text: str, threshold: float = 0.2) -> Dict[str, float]:
        """
        Classify the emotional tones present in the input text.
        
        Args:
            text (str): The text to classify
            threshold (float, optional): Minimum confidence score to include a label
                Defaults to 0.2
                
        Returns:
            dict: Mapping of emotion labels to confidence scores
                Only includes scores above the threshold
                
        Raises:
            ValueError: If text is empty or not a string
            RuntimeError: If classification fails
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")
            
        try:
            # Construct the prompt
            prompt = f"""Analyze the emotional tones in the following text. Consider these possible emotions: {', '.join(self.labels)}

Text: "{text}"

For each emotion that is present, assign a confidence score between 0 and 1.
Only include emotions with confidence > {threshold}.
Format your response as a Python dictionary, like this example:
{{"happy": 0.8, "excited": 0.6}}

Response:"""

            # Get Gemini's response
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract the dictionary part
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("Failed to parse Gemini response")
                
            # Evaluate the dictionary string (safe since we control the input)
            scores = eval(response_text[start:end])
            
            # Validate and clean up the scores
            if not isinstance(scores, dict):
                raise ValueError("Invalid response format from Gemini")
                
            return {k: round(float(v), 3) for k, v in scores.items() 
                   if isinstance(v, (int, float)) and v > threshold}
            
        except Exception as e:
            raise RuntimeError(f"Classification failed: {str(e)}") 