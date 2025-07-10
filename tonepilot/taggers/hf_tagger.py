"""
Zero-shot emotion and tone classification using HuggingFace's transformers.

This module provides a zero-shot classifier that can detect emotional tones in text
without requiring specific training data for each emotion.
"""

from typing import Dict, List, Optional
from transformers import pipeline
import yaml
from .base_tagger import BaseTagger

class HFTagger(BaseTagger):
    """
    A zero-shot classifier for detecting emotional tones in text.
    
    This class uses the BART-large-MNLI model to perform zero-shot classification
    of text into predefined emotional categories. It can be configured with custom
    emotional labels through a config file.
    """
    
    DEFAULT_LABELS = [
        "curious", "angry", "sad", "excited", "confused",
        "hopeful", "tired", "scared", "playful", "assertive"
    ]
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the tagger with optional custom configuration.
        
        Args:
            config_path (str, optional): Path to YAML config file containing custom labels
                The config should have a 'labels' key with a list of emotional labels
                
        Raises:
            FileNotFoundError: If config_path is provided but file doesn't exist
            ValueError: If config file is invalid
        """
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1  # Use CPU by default
            )
            
            # Load custom labels from config if provided
            self.labels = self._load_labels(config_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tagger: {str(e)}")
    
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

    def classify(self, text: str) -> Dict[str, float]:
        """
        Classify the emotional tones present in the input text.
        
        Args:
            text (str): The text to classify
                
        Returns:
            Dict[str, float]: Mapping of emotion labels to confidence scores (0.0 to 1.0)
                
        Raises:
            ValueError: If text is empty or not a string
            RuntimeError: If classification fails
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")
            
        try:
            result = self.classifier(text, self.labels)
            scores = dict(zip(result["labels"], result["scores"]))
            return {k: round(float(v), 3) for k, v in scores.items() if float(v) > 0.2}
            
        except Exception as e:
            raise RuntimeError(f"Classification failed: {str(e)}") 