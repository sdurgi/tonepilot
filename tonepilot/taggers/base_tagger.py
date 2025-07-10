"""
Base class for emotion taggers.
"""
from abc import ABC, abstractmethod
from typing import Dict

class BaseTagger(ABC):
    """Base class for all emotion taggers."""
    
    @abstractmethod
    def classify(self, text: str) -> Dict[str, float]:
        """
        Classify the emotional content of text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Dict[str, float]: Mapping of emotion labels to confidence scores (0.0 to 1.0)
        """
        pass 