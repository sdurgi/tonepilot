"""
Emotion tagger using Google's Gemini model.
"""
import os
import json
import google.generativeai as genai
from typing import List, Dict, Any
from .base_tagger import BaseTagger

class GeminiTagger(BaseTagger):
    """Emotion tagger using Google's Gemini model."""
    
    def __init__(self):
        """Initialize the Gemini tagger."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
            
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
    def classify(self, text: str) -> Dict[str, float]:
        """
        Classify the emotional content of text using Gemini.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Dict[str, float]: Dictionary mapping emotion labels to confidence scores
        """
        prompt = """
        Analyze the emotional content of the following text and return a JSON dictionary mapping emotions to their confidence scores (0-1):
        
        Text: {text}
        
        Focus on these emotions: joy, sadness, anger, fear, surprise, love
        Example format: {{"joy": 0.8, "sadness": 0.2}}
        """.format(text=text)
        
        response = self.model.generate_content(prompt)
        try:
            # Extract the JSON dictionary from the response
            result_text = response.text
            # Find the dictionary portion
            start = result_text.find('{')
            end = result_text.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("Could not find JSON dictionary in response")
            json_str = result_text[start:end]
            return json.loads(json_str)
        except Exception as e:
            raise RuntimeError(f"Failed to parse Gemini response: {str(e)}") 