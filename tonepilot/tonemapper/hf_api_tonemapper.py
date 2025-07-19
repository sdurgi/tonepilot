"""
Hugging Face API-based tone mapping module.

This module provides a lightweight ToneMapper that uses Hugging Face's API
instead of loading the model locally, reducing memory usage and startup time.
"""

import requests
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class HFResponse:
    """Response from Hugging Face API."""
    label: str
    score: float

class HFApiToneMapper:
    """
    Maps input emotional tones to appropriate response tones using HF API.
    
    This class uses the Hugging Face Inference API to perform tone mapping
    without requiring local model loading.
    """
    
    def __init__(self, model_id: Optional[str] = None, api_token: Optional[str] = None) -> None:
        """
        Initialize the HF API tone mapper.
        
        Args:
            model_id (str, optional): HuggingFace model ID (e.g., "username/tonepilot-bert-classifier")
            api_token (str, optional): HuggingFace API token. If None, uses HF_TOKEN env var
            
        Raises:
            ValueError: If model_id or api_token are not provided
            RuntimeError: If API connection fails
        """
        self.model_id = model_id or os.getenv("HF_MODEL_ID")
        self.api_token = api_token or os.getenv("HF_TOKEN")
        
        if not self.model_id:
            raise ValueError("model_id must be provided or set HF_MODEL_ID environment variable")
        if not self.api_token:
            raise ValueError("api_token must be provided or set HF_TOKEN environment variable")
        
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        
        # Test API connection
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test the HF API connection."""
        try:
            test_response = self._query("test")
            print(f"✅ Connected to HF API: {self.model_id}")
        except Exception as e:
            print(f"⚠️  HF API connection warning: {e}")
            print("API may be warming up. Will retry on first actual request.")
    
    def _query(self, text: str, retry_count: int = 3) -> List[Dict]:
        """
        Query the Hugging Face API.
        
        Args:
            text (str): Input text to classify
            retry_count (int): Number of retries if API is loading
            
        Returns:
            list: List of classification results
            
        Raises:
            RuntimeError: If API request fails after retries
        """
        payload = {"inputs": text}
        
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    # Model is loading
                    print(f"🔄 Model loading... (attempt {attempt + 1}/{retry_count})")
                    if attempt < retry_count - 1:
                        import time
                        time.sleep(10)  # Wait 10 seconds
                        continue
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                if attempt == retry_count - 1:
                    raise RuntimeError(f"HF API request failed: {e}")
                print(f"🔄 Retrying API request... (attempt {attempt + 1}/{retry_count})")
                import time
                time.sleep(5)
        
        raise RuntimeError("Max retries exceeded")
    
    def map_tags(self, input_tags: Dict[str, float], top_k: int = 3) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """
        Map input emotional tags to appropriate response tags using HF API.
        
        Args:
            input_tags (dict): Dictionary of input tags and their confidence scores
            top_k (int, optional): Number of top response tags to return
            
        Returns:
            tuple: (response_tags, weights)
                - response_tags (dict): Boolean flags for selected response tones
                - weights (dict): Confidence scores for selected response tones
                
        Raises:
            ValueError: If input_tags is invalid
            RuntimeError: If API request fails
        """
        if not isinstance(input_tags, dict) or not input_tags:
            raise ValueError("input_tags must be a non-empty dictionary")
        
        try:
            # Convert input tags to text (same format as training)
            text = ", ".join(input_tags.keys())
            
            # Query HF API
            results = self._query(text)
            
            # Parse results - handle both single prediction and batch format
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    # Batch format: [[{label, score}, ...]]
                    predictions = results[0]
                else:
                    # Single format: [{label, score}, ...]
                    predictions = results
            else:
                predictions = []
            
            # Filter and sort predictions
            filtered_predictions = [
                (pred["label"], pred["score"]) 
                for pred in predictions 
                if pred["score"] > 0.05  # Apply threshold
            ]
            
            # Sort by score descending
            filtered_predictions.sort(key=lambda x: x[1], reverse=True)
            
            # If no predictions found, use default response tones
            if not filtered_predictions:
                print("⚠️  No response tones found via HF API, using defaults")
                filtered_predictions = [
                    ("supportive", 0.150),
                    ("thoughtful", 0.130),
                    ("helpful", 0.120)
                ]
            
            # Convert to output format
            response_tags = {}
            weights = {}
            
            for label, score in filtered_predictions[:top_k]:
                response_tags[label] = True
                weights[label] = float(score)
            
            return response_tags, weights
            
        except Exception as e:
            raise RuntimeError(f"Failed to map tags via HF API: {str(e)}")
    
    def predict_tones(self, text: str, threshold: float = 0.05) -> List[Tuple[str, float]]:
        """
        Predict appropriate response tones for the input text using HF API.
        
        Args:
            text (str): Input text to analyze
            threshold (float): Minimum probability threshold
            
        Returns:
            list: List of (tone, probability) tuples, sorted by probability
            
        Raises:
            ValueError: If text is invalid
            RuntimeError: If API request fails
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        
        try:
            results = self._query(text)
            
            # Parse results
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    predictions = results[0]
                else:
                    predictions = results
            else:
                return []
            
            # Filter and sort
            filtered = [
                (pred["label"], pred["score"]) 
                for pred in predictions 
                if pred["score"] > threshold
            ]
            
            return sorted(filtered, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            raise RuntimeError(f"Failed to predict tones via HF API: {str(e)}") 