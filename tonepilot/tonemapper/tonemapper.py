"""
Tone mapping module for emotional response generation.

This module provides a BERT-based classifier that maps input emotional tones
to appropriate response tones, enabling contextually appropriate emotional responses.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MultiLabelBinarizer
from torch.serialization import add_safe_globals
import urllib.request
from pathlib import Path

# Add MultiLabelBinarizer to safe globals for model loading
add_safe_globals([MultiLabelBinarizer])

class BERTToneClassifier(nn.Module):
    """
    BERT-based classifier for tone mapping.
    
    This model uses RoBERTa as the base encoder and adds a classification head
    for multi-label tone classification.
    """
    
    def __init__(self, model_name: str = 'roberta-base', num_labels: Optional[int] = None) -> None:
        """
        Initialize the classifier.
        
        Args:
            model_name (str): Name of the base model to use
            num_labels (int, optional): Number of output labels
        """
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels) if num_labels else None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Token IDs
            attention_mask (torch.Tensor): Attention mask
            
        Returns:
            torch.Tensor: Classification logits
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (batch_size, sequence_length, hidden_size)
        
        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sequence_output *= mask_expanded
        summed = torch.sum(sequence_output, dim=1)
        count = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_output = summed / count
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class ToneMapper:
    """
    Maps input emotional tones to appropriate response tones.
    
    This class uses a fine-tuned BERT model to determine which emotional tones
    would be appropriate in a response given the input emotional context.
    """
    
    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the tone mapper.
        
        Args:
            model_path (str, optional): Path to the trained model checkpoint.
                                      If None, uses default cache location.
            
        Raises:
            FileNotFoundError: If model_path doesn't exist and download fails
            RuntimeError: If model loading fails
        """
        try:
            # Determine model path
            if model_path is None:
                model_path = self._get_default_model_path()
            
            # Download model if not found locally
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}, downloading...")
                self._download_model(model_path)
            
            # Load checkpoint on CPU for compatibility
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Get model configuration
            tokenizer_name = checkpoint.get('tokenizer_name', 'roberta-base')
            self.label_binarizer = checkpoint['label_binarizer']
            
            # Initialize model components
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.model = BERTToneClassifier(tokenizer_name, len(self.label_binarizer.classes_))
            
            # Load and convert state dict
            self._load_state_dict(checkpoint['model_state_dict'])
            
            # Set up device
            self.device = self._setup_device()
            self.model.eval()
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tone mapper: {str(e)}")
    
    def _load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load and convert model state dict."""
        new_state_dict = {}
        model_state_keys = set(self.model.state_dict().keys())
        
        for key in state_dict:
            new_key = key.replace('bert.', 'roberta.')
            if new_key in model_state_keys:
                new_state_dict[new_key] = state_dict[key]
        
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")
    
    def _setup_device(self) -> torch.device:
        """Set up and test compute device."""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            try:
                # Test MPS device
                test_tensor = torch.ones(1, device=device)
                test_tensor = test_tensor + 1
                print("Using MPS device")
                self.model = self.model.to(device)
                return device
            except Exception as e:
                print(f"MPS device test failed: {e}")
                print("Falling back to CPU")
        else:
            print("MPS not available, using CPU")
        return torch.device("cpu")

    def _download_model(self, model_path: str) -> None:
        """Download the BERT classifier model from GitHub releases."""
        model_url = "https://github.com/sdurgi/tonepilot/releases/download/v0.1.0/tonepilot_bert_classifier.pt"
        
        try:
            print("Downloading BERT classifier model (475 MB)...")
            print("This may take a few minutes on first run...")
            
            # Create directory if it doesn't exist
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress indication
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    if block_num % 100 == 0:  # Update every 100 blocks
                        print(f"Progress: {percent}%")
            
            urllib.request.urlretrieve(model_url, model_path, progress_hook)
            print(f"Model downloaded successfully to {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {str(e)}")

    def _get_default_model_path(self) -> str:
        """Get the default path for the model file."""
        # Try multiple locations in order of preference
        
        # 1. Current working directory (for backward compatibility)
        current_dir_path = "tonepilot_bert_classifier.pt"
        if os.path.exists(current_dir_path):
            return current_dir_path
        
        # 2. User's home cache directory
        home_cache = Path.home() / ".cache" / "tonepilot" / "tonepilot_bert_classifier.pt"
        if home_cache.exists():
            return str(home_cache)
        
        # 3. Package directory (if installed from wheel)
        try:
            import tonepilot
            package_dir = Path(tonepilot.__file__).parent.parent
            package_model = package_dir / "tonepilot_bert_classifier.pt"
            if package_model.exists():
                return str(package_model)
        except:
            pass
        
        # 4. Default to cache directory for new download
        return str(home_cache)

    def map_tags(self, input_tags: Dict[str, float], top_k: int = 3) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """
        Map input emotional tags to appropriate response tags.
        
        Args:
            input_tags (dict): Dictionary of input tags and their confidence scores
            top_k (int, optional): Number of top response tags to return
            
        Returns:
            tuple: (response_tags, weights)
                - response_tags (dict): Boolean flags for selected response tones
                - weights (dict): Confidence scores for selected response tones
                
        Raises:
            ValueError: If input_tags is invalid
            RuntimeError: If mapping fails
        """
        if not isinstance(input_tags, dict) or not input_tags:
            raise ValueError("input_tags must be a non-empty dictionary")
            
        try:
            # Convert input tags to text
            text = ", ".join(input_tags.keys())
            
            # Get predictions with lower threshold for more candidates
            predictions = self.predict_tones(text, threshold=0.2)
            
            # Convert to output format
            response_tags = {}
            weights = {}
            for label, prob in predictions[:top_k]:
                response_tags[label] = True
                weights[label] = float(prob)
            
            return response_tags, weights
            
        except Exception as e:
            raise RuntimeError(f"Failed to map tags: {str(e)}")

    def predict_tones(self, text: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Predict appropriate response tones for the input text.
        
        Args:
            text (str): Input text to analyze
            threshold (float): Minimum probability threshold
            
        Returns:
            list: List of (tone, probability) tuples, sorted by probability
            
        Raises:
            ValueError: If text is invalid
            RuntimeError: If prediction fails
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
            
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.sigmoid(outputs)
            
            # Process predictions
            probs = probs.cpu().numpy()
            mask = probs[0] > threshold
            labels = self.label_binarizer.classes_[mask]
            probabilities = probs[0][mask]
            
            # Sort by probability
            sorted_indices = np.argsort(probabilities)[::-1]
            labels = labels[sorted_indices]
            probabilities = probabilities[sorted_indices]
            
            return list(zip(labels, probabilities))
            
        except Exception as e:
            raise RuntimeError(f"Failed to predict tones: {str(e)}") 