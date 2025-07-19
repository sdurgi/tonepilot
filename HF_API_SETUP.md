# 🤗 TonePilot + Hugging Face API Setup Guide

This guide helps you upload your BERT classifier to Hugging Face and use the API for lighter, cloud-based tone mapping.

## 📋 Prerequisites

1. **Hugging Face Account**: [Sign up here](https://huggingface.co/join)
2. **API Token**: [Get your token](https://huggingface.co/settings/tokens)
3. **Model File**: Your trained `tonepilot_bert_classifier.pt`

## 🚀 Step 1: Upload Your Model

### 1.1 Update Upload Script
Edit `upload_to_hf.py`:
```python
# Set your details here
HF_USERNAME = "your-actual-username"  # Replace with your HF username
MODEL_NAME = "tonepilot-bert-classifier"
```

### 1.2 Set Environment Variables
```bash
# Set your HF token
export HF_TOKEN="your_huggingface_token_here"

# Or add to .env file
echo "HF_TOKEN=your_huggingface_token_here" >> .env
```

### 1.3 Install Dependencies
```bash
pip install huggingface_hub transformers torch
```

### 1.4 Run Upload Script
```bash
python upload_to_hf.py
```

Expected output:
```
🤗 TonePilot BERT Classifier → Hugging Face Hub
==================================================
📝 Model info:
  - Tokenizer: roberta-base
  - Labels: 73 classes
🚀 Creating repository: username/tonepilot-bert-classifier
🔄 Converting model to HF format...
💾 Saving model and tokenizer...
⬆️  Uploading to username/tonepilot-bert-classifier...
✅ Model uploaded successfully!
🔗 Model URL: https://huggingface.co/username/tonepilot-bert-classifier
```

## 🔧 Step 2: Configure TonePilot for HF API

### 2.1 Set Environment Variables
```bash
# Add to .env file
echo "HF_MODEL_ID=username/tonepilot-bert-classifier" >> .env
echo "HF_TOKEN=your_huggingface_token_here" >> .env
```

### 2.2 Test HF API Mode
```bash
# Test with HF API mode
python -m tonepilot.cli.cli "I am feeling excited and nervous" --mapper-mode hf_api

# Compare with local mode
python -m tonepilot.cli.cli "I am feeling excited and nervous" --mapper-mode local
```

## 💡 Usage Examples

### Local Mode (Default)
```bash
python -m tonepilot.cli.cli "I'm confused about my next steps"
# Uses local BERT model
```

### HF API Mode
```bash
python -m tonepilot.cli.cli "I'm confused about my next steps" --mapper-mode hf_api
# Uses HuggingFace API
```

### With Response Generation
```bash
python -m tonepilot.cli.cli "I feel overwhelmed" --mapper-mode hf_api --respond
```

## 🔄 Benefits of HF API Mode

### ✅ **Advantages:**
- **Lighter Application**: No local model loading (saves ~500MB RAM)
- **Faster Startup**: No model initialization time
- **Cloud Inference**: Leverages HF's optimized infrastructure
- **Easy Sharing**: Model is publicly accessible
- **Automatic Scaling**: HF handles traffic spikes

### ⚠️ **Considerations:**
- **Internet Required**: Needs stable connection
- **API Latency**: ~200-500ms per request
- **Rate Limits**: Free tier has usage limits
- **API Costs**: May incur costs for high usage

## 🔍 Troubleshooting

### Model Loading Issues
```
🔄 Model loading... (attempt 1/3)
```
**Solution**: First API call warms up the model (10-30 seconds)

### Authentication Errors
```
❌ HF API request failed: 401 Unauthorized
```
**Solution**: Check your `HF_TOKEN` is correct and has write permissions

### Missing Environment Variables
```
⚠️ Missing HF_MODEL_ID or HF_TOKEN environment variables
Falling back to local mapper...
```
**Solution**: Set required environment variables in `.env` file

### Network Issues
```
🔄 Retrying API request... (attempt 1/3)
```
**Solution**: Check internet connection and HF service status

## 📊 Performance Comparison

| Mode | Startup Time | Memory Usage | Latency | Offline Support |
|------|-------------|--------------|---------|-----------------|
| **Local** | ~10s | ~500MB | ~50ms | ✅ Yes |
| **HF API** | ~1s | ~50MB | ~300ms | ❌ No |

## 🔧 Integration in Your App

### Python Code
```python
from tonepilot.core.tonepilot import TonePilotEngine

# Local mode
engine_local = TonePilotEngine(mapper_mode='local')

# HF API mode  
engine_api = TonePilotEngine(mapper_mode='hf_api')

# Process text
result = engine_api.run("I'm feeling excited!")
print(result['response_weights'])
```

### Environment Setup
```python
import os
os.environ['HF_TOKEN'] = 'your_token'
os.environ['HF_MODEL_ID'] = 'username/tonepilot-bert-classifier'
```

## 🚀 Production Deployment

### For Production Apps:
1. **Use HF API mode** for lighter deployment
2. **Cache results** to reduce API calls
3. **Implement fallback** to local mode if API fails
4. **Monitor usage** to avoid rate limits

### Hybrid Approach:
```python
# Try HF API first, fallback to local
try:
    engine = TonePilotEngine(mapper_mode='hf_api')
except:
    engine = TonePilotEngine(mapper_mode='local')
```

## 📝 Next Steps

1. **Upload your model** using the provided script
2. **Test both modes** to compare performance  
3. **Choose the mode** that fits your use case
4. **Deploy** with confidence! 🎉

---
**Questions?** Check the [TonePilot documentation](https://github.com/your-repo/tonepilot) or open an issue. 