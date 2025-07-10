# TonePilot

[![PyPI version](https://badge.fury.io/py/tonepilot.svg)](https://badge.fury.io/py/tonepilot)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

# 🧠 TonePilot  
**Emotionally Intelligent Prompt & Response Engine for AI Chatbots**

> **TonePilot** helps your chatbot *understand the user’s tone and personality*—and respond like a human would.  
> Whether you're building a customer support bot, a mental health assistant, or a flirty AI companion, TonePilot makes it emotionally resonant and personality-aware.

**✨ Free, Open Source, and Built for Multi-LLM Integration.**

---

## 🚀 Features

- 🎯 **Emotion Detection** – Detect nuanced tones like *anxious*, *playful*, *confused* using transformer-based models
- 🧠 **Emotionally Matched Responses** – Generate emotionally resonant replies using Gemini or Hugging Face
- 🧩 **Pluggable Architecture** – Easily switch between LLMs or swap in your own tone classifiers
- 💡 **Prompt-Only OR Full Response Mode** – Get just the enhanced prompt, or the complete reply
- 🛠️ **CLI + Python API** – Use it in scripts, backends, or with your own chat interface

---

## 📦 Installation


```bash
pip install tonepilot
```

## Quick Start

### Basic Usage (Prompt Generation Only)

```bash
# Generate an emotionally-aware prompt without API keys
tonepilot "I'm feeling overwhelmed with work deadlines"
```

**Output:**
```
📝 Input: I'm feeling overwhelmed with work deadlines

🏷️  Detected tags:
  - stressed: 0.456
  - anxious: 0.234

⚖️  Response tags and weights:
  - calming_supporter: 0.445
  - practical_helper: 0.289

🔍 Final prompt:
Respond with calm reassurance and practical guidance. Help organize thoughts and provide actionable steps.

User: I'm feeling overwhelmed with work deadlines
Assistant: (Aim to respond in about 89 words)
```

### Full Response Generation

```bash
# Generate complete responses (requires API key)
tonepilot "I'm excited about my new job!" --mode gemini --respond true
```

## Environment Setup

### For Response Generation (Optional)

If you want to generate actual responses (not just prompts), set up API credentials:

**Option 1: Environment Variable**
```bash
export GOOGLE_API_KEY=your_api_key_here
```

**Option 2: .env File**
```bash
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

Get your API key from: [Google AI Studio](https://makersuite.google.com/app/apikey)

## CLI Usage

### Basic Commands

```bash
# Default: HuggingFace mode, prompt only
tonepilot "Your text here"

# Generate full response with Gemini
tonepilot "Your text here" --mode gemini --respond true

# Generate full response with HuggingFace
tonepilot "Your text here" --mode hf --respond true

# Different boolean formats accepted
tonepilot "Text" --respond yes
tonepilot "Text" --respond 1 
tonepilot "Text" --respond false
```

### Available Options

- `--mode {hf,gemini}`: Choose the response generation model (default: hf)
- `--respond {true,false,yes,no,1,0}`: Generate response or just prompt (default: false)

## Python API

```python
from tonepilot.core.tonepilot import TonePilotEngine

# Initialize engine
engine = TonePilotEngine(mode='hf', respond=False)

# Process text
result = engine.run("I'm nervous about my presentation tomorrow")

print("Detected emotions:", result['input_tags'])
print("Response emotions:", result['response_tags'])
print("Generated prompt:", result['final_prompt'])

# For response generation (requires API key)
engine_with_response = TonePilotEngine(mode='gemini', respond=True)
result = engine_with_response.run("I'm nervous about my presentation tomorrow")
print("Generated response:", result['response_text'])
```

## Architecture

TonePilot uses a sophisticated multi-stage pipeline:

1. **Emotion Detection**: Zero-shot classification using BART-large-MNLI
2. **Tone Mapping**: BERT-based classifier maps input emotions to response personalities  
3. **Prompt Blending**: Combines personality traits with weighted importance
4. **Response Generation**: Optional text generation using HuggingFace or Gemini models

### Supported Emotions

**Input Emotions**: curious, angry, sad, excited, confused, hopeful, tired, scared, playful, assertive

**Response Personalities**: empathetic_listener, direct_ally, calming_supporter, practical_helper, and more

## Model Downloads

TonePilot uses a custom-trained BERT classifier for tone mapping. **The model is automatically downloaded on first use** - no manual installation required!

### BERT Tone Classifier

- **File**: `tonepilot_bert_classifier.pt` (475 MB)
- **Download**: [GitHub Releases](https://github.com/sdurgi/tonepilot/releases/download/v0.1.0/tonepilot_bert_classifier.pt)
- **Purpose**: Maps detected emotions to appropriate response personalities
- **Training**: Custom-trained on emotional response datasets

### Automatic Model Management

TonePilot automatically handles model downloads and caching:

1. **First Run**: Downloads model to `~/.cache/tonepilot/` (one-time, ~475 MB)
2. **Subsequent Runs**: Uses cached model for instant loading
3. **Fallback Locations**: Also checks current directory and package directory

**Manual Download** (if needed):
```bash
# Only needed if automatic download fails
wget https://github.com/sdurgi/tonepilot/releases/download/v0.1.0/tonepilot_bert_classifier.pt -P ~/.cache/tonepilot/

# Or place in current directory
curl -L -o tonepilot_bert_classifier.pt https://github.com/sdurgi/tonepilot/releases/download/v0.1.0/tonepilot_bert_classifier.pt
```

**Note**: Internet connection required only on first use for model download.

## Examples

### Different Emotional Contexts

```bash
# Sadness → Empathetic support
tonepilot "My dog passed away yesterday"

# Excitement → Enthusiastic encouragement  
tonepilot "I just got accepted to my dream university!"

# Confusion → Clear guidance
tonepilot "I don't understand this math problem at all"

# Anger → Calming and validation
tonepilot "I'm so frustrated with this broken software!"
```

### Integration Examples

**Customer Support Bot:**
```python
def handle_customer_message(message):
    engine = TonePilotEngine(mode='gemini', respond=True)
    result = engine.run(message)
    return result['response_text']
```

**Content Writing Assistant:**
```python
def get_writing_prompt(topic, desired_tone):
    engine = TonePilotEngine(respond=False)
    result = engine.run(f"Write about {topic} with a {desired_tone} tone")
    return result['final_prompt']
```

## Requirements

- Python 3.8+
- PyTorch (automatically installed)
- Transformers library (automatically installed)
- Internet connection for model downloads on first use

**Optional for response generation:**
- Google API key (for Gemini mode)

## Development

```bash
# Clone repository
git clone https://github.com/sdurgi/tonepilot.git
cd tonepilot

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black tonepilot/
```

## Performance

- **Emotion Detection**: ~50ms on CPU
- **Response Generation**: 1-3 seconds (depending on model and length)
- **Memory Usage**: ~500MB (includes cached models)
- **Model Downloads**: ~1GB on first run (cached locally)

## Troubleshooting

**Import Errors**: Ensure all dependencies are installed with `pip install tonepilot`

**API Key Issues**: Verify your `.env` file or environment variables are set correctly

**Model Download Failures**: Check internet connection; models download automatically on first use

**Memory Issues**: Use smaller models or increase available memory

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{tonepilot2024,
  title={TonePilot: Emotional Intelligence for Text Generation},
  author={Durgi, Srivani},
  year={2024},
  url={https://github.com/sdurgi/tonepilot}
}
```

## Support

- 📧 Email: sdurgi21@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/sdurgi/tonepilot/issues)

---

**Made with ❤️ for building emotionally intelligent AI systems**