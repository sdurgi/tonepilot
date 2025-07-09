# TonePilot

TonePilot is an emotional intelligence system for text generation that adapts its responses based on detected emotional tones.

## Features

- Emotional tone detection using advanced ML models
- Adaptive response generation with personality blending
- Support for both Gemini and HuggingFace models
- Command-line interface for easy interaction

## Installation

```bash
pip install tonepilot
```

## Quick Start

1. For Gemini mode (recommended):
   ```bash
   # Get your API key from https://makersuite.google.com/app/apikey
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   
   # Run TonePilot
   tonepilot "Hello, I'm feeling excited about my new job!" --mode gemini
   ```

2. For HuggingFace mode (no API key needed):
   ```bash
   tonepilot "Hello, I'm feeling excited about my new job!" --mode hf
   ```

## Example Output

```
📝 Input: Hello, I'm feeling excited about my new job!

🏷️ Detected tags:
  - excited: 0.900
  - hopeful: 0.700

⚖️ Response tags and weights:
  - supportive: 0.337
  - enthusiastic: 0.215

💬 Response: That's fantastic news! A new job is such an exciting milestone...
```

## Requirements

- Python 3.8 or higher
- For Gemini mode: Google Gemini API key
- Internet connection for model downloads

## License

This project is licensed under the MIT License - see the LICENSE file for details. 