"""
Command-line interface for TonePilot.

This module provides the CLI entry point for the TonePilot system.
"""

import os
import sys
import argparse
from tonepilot.core.tonepilot import TonePilotEngine
from dotenv import load_dotenv

load_dotenv()

def check_environment():
    """Check if required environment variables are set."""
    if not os.getenv("GEMINI_API_KEY") and len(sys.argv) > 1 and "--mode" in sys.argv and "gemini" in sys.argv:
        print("\n❌ Error: GEMINI_API_KEY environment variable is not set!")
        print("\nTo use Gemini mode, you need to:")
        print("1. Get your API key from: https://makersuite.google.com/app/apikey")
        print("2. Set it up in one of these ways:")
        print("\n   Option 1 - Create .env file:")
        print('   echo "GEMINI_API_KEY=your_api_key_here" > .env')
        print("\n   Option 2 - Set environment variable:")
        print("   export GEMINI_API_KEY=your_api_key_here")
        print("\n   Option 3 - Add to shell profile (permanent):")
        print("   echo 'export GEMINI_API_KEY=your_api_key_here' >> ~/.zshrc")
        print("   source ~/.zshrc")
        print("\nOr use --mode hf to use HuggingFace models instead.\n")
        sys.exit(1)

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="TonePilot: Emotional Intelligence for Text Generation")
    parser.add_argument("input_text", help="The text to process")
    parser.add_argument("--mode", choices=["hf", "gemini"], default="hf",
                      help="Model to use (default: hf)")
    
    args = parser.parse_args()
    
    # Check environment before proceeding
    check_environment()
    
    try:
        # Initialize engine
        engine = TonePilotEngine(mode=args.mode)
        
        # Process text
        result = engine.run(args.input_text)
        
        # Print results in a logical order
        print("\n📝 Input:", args.input_text)
        
        print("\n🏷️  Detected tags:")
        for tag, score in result["input_tags"].items():
            print(f"  - {tag}: {score:.3f}")
            
        print("\n⚖️  Response tags and weights:")
        for tag, weight in result["response_weights"].items():
            print(f"  - {tag}: {weight:.3f}")
            
        # Extract and display the personality part of the prompt
        prompt_parts = result["final_prompt"]
        print("\n🔍 Final prompt:")
        print(prompt_parts)  # Just the personality part
        # print("\nUser:", result["input_text"])
        # print("AI:")
        
        print("\n💬 Response:", result["response_text"])
        print(f"📊 Target length: {result['response_length']} tokens")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()