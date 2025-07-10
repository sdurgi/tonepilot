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

def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_environment(mode: str, respond: bool):
    """Check if required environment variables are set."""
    if respond and mode == "gemini" and not os.getenv("GOOGLE_API_KEY"):
        print("\n❌ Error: GOOGLE_API_KEY environment variable is not set!")
        print("\nTo use Gemini mode with response generation, you need to:")
        print("1. Get your API key from: https://makersuite.google.com/app/apikey")
        print("2. Set it up in one of these ways:")
        print("\n   Option 1 - Create .env file:")
        print('   echo "GOOGLE_API_KEY=your_api_key_here" > .env')
        print("\n   Option 2 - Set environment variable:")
        print("   export GOOGLE_API_KEY=your_api_key_here")
        print("\n   Option 3 - Add to shell profile (permanent):")
        print("   echo 'export GOOGLE_API_KEY=your_api_key_here' >> ~/.zshrc")
        print("   source ~/.zshrc")
        print("\nOr use --mode hf to use HuggingFace models instead.\n")
        sys.exit(1)

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="TonePilot: Emotional Intelligence for Text Generation")
    parser.add_argument("input_text", help="The text to process")
    parser.add_argument("--mode", choices=["hf", "gemini"], default="hf",
                      help="Model to use for response generation (default: hf)")
    parser.add_argument("--respond", type=str2bool, nargs='?', const=True, default=False,
                      help="Generate a response (default: False, only show prompt). Accepts true/false, yes/no, 1/0")
    
    args = parser.parse_args()
    
    # Check environment before proceeding
    check_environment(args.mode, args.respond)
    
    try:
        # Initialize engine
        engine = TonePilotEngine(mode=args.mode, respond=args.respond)
        
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
        print("\n🔍 Final prompt:")
        print(result["final_prompt"])
        
        if args.respond:
            print("\n💬 Response:", result["response_text"])
            print(f"📊 Target length: {result['response_length']} tokens")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()