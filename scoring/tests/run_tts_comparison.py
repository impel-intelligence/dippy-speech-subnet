#!/usr/bin/env python3
"""
Script to run TTS evaluation using synthetic dataset.
"""

import os
from test_tts_comparison import evaluate_tts

def main():
    print("TTS Model Evaluation Tool")
    print("-" * 50)
    print("This tool evaluates TTS models using our synthetic dataset")
    print("Requires: HUME_API_KEY and WHISPER_ENDPOINT environment variables")
    print("-" * 50)
    
    # Get inputs
    audio_dir = input("Enter directory to save generated audio: ")
    repo_namespace = input("Enter HuggingFace repo namespace: ")
    repo_name = input("Enter HuggingFace repo name: ")
    
    # Ensure audio directory exists
    os.makedirs(audio_dir, exist_ok=True)
    
    try:
        # Run evaluation
        result = evaluate_tts(audio_dir, repo_namespace, repo_name)
        
        # Print results
        print("\nResults:")
        print("-" * 50)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Final Score: {result['final_score']:.3f}")
            print("(Score is average of emotion matching and WER across synthetic dataset)")
        print("-" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()
