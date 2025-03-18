#!/usr/bin/env python3
"""
Test script for evaluating TTS models using our scoring system.

This script replicates the functionality of get_tts_score by:
1. Loading the synthetic dataset
2. Processing each sample with a Parler TTS model
3. Analyzing emotion and transcription accuracy
4. Calculating combined scores based on emotion and WER

Requirements:
    - HUME_API_KEY environment variable
    - WHISPER_ENDPOINT environment variable
    - HUGGINGFACE_TOKEN_PRIME environment variable

Usage:
    python test_elevenlabs.py --repo-namespace <namespace> --repo-name <name>
"""

# Standard library imports
import argparse
import json
import logging
import os
import numpy as np
import torch
from typing import Dict, NamedTuple

# Environment setup
from dotenv import load_dotenv
load_dotenv(override=True)

# Local application imports
from scoring.scoring_logic.logic import scoring_workflow
from scoring.get_tts_score import apply_weights, load_dataset, load_parler_model

# Set up logger
logger = logging.getLogger(__name__)

class Request(NamedTuple):
    """Simple class to mimic the request object expected by get_tts_score"""
    repo_namespace: str
    repo_name: str

def get_tts_score(request: Request) -> dict:
    """
    Calculate and return the TTS scores with optional weighting.

    :param request: The request object containing necessary details for scoring.
    :return: A dictionary with the final score and any errors.
    """
    # Load the dataset containing input data for scoring
    data = load_dataset()
    # Initialize an empty list to store scores for each processed text
    scores = []
    # Initialize the result dictionary with a default final score of 0
    result = {"final_score": 0}

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing TTS scoring")

    model, tokenizer = load_parler_model(request.repo_namespace, request.repo_name, device)

    # Iterate over the data
    for i, item in enumerate(data):
        try:
            text = item["target_text"]
            last_user_message = item["last_user_message"]
            voice_description = item["voice_description"]
            character_profile = item["character_profile"]
            emotional_text = item["emotional_text"][0]
            
            logger.info(f"Processing sample {i}: {text[:50]}...")
            
            # Detect the emotion in the audio sample and carry out word error rate analysis
            raw_emotion_score, wer_score = scoring_workflow(
                emotional_text, 
                voice_description, 
                device, 
                model, 
                tokenizer
            )

            expected_emotion = character_profile["selected_emotion"]
            detected_emotion_score = raw_emotion_score[expected_emotion]

            logger.info(f"Expected Emotion: {expected_emotion} - Detected Score: {detected_emotion_score}")

            # Apply weights to the base score based on text properties
            weighted_score = apply_weights(detected_emotion_score, wer_score)

            # Append the weighted score to the scores list
            scores.append(weighted_score)
            
            logger.info(f"Sample {i} score: {weighted_score:.3f}")

        # Catch any exceptions that occur during score calculation
        except Exception as e:
            # Log the error and update the result dictionary with the error message
            logger.error(f"Error calculating score for sample {i}", exc_info=True)
            result["error"] = str(e)

    # Calculate the average score after the loop completes
    if scores:
        clipped_scores = np.clip(scores, 0, 1)
        mean_value = float(np.mean(clipped_scores))
        result["final_score"] = mean_value

    # Return the final result dictionary containing the score and any errors
    return result

def main():
    """
    Command-line interface for the TTS evaluator.
    
    Accepts repository namespace and name arguments.
    Runs the evaluation and prints results to console.
    
    Example:
        python test_elevenlabs.py --repo-namespace <namespace> --repo-name <name>
    """
    parser = argparse.ArgumentParser(description='Evaluate TTS model')
    parser.add_argument('--repo-namespace', required=True, help='Repository namespace')
    parser.add_argument('--repo-name', required=True, help='Repository name')
    parser.add_argument('--output-file', help='Path to save results JSON (optional)')
    
    args = parser.parse_args()
    
    try:
        print("\nEvaluating TTS model...")
        print("-" * 50)
        
        request = Request(repo_namespace=args.repo_namespace, repo_name=args.repo_name)
        result = get_tts_score(request)
        
        print("\nResults:")
        print("-" * 50)
        print(f"Final Score: {result['final_score']:.3f}")
        
        if "error" in result:
            print(f"Error: {result['error']}")
        
        # Save results to file if specified
        if args.output_file:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
            with open(args.output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to: {args.output_file}")
            
        print("-" * 50)
                
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
