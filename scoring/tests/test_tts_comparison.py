import logging
import numpy as np
import torch
from typing import Dict

from scoring.dataset import StreamedSyntheticDataset
from scoring.scoring_logic.logic import scoring_workflow
from scoring.get_tts_score import load_parler_model, apply_weights, load_dataset
from scoring.common import MAX_SEQ_LEN, MAX_GENERATION_LENGTH

logger = logging.getLogger(__name__)

def evaluate_tts(audio_path: str, repo_namespace: str, repo_name: str) -> Dict:
    """
    Evaluate a TTS model using our scoring logic and synthetic dataset.
    
    Args:
        audio_path: Path to save generated audio for testing
        repo_namespace: HuggingFace repo namespace
        repo_name: HuggingFace repo name
        
    Returns:
        Dict containing the final score and any errors
    """
    # Initialize result dictionary
    result = {"final_score": 0}
    scores = []
    
    try:
        # Load dataset
        data = load_dataset()
        
        # Setup device and model
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info("Initializing TTS scoring")
        model, tokenizer = load_parler_model(repo_namespace, repo_name, device)
        
        # Process each sample
        for item in data:
            text = item["target_text"]
            voice_description = item["voice_description"]
            character_profile = item["character_profile"]
            emotional_text = item["emotional_text"][0]
            
            try:
                # Get emotion and WER scores
                raw_emotion_score, wer_score = scoring_workflow(
                    emotional_text, 
                    voice_description,
                    device,
                    model,
                    tokenizer
                )
                
                # Get expected emotion score
                expected_emotion = character_profile["selected_emotion"]
                detected_emotion_score = raw_emotion_score[expected_emotion]
                
                logger.info(f"Expected Emotion: {expected_emotion} - Detected Score: {detected_emotion_score}")
                
                # Calculate weighted score
                weighted_score = apply_weights(detected_emotion_score, wer_score)
                scores.append(weighted_score)
                
            except Exception as e:
                logger.error("Error calculating score", exc_info=True)
                result["error"] = str(e)
        
        # Calculate final score
        if scores:
            clipped_scores = np.clip(scores, 0, 1)
            mean_value = float(np.mean(clipped_scores))
            result["final_score"] = mean_value
            
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        result["error"] = str(e)
        
    return result

def main():
    """Example usage of the TTS evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a TTS model')
    parser.add_argument('--audio-dir', help='Directory to save generated audio', required=True)
    parser.add_argument('--repo-namespace', help='HuggingFace repo namespace', required=True)
    parser.add_argument('--repo-name', help='HuggingFace repo name', required=True)
    
    args = parser.parse_args()
    
    try:
        result = evaluate_tts(args.audio_dir, args.repo_namespace, args.repo_name)
        
        print("\nTTS Evaluation Results:")
        print("-" * 50)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Final Score: {result['final_score']:.3f}")
        print("-" * 50)
                
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
