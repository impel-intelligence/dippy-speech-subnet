import logging
import random
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from elevenlabs import ElevenLabs

from scoring.common import EVALUATION_DATASET_SAMPLE_SIZE, MAX_GENERATION_LENGTH, MAX_SEQ_LEN
from scoring.dataset import StreamedSyntheticDataset
from scoring.benchmark.elevenlabs_logic import scoring_workflow_elevenlabs

logger = logging.getLogger(__name__)  # Create a logger for this module


def load_dataset():

    NUMBER_OF_SAMPLES = 10

    print("Sampling dataset")
    try:
        dataset = StreamedSyntheticDataset(
            max_input_len=MAX_SEQ_LEN - MAX_GENERATION_LENGTH - 200,
            mock=True,
        )

        sampled_data = []
        for _ in range(NUMBER_OF_SAMPLES):

            _CURRENT_EVALUATION_DATASET_SAMPLE_SIZE = len(dataset.dataset)

            random_index = random.randint(0, _CURRENT_EVALUATION_DATASET_SAMPLE_SIZE - 1)  # Generate a random index

            sample = dataset.__getitem__(random_index)  # Get the sample using __getitem__

            # response, query, description, character_profile = sample
            response = sample['target_text']
            query = sample["last_user_message"]
            voice_description = sample["voice_description"]
            character_profile = sample['character_profile']
            emotional_text = sample["emotional_text"]
            # Append the sample dictionary to the list
            sampled_data.append({
                "target_text": response,
                "last_user_message": query,
                "voice_description": voice_description,
                "character_profile": character_profile,
                "emotional_text": emotional_text
            })

        """
        shape of sampled_data: a list structured like the following:
        [
                (
                    # target text
                    "I understand your concern about the project timeline. Let me assure you that we'll meet our deadlines by prioritizing key deliverables and maintaining clear communication throughout the process.",
                    # last user message
                    "I'm worried we won't finish the project on time. What can we do?",
                    # voice_description
                    "Eric's voice is deep and resonant, providing a commanding presence in his speech. A **male voice with an American accent**, his **very low-pitched voice** is captured with crisp clarity. His tone is **authoritative yet slightly monotone**, emphasizing his strong and steady delivery."
                ),
                (
                    "The latest market analysis shows promising growth opportunities in the Asia-Pacific region. Our data indicates a 15% increase in consumer demand for our products.",
                    "What are the key findings from the market research?",
                    "Laura's voice is warm and friendly, conveying messages with clarity and enthusiasm. A **female voice with an American accent** enunciates every word with precision. Her voice is **very close-sounding**, and the recording is excellent, capturing her **slightly high-pitched voice** with crisp clarity. Her tone is **very expressive and animated**, bringing energy to her delivery."
                ),
                (
                    "To improve your Python code performance, consider using list comprehensions instead of loops, leverage built-in functions, and implement proper data structures.",
                    "How can I make my Python code run faster?",
                    "Patrick's voice is authoritative and assertive, perfect for informative and instructional content. A **male voice with an American accent** enunciates every word with precision. His voice is **very close-sounding**, and the recording is excellent, capturing his **fairly low-pitched voice** with crisp clarity. His tone is **slightly monotone**, emphasizing clarity and directness."
                )
            ]
        """
        return sampled_data
    except Exception as e:
        failure_reason = str(e)
        raise Exception(f"Error loading dataset: {failure_reason}")


def apply_weights(base_score: float, wer: float) -> float:
    """
    Adjust the base score by incorporating the Word Error Rate (WER).

    :param base_score: The initial score before weighting.
    :param wer: The Word Error Rate (WER) of the transcription.
    :return: The weighted score, with base_score and WER equally weighted (50% each).
    """
    # Handle WER weighting
    if wer == 0.0:  # Perfect transcription
        wer_weighted_score = 1.0  # Perfect score for WER
    else:
        wer_weighted_score = 1.0 - wer  # Scale WER score (lower WER is better)

    # Combine base score and WER score with 50% weight each
    weighted_score = 0.5 * base_score + 0.5 * wer_weighted_score

    return weighted_score


def initialize_elevenlabs_client(api_key=None):
    """
    Initialize the ElevenLabs client with the provided API key.
    
    :param api_key: ElevenLabs API key. If None, tries to get from environment.
    :return: List of available voices
    """
    if api_key is None:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if api_key is None:
            raise ValueError("ELEVENLABS_API_KEY is not set in the environment.")
    
    try:
        # Initialize the ElevenLabs client with the API key
        client = ElevenLabs(api_key=api_key)
        # Get available voices using the client.voices.get_all() method
        voices_response = client.voices.get_all()
        # The voices are in the 'voices' attribute of the response
        available_voices = voices_response.voices
        logger.info(f"ElevenLabs client initialized successfully. Found {len(available_voices)} voices.")
        return available_voices
    except Exception as e:
        logger.error("Failed to initialize ElevenLabs client", exc_info=True)
        raise RuntimeError(f"ElevenLabs initialization failed: {e}")


def get_tts_score(request: str) -> dict:
    """
    Calculate and return the TTS scores with optional weighting.

    :param request: The request object containing necessary details for scoring.
    :return: A dictionary with the final score and any errors.
    """

    # Load the dataset containing input data for scoring.
    data = load_dataset()
    # Initialize an empty list to store scores for each processed text.
    scores = []
    # Initialize the result dictionary with a default final score of 0.
    result = {"final_score": 0}

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing TTS scoring")

    # Initialize ElevenLabs client
    available_voices = initialize_elevenlabs_client()
    import pandas as pd
    # Iterate over the data, which contains tuples of text, last user message, and voice description.
    for item in data:
        text = item["target_text"]
        last_user_message = item["last_user_message"]
        voice_description = item["voice_description"]
        character_profile = item["character_profile"]
        emotional_text = item["emotional_text"][0]
        try:
            # Detect the emotion in the audio sample and carry out word error rate analysis
            raw_emotion_score, wer_score = scoring_workflow_elevenlabs(emotional_text, voice_description, device)

            expected_emotion = character_profile["selected_emotion"]

            detected_emotion_score = raw_emotion_score[expected_emotion]

            logger.info(f"Expected Emotion: {expected_emotion} - Detected Score: {detected_emotion_score}")

            # Apply weights to the base score based on text properties.
            weighted_score = apply_weights(detected_emotion_score, wer_score)

            # Append the weighted score to the scores list.
            scores.append(weighted_score)

                # --- Prepare DataFrame and append to CSV ---
            df = pd.DataFrame([{
                "Expected_Emotion": expected_emotion,
                "Detected_Emotion_Score": detected_emotion_score,
                "WER_Score": wer_score,
                "Weighted_Score": weighted_score
            }])

            # Append row to CSV, write header if file doesn't exist
            df.to_csv(
                "scores.csv", 
                mode='a', 
                header=not pd.io.common.file_exists("final.csv"), 
                index=False,
                float_format='%.17g'  # Optional: format to preserve precision similar to your example
            )

        # Catch any exceptions that occur during score calculation.
        except Exception as e:
            # Log the error and update the result dictionary with the error message.
            logger.error("Error calculating score", exc_info=True)
            result["error"] = str(e)

    # Calculate the average score after the loop completes.
    if scores:
        clipped_scores = np.clip(scores, 0, 1)
        mean_value = float(np.mean(clipped_scores))
        result["final_score"] = mean_value

        df = pd.DataFrame([{"final_score": mean_value}])
        df.to_csv("final.csv", mode='a', header=not pd.io.common.file_exists("final.csv"), index=False)

    # Return the final result dictionary containing the score and any errors.
    return result
