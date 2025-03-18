import random
import logging

import numpy as np

from scoring.common import EVALUATION_DATASET_SAMPLE_SIZE, MAX_GENERATION_LENGTH, MAX_SEQ_LEN
from scoring.dataset import StreamedSyntheticDataset
from scoring.scoring_logic.logic import scoring_workflow

import torch
import torch.nn as nn
import torch.distributed as dist

from parler_tts import ParlerTTSForConditionalGeneration
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from transformers import AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor
from parler_tts import ParlerTTSConfig, ParlerTTSForConditionalGeneration, build_delay_pattern_mask
from transformers import AutoFeatureExtractor, AutoTokenizer

logger = logging.getLogger(__name__)  # Create a logger for this module


def load_dataset():
    
    NUMBER_OF_SAMPLES = 2


    print("Sampling dataset")
    try:
        dataset = StreamedSyntheticDataset(
            max_input_len=MAX_SEQ_LEN - MAX_GENERATION_LENGTH - 200,
            mock=True,
        )
        
        #sampled_data = dataset.sample_dataset(EVALUATION_DATASET_SAMPLE_SIZE, dummy=True)
        #sampled_data = dataset.__getitem__(5)
        sampled_data = []
        for _ in range(NUMBER_OF_SAMPLES):

            _CURRENT_EVALUATION_DATASET_SAMPLE_SIZE = len(dataset.dataset)

            random_index = random.randint(0, _CURRENT_EVALUATION_DATASET_SAMPLE_SIZE  - 1)  # Generate a random index

            sample = dataset.__getitem__(random_index)  # Get the sample using __getitem__

            response, query, description = sample
    
            # Append the sample tuple to the list
            sampled_data.append((response, query, description))

    
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



def apply_weights(cross_entropy_loss: float, wer: float, k: float = 2.0) -> float:
    """
    Compute a weighted score by normalizing the cross-entropy loss and Word Error Rate (WER),
    ensuring both are mapped to a 0-1 range where higher is better.

    :param cross_entropy_loss: The cross-entropy loss (0 to infinity). Lower values indicate better performance.
                               It is transformed using 1 / (1 + k * cross_entropy_loss) to control steepness.
                               - Cross-entropy loss = 0 → Normalized score = 1 (Best case)
                               - Cross-entropy loss → ∞ → Normalized score → 0 (Worst case)
                               - Cross-entropy loss ~ 1 → Normalized score ~ 1 / (1 + k) (adjustable steepness)
    :param wer: The Word Error Rate (WER) of the transcription (0 to 1). Lower values indicate better accuracy.
                It is transformed using (1 - wer) to align with the scoring direction.
    :param k: Scaling factor to adjust the steepness of the cross-entropy loss normalization.
              - **Big ( k ) (e.g., 5)** → Punishes high loss HARD. Score drops fast.  
              - **Small ( k ) (e.g., 0.5)** → Loss is forgiven more. Score drops slowly.  
              - **( k = 1.0 ) (default)** → Balanced approach between smooth and steep drop-off.  

              🔹 **Think of ( k ) like a sensitivity dial:**  
                 - Turn it **up** → Harsher punishment for bad performance.  
                 - Turn it **down** → More forgiving, smoother scoring.  

    :return: A weighted score, where the normalized cross-entropy score and WER contribute equally (50% each).
             The final score ranges from 0 to 1, where higher values indicate better performance.
    """

    # Normalize cross-entropy loss with adjustable steepness
    cross_entropy_loss_normalized = 1 / (1 + k * cross_entropy_loss)  # Adjustable steepness

    # Normalize WER (lower WER → higher score)
    wer_weighted_score = 1.0 - wer  

    # Combine both scores with equal weights
    weighted_score = 0.6 * cross_entropy_loss_normalized + 0.4 * wer_weighted_score

    return weighted_score




def load_parler_model(repo_namespace, repo_name, device):
    model_name = f"{repo_namespace}/{repo_name}"
    try:
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Parler TTS model '{model_name}' and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load Parler TTS model or tokenizer: {e}", exc_info=True)
        raise RuntimeError(f"Parler TTS model or tokenizer loading failed : {e}")

def load_emotion():
    try:
        inference_pipeline = pipeline(task=Tasks.emotion_recognition, model="iic/emotion2vec_plus_large")
        logger.info("Emotion2Vector Model initialized successfully.")
        return inference_pipeline
    except Exception as e:
        logger.error(f"Failed to process audio for Emotion2Vector: {e}", exc_info=True)
        raise RuntimeError("Emotion2Vector processing failed.")


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
    # # Define the weights for scoring, with 'text_length' having a default weight of 1.
    # weights = {"text_length": 1}

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device selected for computation: {device}")

    # model, tokenizer = load_parler_model(request.repo_namespace, request.repo_name, device)

    # emotion_inference_pipeline = load_emotion()
    config = ParlerTTSConfig.from_pretrained(f"{request.repo_namespace}/{request.repo_name}")
    model = ParlerTTSForConditionalGeneration.from_pretrained(f"{request.repo_namespace}/{request.repo_name}", config=config).to(device).eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(f"{request.repo_namespace}/{request.repo_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"{request.repo_namespace}/{request.repo_name}")

    # Iterate over the data, which contains tuples of text, last user message, and voice description.
    for text, last_user_message, voice_description in data:
        try:
            # Calculate the base score and wer using the scoring workflow function.
            base_score, wer_score = scoring_workflow(request.repo_namespace, request.repo_name, text, voice_description, device, model, tokenizer, config, feature_extractor)

            # Extract float values from each tensor in the 'scores' list for further processing
            #float_values_from_tensors = [score.item() for score in base_score]

            # Apply weights to the base score based on text properties.
            weighted_score = apply_weights(base_score, wer_score)

            # Append the weighted score to the scores list.
            scores.append(weighted_score)

        # Catch any exceptions that occur during score calculation.
        except Exception as e:
            # Log the error and update the result dictionary with the error message.
            logging.info(f"Error calculating score: {e}")
            result["error"] = str(e)

    # Calculate the average score after the loop completes.
    if scores:
        clipped_scores = np.clip(scores, 0, 1)
        mean_value = float(np.mean(clipped_scores))
        result["final_score"] = mean_value

    # Return the final result dictionary containing the score and any errors.
    return result
