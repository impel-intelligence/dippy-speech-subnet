import random
import logging

import numpy as np

from scoring.common import EVALUATION_DATASET_SAMPLE_SIZE, MAX_GENERATION_LENGTH, MAX_SEQ_LEN
from scoring.dataset import StreamedSyntheticDataset
from scoring.scoring_logic.logic import scoring_workflow

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


def apply_weights(base_score: float, wer: float) -> float:
    """
    Adjust the base score by incorporating the Word Error Rate (WER).

    :param base_score: The initial score before weighting.
    :param wer: The Word Error Rate (WER) of the transcription.
    :return: The weighted score, with base_score and WER equally weighted (50% each).
    """
    base_score = base_score[0]

    # Handle WER weighting
    if wer == 0.0:  # Perfect transcription
        wer_weighted_score = 1.0  # Perfect score for WER
    else:
        wer_weighted_score = 1.0 - wer  # Scale WER score (lower WER is better)

    # Combine base score and WER score with 50% weight each
    weighted_score = 0.5 * base_score + 0.5 * wer_weighted_score

    return weighted_score


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

    # Iterate over the data, which contains tuples of text, last user message, and voice description.
    for text, last_user_message, voice_description in data:
        try:
            # Calculate the base score and wer using the scoring workflow function.
            base_score, wer_score = scoring_workflow(request.repo_namespace, request.repo_name, text, voice_description)

            # Extract float values from each tensor in the 'scores' list for further processing
            float_values_from_tensors = [score.item() for score in base_score]

            # Apply weights to the base score based on text properties.
            weighted_score = apply_weights(float_values_from_tensors, wer_score)

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
