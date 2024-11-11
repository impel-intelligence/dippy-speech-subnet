import random

import numpy as np

from scoring.common import EVALUATION_DATASET_SAMPLE_SIZE, MAX_GENERATION_LENGTH, MAX_SEQ_LEN
from scoring.dataset import StreamedSyntheticDataset
from scoring.entrypoint import write_to_json
from scoring.scoring_logic.logic import scoring_workflow


def load_dataset():

    print("Sampling dataset")
    try:
        dataset = StreamedSyntheticDataset(
            max_input_len=MAX_SEQ_LEN - MAX_GENERATION_LENGTH - 200,
            mock=True,
        )
        sampled_data = dataset.sample_dataset(EVALUATION_DATASET_SAMPLE_SIZE, dummy=True)
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


def apply_weights(base_score, text, weights):
    """
    Apply multiple weighting mechanisms to adjust the base score.
    
    :param base_score: The initial score before weighting.
    :param text: The text input used to calculate length-based weights.
    :param voice_description: The voice description to apply specific weights.
    :param weights: A dictionary of weights to apply.
    :return: Weighted score.
    """
    weighted_score = base_score
    
    # Apply text length weight if provided
    if isinstance(weights, dict) and "text_length" in weights:
        length_factor = len(text) / 100  # Normalize text length by dividing by 100
        weighted_score *= weights["text_length"] * length_factor
    
    # Example to add other weights
    # # Apply voice description weight if provided
    # if "voice_description" in weights:
    #     description_factor = len(voice_description) / 50  # Normalize by dividing by 50 as an example
    #     weighted_score *= weights["voice_description"] * description_factor
    
    return weighted_score


def get_tts_score(request: str):
    """
    Calculate and return the TTS scores with optional weighting.

    :param request: The request object containing necessary details for scoring.
    :return: A list of weighted scores.
    """

    # Define the weight dictionary; can be extended with additional weights.
    weights = {'text_length': 1} 

    data = load_dataset()

    result = []

    for text, last_user_message, voice_description in data:
        # Calculate the base score using the provided scoring workflow
        base_score = scoring_workflow(request.repo_namespace, request.repo_name, text, voice_description)

        # Apply weights if provided
        weighted_score = apply_weights(base_score, text, weights )

        result.append(weighted_score)
    
    # Flatten the outputs to extract numerical values
    flattened_scores = [score.item() for score in result]  # Extract single value from each array

    # Normalize the scores to a 0-1 range
    min_score = min(flattened_scores)
    max_score = max(flattened_scores)

     # Handle the edge case where all scores are the same
    if min_score == max_score:
        normalized_scores = [1.0 for _ in flattened_scores]  # Set to max of range if all scores are the same
    else:
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in flattened_scores]

    # Aggregate the normalized scores (e.g., using the average)
    average_normalized_score = np.mean(normalized_scores)

    # Scale to a 1-10 range
    final_score = 1 + 9 * average_normalized_score  # Scale normalized average to 1-10


    import pdb;

    pdb.set_trace()
    # final_score = 0
    # result = {"final_score": final_score}
    # try:
    #     final_score = random.random()
    #     result["final_score"] = final_score
    # except Exception as e:
    #     print(f"error calculating score: {e}")
    #     result["error"] = str(e)

    return result
    # write_to_json(result, "/tmp/tts_output.json")

    # start_time = time.time()  # Start timing
    human_similarity_score = scoring_workflow_lda(repo_namespace, repo_name, text, voice_description)
    # end_time = time.time()  # End timing

    # duration = end_time - start_time
    # print(f"Scoring took {duration:.2f} seconds")
    # print(f"Score: {human_similarity_score[0][0]}")

    # result={"human_similarity_score": human_similarity_score[0][0]}
