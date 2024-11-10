


import random

from scoring.common import EVALUATION_DATASET_SAMPLE_SIZE, MAX_GENERATION_LENGTH, MAX_SEQ_LEN
from scoring.dataset import StreamedSyntheticDataset


from scoring.entrypoint import write_to_json



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


def get_tts_score(
    repo_name: str ,
    repo_namespace: str,
):
    
    final_score = 0
    result = {"final_score": final_score}
    try:
        final_score = random.random()
        result["final_score"] = final_score
    except Exception as e:
        print(f"error calculating score: {e}")
        result["error"] = str(e)
        
    write_to_json(result, "/tmp/tts_output.json")
    
    # start_time = time.time()  # Start timing
    # human_similarity_score = scoring_workflow_lda(repo_namespace, repo_name, text, voice_description)
    # end_time = time.time()  # End timing

    # duration = end_time - start_time
    # print(f"Scoring took {duration:.2f} seconds")
    # print(f"Score: {human_similarity_score[0][0]}")

    # result={"human_similarity_score": human_similarity_score[0][0]}

