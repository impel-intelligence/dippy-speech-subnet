import os
import tempfile
import uuid

import joblib
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

SPEAKERS = [
    {
        "name": "Will",
        "description": "Will's voice is confident and engaging, delivering speeches with a clear and steady tone. A **male voice with an American accent** enunciates every word with precision. His voice is **fairly close-sounding**, and the recording is excellent, capturing his voice with crisp clarity. His tone is **fairly expressive and animated**, conveying enthusiasm and passion.",
    },
    {
        "name": "Eric",
        "description": "Eric's voice is deep and resonant, providing a commanding presence in his speech. A **male voice with an American accent**, his **very low-pitched voice** is captured with crisp clarity. His tone is **authoritative yet slightly monotone**, emphasizing his strong and steady delivery.",
    },
    {
        "name": "Laura",
        "description": "Laura's voice is warm and friendly, conveying messages with clarity and enthusiasm. A **female voice with an American accent** enunciates every word with precision. Her voice is **very close-sounding**, and the recording is excellent, capturing her **slightly high-pitched voice** with crisp clarity. Her tone is **very expressive and animated**, bringing energy to her delivery.",
    },
    {
        "name": "Alisa",
        "description": "Alisa's voice is smooth and melodic, delivering content with grace and poise. A **female voice with an American accent**, her voice is **very close-sounding and clean**, and the recording is excellent, capturing her voice with crisp clarity. Her tone is **quite expressive and animated**, adding depth and emotion to her speech.",
    },
    {
        "name": "Patrick",
        "description": "Patrick's voice is authoritative and assertive, perfect for informative and instructional content. A **male voice with an American accent** enunciates every word with precision. His voice is **very close-sounding**, and the recording is excellent, capturing his **fairly low-pitched voice** with crisp clarity. His tone is **slightly monotone**, emphasizing clarity and directness.",
    },
    {
        "name": "Rose",
        "description": "Rose's voice is gentle and soothing, ideal for calm and reassuring messages. A **female voice with an American accent**, her voice is **very close-sounding**, and the recording is excellent, capturing her **smooth and calm voice** with crisp clarity. Her tone is **fairly expressive**, conveying warmth and comfort.",
    },
    {
        "name": "Jerry",
        "description": "Jerry's voice is lively and energetic, bringing excitement to his speeches. A **male voice with an American accent** enunciates every word with precision. His voice is **fairly close-sounding**, and the recording is excellent, capturing his **fairly high-pitched voice** with crisp clarity. His tone is **very expressive and animated**, injecting enthusiasm into his delivery.",
    },
    {
        "name": "Jordan",
        "description": "Jordan's voice is articulate and precise, ensuring every word is clearly understood. A **male voice with an American accent**, his voice is **very close-sounding**, and the recording is excellent, capturing his voice with crisp clarity. His tone is **slightly expressive and animated**, adding emphasis where needed.",
    },
    {
        "name": "Lauren",
        "description": "Lauren's voice is expressive and dynamic, adding emotion and depth to her delivery. A **female voice with an American accent** enunciates every word with precision. Her voice is **quite close-sounding and clean**, and the recording is excellent, capturing her voice with crisp clarity. Her tone is **very expressive and animated**.",
    },
    {
        "name": "Jenna",
        "description": "Jenna's voice is bright and cheerful, making her speeches uplifting and positive. A **female voice with an American accent**, her voice is **very close-sounding**, and the recording is excellent, capturing her **slightly high-pitched voice** with crisp clarity. Her tone is **very expressive and animated**, conveying joy and enthusiasm.",
    },
    {
        "name": "Karen",
        "description": "Karen's voice is strong and persuasive, effectively conveying compelling arguments. A **female voice with an American accent** enunciates every word with precision. Her voice is **very close-sounding**, and the recording is excellent, capturing her voice with crisp clarity. Her tone is **fairly expressive and assertive**.",
    },
    {
        "name": "Rick",
        "description": "Rick's voice is robust and powerful, providing impactful and memorable speeches. A **male voice with an American accent**, his voice is **very close-sounding**, and the recording is excellent, capturing his **low-pitched voice** with crisp clarity. His tone is **quite expressive and animated**, commanding attention.",
    },
    {
        "name": "Bill",
        "description": "Bill's voice is steady and measured, ideal for detailed and comprehensive explanations. A **male voice with an American accent** enunciates every word with precision. His voice is **fairly close-sounding**, and the recording is excellent, capturing his voice with crisp clarity. His tone is **slightly monotone**, emphasizing clarity.",
    },
    {
        "name": "James",
        "description": "James's voice is clear and concise, ensuring his messages are direct and to the point. A **male voice with an American accent**, his voice is **very close-sounding**, and the recording is excellent, capturing his voice with crisp clarity. His tone is **slightly expressive and animated**.",
    },
    {
        "name": "Yann",
        "description": "Yann's voice is smooth and calm, perfect for soothing and meditative content. A **male voice with an American accent**, his voice is **very close-sounding and clean**, and the recording is excellent, capturing his voice with crisp clarity. His tone is **very monotone**, providing a relaxing effect.",
    },
    {
        "name": "Emily",
        "description": "Emily's voice is vibrant and expressive, bringing life and energy to her speeches. A **female voice with an American accent** enunciates every word with precision. Her voice is **very close-sounding**, and the recording is excellent, capturing her **slightly high-pitched voice** with crisp clarity. Her tone is **very expressive and animated**.",
    },
    {
        "name": "Anna",
        "description": "Anna's voice is elegant and refined, delivering messages with sophistication and charm. A **female voice with an English accent** enunciates every word with precision. Her voice is **very close-sounding**, and the recording is excellent, capturing her voice with crisp clarity. Her tone is **quite expressive and animated**.",
    },
    {
        "name": "Jon",
        "description": "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise. A **male voice with an American accent**, his voice is **very close-sounding**, and the recording is excellent, capturing his **slightly low-pitched voice** with crisp clarity. His tone is **fairly monotone**, and he speaks **slightly quickly**.",
    },
    {
        "name": "Brenda",
        "description": "Brenda's voice is confident and assertive, effectively engaging her audience. A **female voice with an American accent** enunciates every word with precision. Her voice is **very close-sounding**, and the recording is excellent, capturing her voice with crisp clarity. Her tone is **fairly expressive and animated**.",
    },
    {
        "name": "Barbara",
        "description": "Barbara's voice is calm and reassuring, ideal for informative and supportive content. A **female voice with an American accent**, her voice is **very close-sounding and clean**, and the recording is excellent, capturing her voice with crisp clarity. Her tone is **fairly expressive**, conveying warmth and understanding.",
    },
]


class EmotionMLPRegression(nn.Module):
    def __init__(self, input_size=200, hidden_size=512, intermediate_size=256, dropout_prob=0.07):
        super(EmotionMLPRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(hidden_size, intermediate_size)
        self.bn2 = nn.BatchNorm1d(intermediate_size)

        self.fc3 = nn.Linear(intermediate_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        return out


# Function to generate and save audio
def generate_and_save_audio(speaker, prompt_text, sample_number, model, tokenizer, device, tempdir):
    try:
        description = speaker["description"]
        speaker_name = speaker["name"]

        # Tokenize the description and prompt
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

        # Generate audio
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)

        # Convert to numpy array
        audio_arr = generation.cpu().numpy().squeeze()

        # Define output filename
        output_filename = f"{sample_number}_{speaker_name}.wav"
        output_path = os.path.join(tempdir, output_filename)

        # Save audio to file
        sf.write(output_path, audio_arr, model.config.sampling_rate)
        print(f"Saved audio for Sample_{sample_number} with {speaker_name} to {output_path}")

        # Return the path to the output file
        return output_path
    except Exception as e:
        print(f"Failed to generate audio for Sample_{sample_number} with {speaker.get('name', 'Unknown Speaker')}: {e}")
        return None


def calculate_human_similarity_score(audio_emo_vector, model_path, pca_model_path):
    """Calculate the human similarity score based on the audio emotion vector."""

    # Ensure the input is a PyTorch tensor
    if isinstance(audio_emo_vector, np.ndarray):
        audio_emo_vector = torch.tensor(audio_emo_vector, dtype=torch.float32)

    # Initialize the model
    model = EmotionMLPRegression(input_size=200, hidden_size=512)

    # Load the state dictionary into the model
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Load the PCA model
    pca = joblib.load(pca_model_path)

    # Ensure the input has the correct shape (batch dimension)
    if audio_emo_vector.dim() == 1:
        audio_emo_vector = audio_emo_vector.unsqueeze(0)  # Add batch dimension if needed

    # Apply PCA transformation and move the tensor to the appropriate device
    audio_emo_vector_pca = torch.tensor(pca.transform(audio_emo_vector.cpu().numpy()), dtype=torch.float32).to(
        device
    )  # Ensure the tensor is on the same device as the model

    # Make a prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        score = model(audio_emo_vector_pca)

    return score


def scoring_workflow(repo_namespace, repo_name, text, voice_description):
    # Get path relative to the current file's location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    DISCRIMINATOR_PATH = os.path.join(base_dir, "model/emotion_mlp_regression.pth")
    MODEL_PCA_PATH = os.path.join(base_dir, "model/pca_model.pkl")

    # Setup device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the mini parler model from hugging face
    # model_name = "parler-tts/parler-tts-mini-v1"  # an 880M parameter model.
    model_name = f"{repo_namespace}/{repo_name}"  # an 1.2B parameter model.
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize speaker index
    speaker_index = 0

    # Select a speaker from the list
    if speaker_index >= len(SPEAKERS):
        speaker_index = 0  # Reset index if we've used all speakers
    speaker = SPEAKERS[speaker_index]

    # Generate a unique sample number with the speaker's name
    sample_number = f"{speaker['name']}_{uuid.uuid4()}"
    prompt_text = text

    # Create a temporary directory for audio files
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Generate and save audio for the selected sample
        audio_path = generate_and_save_audio(speaker, prompt_text, sample_number, model, tokenizer, device, tmpdirname)

        # === Initialize the Emotion2Vector Model ===
        try:
            inference_pipeline = pipeline(task=Tasks.emotion_recognition, model="iic/emotion2vec_plus_large")
            print("Emotion2Vector Model initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Emotion2Vector Model initialization failed: {e}")

        if not audio_path:
            print(f"Audio file not found for {sample_number} at {audio_path}")
        else:
            try:
                # Extract emotion2vec vectors for audio file
                rec_result = inference_pipeline(audio_path, granularity="utterance", extract_embedding=True)

                audio_emo_vector = rec_result[0]["feats"]  # Extract the embedding from the result

            except Exception as e:
                print(f"Error processing audio file for {sample_number} at {audio_path}: {e}")

    # Calculate the human similarity score
    score = calculate_human_similarity_score(audio_emo_vector, DISCRIMINATOR_PATH, MODEL_PCA_PATH)

    return score
