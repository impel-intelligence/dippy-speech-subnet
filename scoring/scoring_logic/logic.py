import logging
import os
import tempfile
import uuid
import httpx
import torchaudio
import re

import gc
import torch
import ray
import torch.distributed as dist

import joblib
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from jiwer import Compose, RemovePunctuation, Strip, ToLowerCase, wer
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor
from parler_tts import ParlerTTSConfig, ParlerTTSForConditionalGeneration, build_delay_pattern_mask
from huggingface_hub import login

logger = logging.getLogger(__name__)

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

TRANSCRIPTION_URL = os.environ.get("WHISPER_ENDPOINT")

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


def calculate_human_similarity_score(audio_emo_vector, model_file_name, pca_file_name):
    """Calculate the human similarity score based on the audio emotion vector."""

    # Ensure the input is a PyTorch tensor
    if isinstance(audio_emo_vector, np.ndarray):
        audio_emo_vector = torch.tensor(audio_emo_vector, dtype=torch.float32)

    # Initialize the model
    model = EmotionMLPRegression(input_size=200, hidden_size=512)

    # Load the state dictionary into the model
    model_path = hf_hub_download(
        repo_id="DippyAI-Speech/Discriminator",
        filename=model_file_name,  # Replace with the correct filename if different
    )

    # Load the state dictionary into the model
    pca_model_path = hf_hub_download(
        repo_id="DippyAI-Speech/PCA", filename=pca_file_name  # Replace with the correct filename if different
    )

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


def calculate_wer(reference: str, hypothesis: str, apply_preprocessing: bool = True) -> float:
    """
    Calculate the Word Error Rate (WER) between a reference text and a hypothesis text.

    Args:
        reference (str): The ground truth reference text.
        hypothesis (str): The transcribed hypothesis text.
        apply_preprocessing (bool): Whether to apply preprocessing to normalize texts.

    Returns:
        float: The Word Error Rate (WER).
    """
    # Preprocessing pipeline
    if apply_preprocessing:
        preprocessing = Compose(
            [
                RemovePunctuation(),  # Remove punctuation
                ToLowerCase(),  # Convert to lowercase
                Strip(),  # Strip leading/trailing spaces
            ]
        )
        # Apply preprocessing to both reference and hypothesis
        reference = preprocessing(reference)
        hypothesis = preprocessing(hypothesis)

    # Calculate WER
    error_rate = wer(reference, hypothesis)

    return error_rate


def generate_audio(speaker, prompt_text, sample_number, model, tokenizer, device, tempdir):
    try:
        description = speaker["description"]
        speaker_name = speaker["name"]

        # Tokenize the text and count tokens
        encoded = tokenizer(prompt_text, return_tensors="pt")
        token_count = encoded.input_ids.shape[1]

        # Check if the token count exceeds 100
        if token_count > 100:
            # Truncate while keeping meaning
            truncated_tokens = encoded.input_ids[0][:100]  # Keep only first 100 tokens
            prompt_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

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
        logger.info(f"Saved audio for Sample_{sample_number} with {speaker_name} to {output_path}")

        # Return the path to the output file
        return output_path
    except Exception as e:
        logger.error(
            f"Failed to generate audio for Sample_{sample_number} with {speaker.get('name', 'Unknown Speaker')}: {e}"
        )
        raise RuntimeError(f"Audio generation failed. {e}")


def process_emotion(audio_path, emotion_inference_pipeline):
    try:
        rec_result = emotion_inference_pipeline(audio_path, granularity="utterance", extract_embedding=True)
        return rec_result[0]["feats"]
    except Exception as e:
        logger.error(f"Failed to process audio for Emotion2Vector: {e}", exc_info=True)
        raise RuntimeError("Emotion2Vector processing failed.")


def transcribe_audio(audio_path, transcription_url=TRANSCRIPTION_URL):
    """
    Transcribes an audio file by sending it to a remote transcription service via an HTTP POST request.

    Args:
        audio_path (str): Path to the audio file to be transcribed.
        transcription_url (str): URL of the transcription service endpoint.

    Returns:
        str: The transcription of the audio file.

    Raises:
        RuntimeError: If the transcription fails due to connection issues or other errors.
    """
    try:
        # Open the audio file and send it in the POST request
        with open(audio_path, 'rb') as f:
            files = {'file': (audio_path, f)}
            response = httpx.post(transcription_url, files=files)

        # Check if the request was successful
        if response.status_code == 200:
            transcription = response.text.strip()
            logger.info(f"\nAudio Transcription Response:\n{transcription}")
            return transcription
        else:
            error_message = f"Transcription service returned status code {response.status_code}: {response.text}"
            logger.info(error_message)
            raise RuntimeError(error_message)

    except FileNotFoundError:
        error_message = f"Error: The file '{audio_path}' was not found."
        logger.info(error_message)
        raise RuntimeError(error_message)

    except httpx.ConnectError as e:
        error_message = f"Failed to connect to transcription endpoint: {e}"
        logger.info(error_message)
        raise RuntimeError(error_message)

    except Exception as e:
        error_message = f"An unexpected error occurred during transcription: {e}"
        logger.info(error_message)
        raise RuntimeError(error_message)


def process_audio(audio_path, feature_extractor, audio_encoder, num_codebooks, bos_token_id, eos_token_id, device):
    audio, sr = torchaudio.load(audio_path)

    if sr != feature_extractor.sampling_rate:
        audio = torchaudio.functional.resample(audio, sr, feature_extractor.sampling_rate)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    inputs = feature_extractor(
        audio.squeeze().numpy(),
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        tokens = audio_encoder.encode(**inputs)["audio_codes"]
    
    # Debug information
    logger.info(f"Tokens shape: {tokens.shape}")
    
    # Handle different dimensions
    if tokens.dim() == 4:  # If tokens has 4 dimensions
        tokens = tokens.squeeze(0)  # Remove the first dimension
    
    bos_labels = torch.full((1, num_codebooks, 1), bos_token_id, device=device, dtype=torch.long)
    logger.info(f"BOS labels shape: {bos_labels.shape}")
    
    # Now both should be 3-dimensional
    tokens = torch.cat([bos_labels, tokens], dim=-1)
    logger.info(f"Combined tokens shape: {tokens.shape}")

    max_length = tokens.shape[-1] + num_codebooks
    input_ids, pattern_mask = build_delay_pattern_mask(
        input_ids=tokens,
        bos_token_id=bos_token_id,
        pad_token_id=eos_token_id,
        max_length=max_length,
        num_codebooks=num_codebooks
    )

    pattern_mask = torch.where(pattern_mask == -1, eos_token_id, pattern_mask)

    return pattern_mask

def cross_entropy(model, config, tokenizer, prompt_text, description_text, device, feature_extractor, groundTruth):

    num_codebooks = model.decoder.config.num_codebooks
    bos_token_id = model.generation_config.decoder_start_token_id
    eos_token_id = config.decoder.eos_token_id


    # Tokenize prompt and description
    encoded_prompt = tokenizer(prompt_text, return_tensors="pt", padding=True).to(device)
    encoded_description = tokenizer(description_text, return_tensors="pt", padding=True).to(device)

    # Prepare ground truth tokens
    ground_truth_tokens = process_audio(
        groundTruth,
        feature_extractor,
        model.audio_encoder,
        num_codebooks,
        bos_token_id,
        eos_token_id,
        device
    )
    ground_truth_tokens = ground_truth_tokens.unsqueeze(0).permute(0, 2, 1)  # (1, seq_len, codebooks)
    ground_truth_mask = (ground_truth_tokens != eos_token_id).all(dim=-1).long()

    # === Model Evaluation ===
    with torch.no_grad():
        outputs = model(
            input_ids=encoded_description['input_ids'],
            attention_mask=encoded_description['attention_mask'],
            prompt_input_ids=encoded_prompt['input_ids'],
            prompt_attention_mask=encoded_prompt['attention_mask'],
            labels=ground_truth_tokens,
            decoder_attention_mask=ground_truth_mask,
            return_dict=True,
            loss_reduction="sum"
        )


    # Normalize loss
    total_valid_tokens = ground_truth_mask.sum().float()
    normalized_loss = (outputs.loss / total_valid_tokens).item()
    
    logger.info(f"Normalized loss: {normalized_loss}")
    return normalized_loss


def generate_ground_truth(text, output_path="ground_truth.wav", tmpdirname=None):
    """
    Generate speech from text using the TTS API endpoint.
    
    Args:
        text (str): The text to synthesize into speech
        output_path (str): Path where the output WAV file will be saved
        tmpdirname (str, optional): Temporary directory to use for saving the file
        
    Returns:
        str: Path to the generated audio file
        
    Raises:
        RuntimeError: If the API request fails
        
    """
    # If tmpdirname is provided, use it for the output path
    if tmpdirname:
        output_path = os.path.join(tmpdirname, os.path.basename(output_path))
    api_key = os.environ.get("GROUND_TRUTH_API_KEY")
    api_url = os.environ.get("GROUND_TRUTH_API_URL")
    try:
        # Prepare headers and data for the request
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
        
        data = {
            "text": text,
            "max_audio_length_ms": 100000,  # Set maximum audio length to 100 seconds
            "temperature": 1.0  # Set temperature parameter
        }
        
        # Make the POST request with a longer timeout
        timeout = httpx.Timeout(300.0)  # 5 minutes timeout
        with httpx.Client(timeout=timeout) as client:
            logger.info(f"Sending request to TTS API with text: {text[:50]}...")
            response = client.post(api_url, headers=headers, json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Save the response content to the output file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Speech generated successfully and saved to {output_path}")
            return output_path
        else:
            error_message = f"API request failed with status code {response.status_code}: {response.text}"
            logger.error(error_message)
            raise RuntimeError(error_message)
            
    except httpx.TimeoutException as e:
        error_message = f"Request to TTS API timed out after 300 seconds: {e}"
        logger.error(error_message)
        raise RuntimeError(error_message)
        
    except httpx.ConnectError as e:
        error_message = f"Failed to connect to TTS API endpoint: {e}"
        logger.error(error_message)
        raise RuntimeError(error_message)
        
    except Exception as e:
        error_message = f"An unexpected error occurred during ground truth speech generation: {e}"
        logger.error(error_message)
        raise RuntimeError(error_message)


# def clean_text(text: str) -> str:
#     """Removes special characters from a string except letters, numbers, and spaces."""
#     return re.sub(r'[^a-zA-Z0-9\s]', '', text)


def clean_text(text: str) -> str:
    """Removes special characters from a string except letters, numbers, spaces, and select punctuation."""
    # Remove standalone backslashes
    text = re.sub(r"\\+", '', text)
    # Remove unwanted special characters while keeping specific ones
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'()\"\[\]-]", '', text)
    return text


def scoring_workflow(repo_namespace, repo_name, text, voice_description, device, model, tokenizer, config, feature_extractor):
    DISCRIMINATOR_FILE_NAME = "discriminator_v1.0.pth"
    MODEL_PCA_FILE_NAME = "discriminator_pca_v1.0.pkl"

    try:
        # Attempt to retrieve the Whisper API token from the environment variables
        whisper_endpoint = os.environ.get("WHISPER_ENDPOINT")
        
        # Check if the token is None (i.e., not set in the environment)
        if whisper_endpoint is None:
            raise ValueError("WHISPER_ENDPOINT is not set in the environment.")
        

    except Exception as e:
        # Handle the exception (e.g., log the error, notify the user, etc.)
        raise RuntimeError(f"An error occurred getting wishper endpoint url from env: {e}")
    

    try:
        token = os.environ.get("HUGGINGFACE_TOKEN_PRIME")
        if token is None:
            raise ValueError("HUGGINGFACE_TOKEN_PRIME is not set in the environment.")
        
        login(token=token)
    except Exception as e:
        # Handle the exception (e.g., log the error, notify the user, etc.)
        raise RuntimeError(f"An error occurred during hf login: {e}")



    # Initialize speaker
    speaker_index = 0
    speaker = SPEAKERS[speaker_index % len(SPEAKERS)]
    sample_number = f"{speaker['name']}_{uuid.uuid4()}"

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Prepare text - check if it needs truncation
        encoded = tokenizer(text, return_tensors="pt")
        token_count = encoded.input_ids.shape[1]
        
        # Truncate text if needed to ensure consistent length
        if token_count > 100:
            truncated_tokens = encoded.input_ids[0][:100]  # Keep only first 100 tokens
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        else:
            truncated_text = text

        truncated_text = clean_text(truncated_text)
            
        # Generate audio with potentially truncated text
        audio_path = generate_audio(speaker, truncated_text, sample_number, model, tokenizer, device, tmpdirname)

        # Use the same truncated text for ground truth to ensure consistency
        ground_truth_path = generate_ground_truth(truncated_text, f"ground_truth_{sample_number}.wav", tmpdirname)


        # Process emotion
        # audio_emo_vector = process_emotion(audio_path, emotion_inference_pipeline)
        loss = cross_entropy(model, config, tokenizer, text, voice_description, device, feature_extractor, ground_truth_path)
        logger.info(f"Cross Entropy Loss: {loss}")

        # Transcribe audio
        transcription = transcribe_audio(audio_path)


    try:
        wer_score = calculate_wer(text, transcription)
        logger.info(f"Word Error Rate (WER) calculated: {wer_score}")
    except Exception as e:
        logger.error(f"Failed to calculate Word Error Rate (WER): {e}", exc_info=True)
        raise RuntimeError(f"WER calculation failed: {e}")


    try:
        del model
    except NameError:
        logger.info("Model was not defined")

    try:
        del tokenizer
    except NameError:
        logger.info("Tokenizer was not defined")



    return (loss, wer_score)
