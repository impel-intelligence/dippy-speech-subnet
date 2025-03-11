import gc
import json
import logging
import os
import tempfile
import time
import uuid
from typing import Any, Dict, Optional

import httpx
import joblib
import librosa
import numpy as np
import ray
import requests
import soundfile as sf
import torch
import torch.distributed as dist
import torch.nn as nn
from dotenv import load_dotenv
from elevenlabs import ElevenLabs, voices, Voice, VoiceSettings
from huggingface_hub import hf_hub_download, login
from jiwer import Compose, RemovePunctuation, Strip, ToLowerCase, wer
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from scoring.scoring_logic.emotion_processing import compute_z_scores, extract_emotions, get_top_n_emotions

load_dotenv(override=True)

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

# Map speaker descriptions to ElevenLabs voices
ELEVENLABS_VOICE_MAPPING = {
    "male": ["Roger", "Charlie", "George", "Callum", "River", "Liam", "Eric", "Chris", "Brian", "Daniel", "Bill"],
    "female": ["Aria", "Sarah", "Laura", "Charlotte", "Alice", "Matilda", "Jessica", "Lily"]
}


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


def select_elevenlabs_voice(voice_description):
    """
    Select an appropriate ElevenLabs voice based on the voice description.
    
    Args:
        voice_description (str): Description of the voice characteristics.
        
    Returns:
        str: Name of the selected ElevenLabs voice.
    """
    # Determine gender from description
    if "male voice" in voice_description.lower():
        gender = "male"
    elif "female voice" in voice_description.lower():
        gender = "female"
    else:
        # Default to male if gender not specified
        gender = "male"
    
    # Get available voices for the gender
    available_voices = ELEVENLABS_VOICE_MAPPING[gender]
    
    # Select a voice based on other characteristics in the description
    if "monotone" in voice_description.lower():
        # For monotone voices
        if gender == "male":
            return "Daniel"  # More neutral/monotone male voice
        else:
            return "Charlotte"   # More neutral/monotone female voice
    elif "expressive" in voice_description.lower() or "animated" in voice_description.lower():
        # For expressive voices
        if gender == "male":
            return "Charlie"  # More expressive male voice
        else:
            return "Sarah"   # More expressive female voice
    elif "low-pitched" in voice_description.lower():
        # For deep voices
        if gender == "male":
            return "George"  # Deeper male voice
        else:
            return "Laura" # Deeper female voice
    elif "high-pitched" in voice_description.lower():
        # For higher pitched voices
        if gender == "male":
            return "River"    # Higher pitched male voice
        else:
            return "Lily"    # Higher pitched female voice
    else:
        # Default voices if no specific characteristics match
        if gender == "male":
            return "Roger"
        else:
            return "Aria"


def generate_audio_elevenlabs(speaker, voice_description, prompt_text, sample_number, device, tempdir):
    """
    Generate audio using ElevenLabs API based on the voice description and text.
    
    Args:
        speaker (dict): Speaker information.
        voice_description (str): Description of the voice characteristics.
        prompt_text (str): Text to convert to speech.
        sample_number (str): Unique identifier for the sample.
        device (str): Device to use for processing (not used for ElevenLabs but kept for API compatibility).
        tempdir (str): Directory to save the audio file.
        
    Returns:
        str: Path to the generated audio file.
    """
    try:
        speaker_name = speaker["name"]
        
        # Check if the text is too long (ElevenLabs has a character limit)
        if len(prompt_text) > 5000:
            # Truncate to 5000 characters
            prompt_text = prompt_text[:5000]
            logger.info(f"Text truncated to 5000 characters for ElevenLabs API")
        
        # Select appropriate ElevenLabs voice
        elevenlabs_voice = select_elevenlabs_voice(voice_description)
        
        # Set voice settings based on description
        stability = 0.5  # Default stability
        similarity_boost = 0.75  # Default similarity boost
        
        if "monotone" in voice_description.lower():
            stability = 0.8  # More stable/consistent for monotone voices
        elif "expressive" in voice_description.lower() or "animated" in voice_description.lower():
            stability = 0.3  # Less stable for more expressive voices
            similarity_boost = 0.5  # Less similarity boost for more variation
            
        voice_settings = VoiceSettings(
            stability=stability,
            similarity_boost=similarity_boost
        )
        
        # Get API key from environment
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not set in environment")
            
        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=api_key)
        
        # Get all available voices
        voices_response = client.voices.get_all()
        available_voices = voices_response.voices
        # Log available voices for debugging
        logger.info(f"Available voices: {[v.name for v in available_voices]}")
        selected_voice = next((v for v in available_voices if v.name == elevenlabs_voice), None)
        if not selected_voice:
            raise ValueError(f"Voice {elevenlabs_voice} not found")

        # Generate audio using ElevenLabs
        audio_generator = client.generate(
            text=prompt_text,
            voice=selected_voice,
            model="eleven_multilingual_v2"
        )
        
        # Convert generator to bytes
        audio_data = b"".join(chunk for chunk in audio_generator)
        
        # Define output filename
        output_filename = f"{sample_number}_{speaker_name}.wav"
        output_path = os.path.join(tempdir, output_filename)
        
        # Save audio to file
        with open(output_path, "wb") as f:
            f.write(audio_data)
            
        logger.info(f"Saved ElevenLabs audio for Sample_{sample_number} with {speaker_name} to {output_path}")
        
        # Return the path to the output file
        return output_path
    except Exception as e:
        logger.error(
            f"Failed to generate ElevenLabs audio for Sample_{sample_number} with {speaker.get('name', 'Unknown Speaker')}: {e}"
        )
        raise RuntimeError(f"ElevenLabs audio generation failed. {e}")


def process_emotion(audio_path):
    try:
        # Get API key from environment
        api_key = os.environ.get("HUME_API_KEY")
        if not api_key:
            raise ValueError("HUME_API_KEY not set in environment")

        # Process with Hume AI
        predictions = process_hume_ai_file(api_key=api_key, file_path=audio_path)

        # Extract and process emotions
        emotion_scores = extract_emotions(data=predictions)

        # Get z-scores
        z_scores = compute_z_scores(emotion_scores=emotion_scores)

        # Get top emotions
        top_emotions = get_top_n_emotions(emotion_scores=emotion_scores, n=3)

        return {"raw_scores": dict(emotion_scores), "z_scores": z_scores, "top_emotions": top_emotions}

    except Exception as e:
        logger.error(f"Failed to process audio with Hume AI: {e}", exc_info=True)
        raise RuntimeError("Hume AI processing failed.")


def submit_hume_ai_job(api_key: str, file_path: str, models: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit a batch job to Hume AI for processing.

    Args:
        api_key (str): Your Hume AI API key
        file_path (str): Path to the audio file to be processed
        models (dict, optional): Configuration for Hume AI models.
                                 Defaults to burst and prosody models.

    Returns:
        str: The job ID for tracking the processing status
    """
    # Default models configuration if not provided
    if models is None:
        models = {"burst": {}, "prosody": {"window": {"length": 60}}}

    # Prepare the POST request
    post_response = requests.post(
        "https://api.hume.ai/v0/batch/jobs",
        headers={
            "X-Hume-Api-Key": api_key,
        },
        data={
            "json": json.dumps({"models": models}),
        },
        files=[("file", (file_path.split("/")[-1], open(file_path, "rb")))],
    )

    # Check response and extract job ID
    post_response.raise_for_status()
    job_info = post_response.json()
    job_id = job_info.get("job_id")

    if not job_id:
        raise ValueError("Job ID not found in response!")

    return job_id


def wait_for_hume_ai_job(api_key: str, job_id: str, polling_interval: int = 4) -> Dict[str, Any]:
    """
    Poll the Hume AI job status and retrieve predictions once complete.

    Args:
        api_key (str): Your Hume AI API key
        job_id (str): The job ID to track
        polling_interval (int, optional): Seconds to wait between status checks.
                                          Defaults to 10.

    Returns:
        dict: The job predictions
    """
    job_url = f"https://api.hume.ai/v0/batch/jobs/{job_id}"

    while True:
        # Check job status
        status_response = requests.get(job_url, headers={"X-Hume-Api-Key": api_key})
        status_response.raise_for_status()

        status_data = status_response.json()
        current_status = status_data.get("state", {}).get("status", "").lower()
        print("Current job status:", current_status)

        if current_status == "completed":
            # Retrieve predictions
            predictions_response = requests.get(f"{job_url}/predictions", headers={"X-Hume-Api-Key": api_key})
            predictions_response.raise_for_status()
            return predictions_response.json()

        elif current_status == "failed":
            raise Exception("The Hume AI job failed to complete.")

        # Wait before next polling
        time.sleep(polling_interval)


def process_hume_ai_file(
    api_key: str, file_path: str, output_path: Optional[str] = None, models: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Comprehensive function to process an audio file through Hume AI.

    Args:
        api_key (str): Your Hume AI API key
        file_path (str): Path to the audio file to be processed
        output_path (str, optional): Path to save predictions JSON.
                                     Defaults to None (no file saved).
        models (dict, optional): Custom models configuration

    Returns:
        dict: The job predictions
    """
    # Submit the job
    job_id = submit_hume_ai_job(api_key, file_path, models)

    # Wait for and retrieve predictions
    predictions = wait_for_hume_ai_job(api_key, job_id)

    # Optionally save predictions to a file
    if output_path:
        with open(output_path, "w") as json_file:
            json.dump(predictions, json_file, indent=4)

    return predictions


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
    logger.info(f"Starting Whisper transcription for audio: {audio_path}")

    try:
        # Verify audio file exists
        if not os.path.exists(audio_path):
            error_msg = f"Audio file not found at path: {audio_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Verify transcription URL is set
        if not transcription_url:
            error_msg = "Whisper endpoint URL is not configured"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Sending request to Whisper endpoint: {transcription_url}")

        # Open the audio file and send it in the POST request
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path, f)}
            try:
                response = httpx.post(transcription_url, files=files)
            except httpx.ConnectError as e:
                error_msg = f"Failed to connect to Whisper endpoint: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            except httpx.TimeoutException as e:
                error_msg = f"Whisper request timed out: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            except httpx.RequestError as e:
                error_msg = f"Whisper request failed: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        # Check if the request was successful
        if response.status_code == 200:
            transcription = response.text.strip()
            if not transcription:
                error_msg = "Whisper returned empty transcription"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            logger.info("Whisper transcription completed successfully")
            logger.debug(f"Transcription result: {transcription}")
            return transcription
        else:
            error_msg = f"Whisper service error (status {response.status_code}): {response.text}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    except Exception as e:
        error_msg = f"Whisper transcription failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg)


def scoring_workflow_elevenlabs(text, voice_description, device):
    """
    Workflow for scoring TTS using ElevenLabs.
    
    Args:
        text (str): Text to convert to speech.
        voice_description (str): Description of the voice characteristics.
        device (str): Device to use for processing.
        
    Returns:
        tuple: (raw_emotion_scores, wer_score)
    """
    try:
        # Attempt to retrieve the Whisper API token from the environment variables
        whisper_endpoint = os.environ.get("WHISPER_ENDPOINT")

        # Check if the token is None (i.e., not set in the environment)
        if whisper_endpoint is None:
            raise ValueError("WHISPER_ENDPOINT is not set in the environment.")

    except Exception as e:
        # Handle the exception (e.g., log the error, notify the user, etc.)
        raise RuntimeError(f"An error occurred getting whisper endpoint url from env: {e}")

    try:
        # Check for ElevenLabs API key
        elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
        if elevenlabs_api_key is None:
            raise ValueError("ELEVENLABS_API_KEY is not set in the environment.")
    except Exception as e:
        # Handle the exception
        raise RuntimeError(f"An error occurred setting up ElevenLabs: {e}")

    # Initialize speaker
    speaker_index = 0
    speaker = SPEAKERS[speaker_index % len(SPEAKERS)]
    sample_number = f"{speaker['name']}_{uuid.uuid4()}"

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Generate audio using ElevenLabs
        audio_path = generate_audio_elevenlabs(
            speaker, voice_description, text, sample_number, device, tmpdirname
        )

        # Process emotion
        audio_emo_output = process_emotion(audio_path)

        # Transcribe audio
        transcription = transcribe_audio(audio_path)

    # Validate results
    if audio_emo_output is None or not audio_emo_output.get("raw_scores"):
        raise RuntimeError("Emotion vector is missing or empty.")
    if not transcription.strip():
        raise RuntimeError("Transcription is missing or empty.")

    # Calculate WER
    try:
        wer_score = calculate_wer(text, transcription)
        logger.info(f"Word Error Rate (WER) calculated: {wer_score}")
    except Exception as e:
        logger.error(f"Failed to calculate Word Error Rate (WER): {e}", exc_info=True)
        raise RuntimeError(f"WER calculation failed: {e}")

    # Detect Expected emotion
    try:
        raw_emotion_scores = audio_emo_output["raw_scores"]
    except Exception as e:
        logger.error(f"Failed to calculate human similarity score: {e}", exc_info=True)
        raise RuntimeError(f"Human similarity score calculation failed.{e}")

    try:
        # Force garbage collection before clearing CUDA cache
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.info(f"CUDA cleanup error: {e}")

    try:
        # Shut down Ray if it's running
        if ray.is_initialized():
            ray.shutdown()
    except Exception as e:
        logger.info(f"Ray shutdown error: {e}")

    try:
        # Destroy process group if initialized
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.info(f"Torch distributed cleanup error: {e}")

    return (raw_emotion_scores, wer_score)
