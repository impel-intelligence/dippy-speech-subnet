import logging
from scoring.scoring_logic.logic import process_emotion
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hume_integration():
    # Load environment variables from .env file
    load_dotenv(override=True)
    
    # Path to your test audio file
    test_audio_path = "scoring/tests/test_audio/Sample_1_Will.wav"
    
    try:
        # Process the audio file
        logger.info(f"Processing audio file: {test_audio_path}")
        results = process_emotion(test_audio_path, emotion_inference_pipeline=None)
        logger.info("Results:")
        logger.info(f"Raw scores: {results['raw_scores']}")
        logger.info(f"Z-scores: {results['z_scores']}")
        logger.info(f"Top emotions: {results['top_emotions']}")
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")

if __name__ == "__main__":
    test_hume_integration()
