import os
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
    
    # Verify test audio file exists
    assert os.path.exists(test_audio_path), f"Test audio file not found at {test_audio_path}"
    
    try:
        # Process the audio file
        logger.info(f"Processing audio file: {test_audio_path}")
        results = process_emotion(test_audio_path)
        
        # Validate results structure
        assert isinstance(results, dict), "Results should be a dictionary"
        assert all(key in results for key in ['raw_scores', 'z_scores', 'top_emotions']), "Missing required keys in results"
        
        # Validate raw_scores
        assert isinstance(results['raw_scores'], dict), "Raw scores should be a dictionary"
        assert len(results['raw_scores']) > 0, "Raw scores should not be empty"
        assert all(isinstance(score, (int, float)) for score in results['raw_scores'].values()), "Raw scores should be numeric"
        
        # Validate z_scores
        assert isinstance(results['z_scores'], dict), "Z-scores should be a dictionary"
        assert len(results['z_scores']) > 0, "Z-scores should not be empty"
        assert all(isinstance(score, (int, float)) for score in results['z_scores'].values()), "Z-scores should be numeric"
        
        # Validate top_emotions
        assert isinstance(results['top_emotions'], list), "Top emotions should be a list"
        assert len(results['top_emotions']) > 0, "Top emotions should not be empty"
        assert all(isinstance(emotion, tuple) and len(emotion) == 2 for emotion in results['top_emotions']), "Each emotion should be a tuple of (name, score)"
        assert all(isinstance(emotion[0], str) and isinstance(emotion[1], (int, float)) for emotion in results['top_emotions']), "Each emotion tuple should contain a string name and numeric score"
        
        # Log results for visibility
        logger.info("Results:")
        logger.info(f"Raw scores: {results['raw_scores']}")
        logger.info(f"Z-scores: {results['z_scores']}")
        logger.info(f"Top emotions: {results['top_emotions']}")
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise

if __name__ == "__main__":
    test_hume_integration()
