import logging
import os
import pytest
import httpx
from unittest.mock import patch, Mock
from pathlib import Path
from scoring.scoring_logic.logic import transcribe_audio, calculate_wer
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def test_audio_path():
    """Fixture to provide path to test audio file"""
    current_dir = Path(__file__).parent
    return str(current_dir / "test_audio" / "Sample_1_Will.wav")

@pytest.fixture
def whisper_url():
    """Fixture to provide whisper endpoint URL from environment"""
    url = os.environ.get("WHISPER_ENDPOINT")
    if not url:
        pytest.skip("WHISPER_ENDPOINT environment variable not set")
    return url

def test_transcribe_audio_basic(test_audio_path, whisper_url):
    """Test basic audio transcription functionality"""
    try:
        # Mock response data
        expected_transcription = "This is a test transcription"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = expected_transcription

        # Test transcription
        with patch('httpx.post', return_value=mock_response) as mock_post:
            result = transcribe_audio(test_audio_path, whisper_url)
            
            # Verify the result
            assert result == expected_transcription
            assert isinstance(result, str)
            assert len(result) > 0
            
            # Verify the POST request
            mock_post.assert_called_once()
            assert mock_post.call_args[0][0] == whisper_url
            
            # Verify request payload
            files = mock_post.call_args[1]['files']
            assert 'file' in files
            assert len(files) == 1
            assert files['file'][0] == test_audio_path  # Check filename
            
            logger.info("Basic transcription test passed successfully")
            
    except Exception as e:
        logger.error(f"Error in basic transcription test: {e}")
        raise

def test_transcribe_audio_error_handling(test_audio_path, whisper_url):
    """Test error handling in transcription"""
    test_cases = [
        {
            'name': 'empty_response',
            'response': Mock(status_code=200, text='   '),
            'expected_error': 'Whisper returned empty transcription'
        },
        {
            'name': 'server_error',
            'response': Mock(status_code=500, text='Server Error'),
            'expected_error': 'Whisper service error'
        }
    ]
    
    for case in test_cases:
        with patch('httpx.post', return_value=case['response']):
            with pytest.raises(RuntimeError) as exc_info:
                transcribe_audio(test_audio_path, whisper_url)
            assert case['expected_error'] in str(exc_info.value)
            logger.info(f"Error handling test '{case['name']}' passed")

def test_transcribe_audio_missing_url(test_audio_path):
    """Test handling of missing whisper endpoint URL"""
    with pytest.raises(RuntimeError) as exc_info:
        transcribe_audio(test_audio_path, None)
    assert "Whisper endpoint URL is not configured" in str(exc_info.value)
    logger.info("Missing URL test passed successfully")

def test_transcribe_audio_file_not_found(whisper_url):
    """Test handling of non-existent audio file"""
    with pytest.raises(RuntimeError) as exc_info:
        transcribe_audio("nonexistent_audio.wav", whisper_url)
    assert "Audio file not found" in str(exc_info.value)
    logger.info("File not found test passed successfully")

def test_transcribe_audio_network_errors(test_audio_path, whisper_url):
    """Test network-related error handling"""
    test_cases = [
        {
            'name': 'connection_error',
            'exception': httpx.ConnectError("Failed to connect"),
            'expected_error': 'Failed to connect to Whisper endpoint'
        },
        {
            'name': 'timeout_error',
            'exception': httpx.TimeoutException("Request timed out"),
            'expected_error': 'Whisper request timed out'
        }
    ]
    
    for case in test_cases:
        with patch('httpx.post', side_effect=case['exception']):
            with pytest.raises(RuntimeError) as exc_info:
                transcribe_audio(test_audio_path, whisper_url)
            assert case['expected_error'] in str(exc_info.value)
            logger.info(f"Network error test '{case['name']}' passed")

def test_wer_calculation():
    """Test Word Error Rate calculation"""
    test_cases = [
        {
            'reference': "this is a test",
            'hypothesis': "this is a test",
            'expected_wer': 0.0
        },
        {
            'reference': "this is a test",
            'hypothesis': "this is test",
            'expected_wer': 0.25  # One word different out of 4
        },
        {
            'reference': "this is a test",
            'hypothesis': "that was the test",
            'expected_wer': 0.75  # Three words different out of 4
        }
    ]
    
    for case in test_cases:
        wer = calculate_wer(case['reference'], case['hypothesis'])
        assert abs(wer - case['expected_wer']) < 0.01  # Allow small floating point differences
        logger.info(f"WER calculation test passed for case: {case['reference']} vs {case['hypothesis']}")

def test_wer_preprocessing():
    """Test WER calculation with different text preprocessing"""
    reference = "This is a TEST!"
    hypothesis = "this is a test"
    
    # With preprocessing (default)
    wer_with_preprocessing = calculate_wer(reference, hypothesis)
    assert wer_with_preprocessing == 0.0
    
    # Without preprocessing
    wer_without_preprocessing = calculate_wer(reference, hypothesis, apply_preprocessing=False)
    assert wer_without_preprocessing > 0.0
    
    logger.info("WER preprocessing tests passed successfully")

if __name__ == "__main__":
    pytest.main([__file__])
