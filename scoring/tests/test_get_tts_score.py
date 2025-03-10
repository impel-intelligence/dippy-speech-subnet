import logging
from scoring.get_tts_score import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_load_dataset():
    """Test the dataset loading and validate its structure and content"""
    try:
        # Load the dataset
        data = load_dataset()
        
        # Test dataset size
        assert len(data) == 40, f"Dataset should contain 40 samples, but got {len(data)}"
        
        # Test data structure and content
        for i, item in enumerate(data):
            # Test dictionary structure
            assert isinstance(item, dict), f"Item {i} should be a dictionary, but got {type(item)}"
            required_keys = ["target_text", "last_user_message", "voice_description", "character_profile", "emotional_text"]
            assert all(key in item for key in required_keys), f"Item {i} missing required keys. Required: {required_keys}, Got: {item.keys()}"
            
            # Test target_text (response)
            assert isinstance(item["target_text"], str), f"Response in item {i} should be string, but got {type(item['target_text'])}"
            assert len(item["target_text"].strip()) > 0, f"Response in item {i} should not be empty"
            
            # Test last_user_message (query)
            assert isinstance(item["last_user_message"], str), f"Query in item {i} should be string, but got {type(item['last_user_message'])}"
            assert len(item["last_user_message"].strip()) > 0, f"Query in item {i} should not be empty"
            
            # Test voice_description
            assert isinstance(item["voice_description"], str), f"Description in item {i} should be string, but got {type(item['voice_description'])}"
            assert len(item["voice_description"].strip()) > 0, f"Description in item {i} should not be empty"
            
            # Test character_profile
            assert isinstance(item["character_profile"], dict), f"Character profile in item {i} should be a dictionary, but got {type(item['character_profile'])}"
            
            # Test emotional_text
            assert isinstance(item["emotional_text"], list), f"Emotional text in item {i} should be a list, but got {type(item['emotional_text'])}"
            assert len(item["emotional_text"]) > 0, f"Emotional text in item {i} should not be empty"
            
            # Log sample information for visibility
            if i < 2:
                logger.info(f"\nItem {i+1}:")
                logger.info(f"Response: {item['target_text'][:100]}...")
                logger.info(f"Query: {item['last_user_message']}")
                logger.info(f"Description: {item['voice_description'][:100]}...")
                logger.info(f"Character Profile: {item['character_profile']}")
                logger.info(f"Emotional Text: {item['emotional_text']}")
                
        logger.info(f"Successfully validated dataset with {len(data)} samples")
                
    except Exception as e:
        logger.error(f"Error in test_load_dataset: {e}")
        raise

if __name__ == "__main__":
    test_load_dataset()
