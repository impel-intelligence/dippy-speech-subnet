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
            # Test tuple structure
            assert isinstance(item, tuple), f"Item {i} should be a tuple, but got {type(item)}"
            assert len(item) == 4, f"Item {i} should have 4 elements, but got {len(item)}"
            
            # Test response
            assert isinstance(item[0], str), f"Response in item {i} should be string, but got {type(item[0])}"
            assert len(item[0].strip()) > 0, f"Response in item {i} should not be empty"
            
            # Test query
            assert isinstance(item[1], str), f"Query in item {i} should be string, but got {type(item[1])}"
            assert len(item[1].strip()) > 0, f"Query in item {i} should not be empty"
            
            # Test description
            assert isinstance(item[2], str), f"Description in item {i} should be string, but got {type(item[2])}"
            assert len(item[2].strip()) > 0, f"Description in item {i} should not be empty"
         
            
            # Test emotions
            assert isinstance(item[3], list), f"Emotions in item {i} should be a list, but got {type(item[3])}"
            
            # Log sample information for visibility
            if i < 2:
                logger.info(f"\nItem {i+1}:")
                logger.info(f"Response: {item[0][:100]}...")
                logger.info(f"Query: {item[1]}")
                logger.info(f"Description: {item[2][:100]}...")
                logger.info(f"top_k_emotions: {item[3]}")
                
        logger.info(f"Successfully validated dataset with {len(data)} samples")
                
    except Exception as e:
        logger.error(f"Error in test_load_dataset: {e}")
        raise

if __name__ == "__main__":
    test_load_dataset()
