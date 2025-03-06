import logging
from scoring.get_tts_score import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_load_dataset():
    """Test to print the loaded dataset structure"""
    try:
        # Load the dataset
        data = load_dataset()
        
        # Print basic information
        logger.info(f"Dataset length: {len(data)}")
        
        # Print first item structure
        if data:
            logger.info("\nFirst item structure:")
            first_item = data[0]
            logger.info(f"Type: {type(first_item)}")
            logger.info(f"Number of elements: {len(first_item)}")
            

            # Print first few items
            logger.info("\nFirst 2 items for inspection:")
            for i, item in enumerate(data[:2]):
                logger.info(f"\nItem {i+1}:")
                logger.info(f"Response: {item[0][:100]}...")  # First 100 chars
                logger.info(f"Query: {item[1]}")
                logger.info(f"Description: {item[2][:100]}...")  # First 100 chars
                logger.info(f"top_k_emotions: {item[3]}")

                
    except Exception as e:
        logger.error(f"Error in test_load_dataset: {e}")
        raise

if __name__ == "__main__":
    test_load_dataset()
