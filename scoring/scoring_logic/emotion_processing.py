from collections import defaultdict
import json
import statistics
import logging
from typing import Dict, List, Tuple, Union

logger = logging.getLogger(__name__)

def extract_emotions(data: Union[Dict, List]) -> Dict[str, float]:
    logger.info("Starting emotion extraction")
    """
    Extracts and aggregates emotion scores from the JSON data.

    The function navigates the nested JSON structure. It supports input
    that is either a single dictionary or a list of dictionaries. Each
    dictionary is expected to contain predictions under either:
      - data["results"]["predictions"], or
      - data["predictions"]

    Args:
        data: JSON data containing emotion predictions, either as a dict or list of dicts

    Returns:
        A dictionary with emotion names as keys and their aggregated scores as values.
    """
    emotion_scores = defaultdict(float)

    # Normalize data to a list for unified processing
    items = data if isinstance(data, list) else [data]

    for item in items:
        # Determine the list of prediction entries.
        if "results" in item and "predictions" in item["results"]:
            predictions = item["results"]["predictions"]
        elif "predictions" in item:
            predictions = item["predictions"]
        else:
            predictions = []
            logger.warning("No predictions found in data")

        # Iterate through all predictions and sum the emotion scores.
        for entry in predictions:
            models = entry.get("models", {})
            prosody = models.get("prosody", {})
            groups = prosody.get("grouped_predictions", [])
            for group in groups:
                for pred in group.get("predictions", []):
                    for emo in pred.get("emotions", []):
                        name = emo.get("name")
                        score = emo.get("score", 0)
                        if name:
                            emotion_scores[name] += score

    return emotion_scores


def get_top_n_emotions(emotion_scores: Dict[str, float], n: int = 3) -> List[Tuple[str, float]]:
    logger.info("Getting top emotions")
    """
    Returns the top n emotions from the aggregated emotion_scores dictionary.

    Args:
        emotion_scores: Dictionary with emotion names as keys and scores as values.
        n: Number of top emotions to return.

    Returns:
        A list of tuples (emotion_name, score) sorted by score in descending order.
    """
    return sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:n]


def compute_z_scores(emotion_scores: Dict[str, float]) -> Dict[str, float]:
    logger.info("Computing z-scores")
    """
    Computes the z-score for each emotion score.

    Args:
        emotion_scores: Dictionary with emotion names as keys and scores as values.

    Returns:
        A dictionary with emotion names as keys and their z-scores as values.
    """
    scores = list(emotion_scores.values())
    if not scores:
        logger.warning("No emotion scores to compute z-scores")
        return {}
    
    try:
        mean_val = statistics.mean(scores)
        std_val = statistics.stdev(scores)
        if std_val == 0:
            logger.warning("Standard deviation is 0, returning zero z-scores")
            return {emotion: 0 for emotion in emotion_scores}
            
        return {emotion: (score - mean_val) / std_val for emotion, score in emotion_scores.items()}
        
    except statistics.StatisticsError as e:
        logger.warning("Statistics error: single value, returning zero scores")
        # Handle case where there's only one value (can't compute stdev)
        return {emotion: 0 for emotion in emotion_scores}
