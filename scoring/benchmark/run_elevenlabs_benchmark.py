#!/usr/bin/env python3
"""
Runner script for the ElevenLabs TTS benchmarking.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs_score import get_tts_score

# Load environment variables from .env file
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRequest:
    """Simple request object for the benchmark."""
    repo_namespace: str
    repo_name: str
    api_key: str = None


def main():
    """Main entry point for the ElevenLabs benchmark."""
    parser = argparse.ArgumentParser(description="Run ElevenLabs TTS benchmark")
    parser.add_argument("--api-key", help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var)")
    parser.add_argument("--repo-namespace", default="elevenlabs", help="Repository namespace")
    parser.add_argument("--repo-name", default="multilingual-v2", help="Repository name")
    parser.add_argument("--env-file", help="Path to .env file", default=".env")
    
    args = parser.parse_args()
    
    # Load environment variables from specified .env file if it exists
    if args.env_file and Path(args.env_file).exists():
        load_dotenv(args.env_file, override=True)
        logger.info(f"Loaded environment variables from {args.env_file}")
    
    # Check for API key in environment if not provided
    api_key = args.api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        logger.error("ElevenLabs API key not provided. Use --api-key or set ELEVENLABS_API_KEY environment variable.")
        sys.exit(1)
    
    # Set API key in environment for the benchmark
    os.environ["ELEVENLABS_API_KEY"] = api_key
    
    # Check for Whisper endpoint
    if not os.environ.get("WHISPER_ENDPOINT"):
        logger.error("WHISPER_ENDPOINT environment variable not set.")
        sys.exit(1)
        
    # Check for Hume AI key
    if not os.environ.get("HUME_API_KEY"):
        logger.error("HUME_API_KEY environment variable not set.")
        sys.exit(1)
    
    # Create request object
    request = BenchmarkRequest(
        repo_namespace=args.repo_namespace,
        repo_name=args.repo_name,
        api_key=api_key
    )
    
    logger.info(f"Starting ElevenLabs benchmark for {request.repo_namespace}/{request.repo_name}")
    
    try:
        # Run the benchmark
        result = get_tts_score(request)
        
        # Display results
        logger.info("Benchmark completed successfully")
        logger.info(f"Final score: {result['final_score']:.4f}")
        
        if "error" in result:
            logger.warning(f"Errors encountered: {result['error']}")
            
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
