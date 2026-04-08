"""Configuration constants for the Style-Aware AI Text Generator.

This module contains all configuration constants including model names,
directory paths, and application settings.
"""

import os

# Model Configuration
MODEL_NAME = "gemma4:31b-cloud"
"""str: Ollama model for text generation (Gemma 4)."""

EMBED_MODEL = "nomic-embed-text"
"""str: Ollama model for embeddings."""

# Directory Paths
VECTOR_DB_PATH = "./vector_db"
"""str: Path to ChromaDB vector database storage."""

TRAINING_DIR = "./training_data"
"""str: Directory for training data files."""

FEEDBACK_DIR = "./feedback"
"""str: Directory for high-rated feedback files."""

# Application Settings
APP_TITLE = "Style-Aware AI Text Generator"
"""str: FastAPI application title."""

APP_VERSION = "1.0.0"
"""str: Application version."""

# Model Parameters
TEMPERATURE = 0.6
"""float: LLM temperature for generation."""

CHUNK_SIZE = 500
"""int: Document chunk size for text splitting."""

CHUNK_OVERLAP = 80
"""int: Overlap between document chunks."""

RETRIEVAL_K = 3
"""int: Number of style examples to retrieve."""

# Server Settings
HOST = "0.0.0.0"
"""str: Server host address."""

PORT = 8000
"""int: Server port number."""

LOG_LEVEL = "info"
"""str: Logging level for uvicorn."""


def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [TRAINING_DIR, FEEDBACK_DIR, VECTOR_DB_PATH]:
        os.makedirs(directory, exist_ok=True)