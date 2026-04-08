"""Utility tools and helper functions for the Style-Aware AI Text Generator.

This module contains document loading, text processing, and other utility

"""

import os
import glob
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import (
    TRAINING_DIR,
    FEEDBACK_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_documents() -> List[Document]:
    """Load all documents from training and feedback directories.
    
    Searches for .txt and .md files in both TRAINING_DIR and FEEDBACK_DIR.
    
    Returns:
        List[Document]: List of loaded documents.
        
    Raises:
        Exception: If a file fails to load, prints warning and continues.
    """
    patterns = ["*.[tT][xX][tT]", "*.[mM][dD]"]
    files = []
    
    for directory in [TRAINING_DIR, FEEDBACK_DIR]:
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(directory, pattern)))
    
    documents = []
    for file_path in files:
        try:
            documents.extend(TextLoader(file_path, encoding="utf-8").load())
        except Exception as e:
            print(f"⚠️ Failed to load {file_path}: {e}")
    
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of documents to split.
        
    Returns:
        List[Document]: List of document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def format_docs(documents: List[Document]) -> str:
    """Format retrieved documents into a single context string.
    
    Args:
        documents: List of retrieved documents.
        
    Returns:
        str: Formatted context string with document separators.
    """
    if not documents:
        return "[No style examples found. Generating in neutral tone.]"
    
    return "\n\n--- EXAMPLE ---\n\n".join([doc.page_content for doc in documents])


def save_feedback_file(content: str, rating: int, output_dir: str) -> str:
    """Save high-rated feedback to a file.
    
    Args:
        content: The generated text content.
        rating: User rating (1-5).
        output_dir: Directory to save the feedback file.
        
    Returns:
        str: Path to the saved feedback file.
    """
    import time
    
    filename = os.path.join(output_dir, f"approved_{int(time.time())}.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Style-Approved Output (Rating: {rating}/5)\n\n{content}")
    
    return filename