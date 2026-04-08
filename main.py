"""FastAPI application for the Style-Aware AI Text Generator.

This module defines the REST API endpoints and server configuration.
"""

import os
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import uvicorn

from config import (
    APP_TITLE,
    APP_VERSION,
    TRAINING_DIR,
    FEEDBACK_DIR,
    HOST,
    PORT,
    LOG_LEVEL,
    ensure_directories,
)
from agent import StyleAgent
from tools import save_feedback_file


# Initialize application
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# Initialize agent
agent = StyleAgent()


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    ensure_directories()
    agent.ingest_documents()
    print("🚀 Bot started. Vector DB loaded.")


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload training files (.txt or .md).
    
    Args:
        files: List of files to upload.
        
    Returns:
        dict: Ingestion status message.
    """
    for file in files:
        if not file.filename.endswith(('.txt', '.md')):
            continue
        
        content = await file.read()
        path = os.path.join(TRAINING_DIR, file.filename)
        
        with open(path, "wb") as f:
            f.write(content)
    
    return agent.ingest_documents()


@app.post("/generate")
async def generate(
    task: str = Form(
        ...,
        description="What to generate (e.g., 'Write a product description for...')"
    ),
    use_web_search: bool = Form(
        False,
        description="Enable live web research"
    )
):
    """Generate style-aware content with streaming response.
    
    Args:
        task: The generation task description.
        use_web_search: Whether to include web research.
        
    Returns:
        StreamingResponse: SSE stream of generated content.
    """
    return StreamingResponse(
        agent.generate_stream(task, use_web_search),
        media_type="text/event-stream"
    )


@app.post("/feedback")
async def save_feedback(
    generated_text: str = Form(...),
    rating: int = Form(
        ...,
        ge=1,
        le=5,
        description="1-5 rating. ≥4 auto-improves style"
    )
):
    """Save feedback for generated content.
    
    Args:
        generated_text: The generated text content.
        rating: User rating from 1 to 5.
        
    Returns:
        dict: Feedback status message.
    """
    if rating >= 4:
        filename = save_feedback_file(generated_text, rating, FEEDBACK_DIR)
        return {
            "message": "✅ Saved. High-rated outputs will improve style matching on next /retrain or restart."
        }
    
    return {"message": "📝 Feedback recorded. Only 4+ rated outputs auto-train the model."}


@app.post("/retrain")
async def retrain():
    """Force retraining of the style model.
    
    Returns:
        dict: Retraining status message.
    """
    return agent.ingest_documents()


@app.get("/health")
def health():
    """Get health status of the application.
    
    Returns:
        dict: Health status with model information.
    """
    return agent.get_health_status()


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL)