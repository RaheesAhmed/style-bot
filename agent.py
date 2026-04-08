"""Core agent logic for the Style-Aware AI Text Generator.

This module contains the main agent class responsible for coordinating
document retrieval, web search, and text generation.
"""

import json
import asyncio
from typing import AsyncGenerator

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

from config import (
    MODEL_NAME,
    EMBED_MODEL,
    VECTOR_DB_PATH,
    TEMPERATURE,
    RETRIEVAL_K,
)
from prompts import create_style_prompt
from tools import format_docs


class StyleAgent:
    """Agent for generating style-aware text content.
    
    This class manages the RAG pipeline, web search integration,
    and streaming text generation.
    
    Attributes:
        embeddings: Ollama embeddings model.
        vector_db: ChromaDB vector database.
        llm: ChatOllama language model.
        search_tool: DuckDuckGo search tool.
        retriever: Vector database retriever.
        prompt_template: ChatPromptTemplate for generation.
    """
    
    def __init__(self):
        """Initialize the StyleAgent with all required components."""
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        
        # Initialize vector database
        self.vector_db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=self.embeddings
        )
        
        # Initialize LLM
        self.llm = ChatOllama(
            model=MODEL_NAME,
            temperature=TEMPERATURE
        )
        
        # Initialize search tool
        self.search_tool = DuckDuckGoSearchRun()
        
        # Initialize retriever
        self.retriever = self.vector_db.as_retriever(
            search_kwargs={"k": RETRIEVAL_K}
        )
        
        # Initialize prompt template
        self.prompt_template = create_style_prompt()
    
    def ingest_documents(self) -> dict:
        """Ingest documents from training and feedback directories.
        
        Returns:
            dict: Status message with ingestion results.
        """
        from tools import load_documents, split_documents
        
        documents = load_documents()
        if not documents:
            return {"message": "No documents found to ingest."}
        
        chunks = split_documents(documents)
        
        # Clear old data and add new chunks
        self.vector_db.reset_collection()
        self.vector_db.add_documents(chunks)
        
        return {"message": f"✅ Successfully ingested {len(chunks)} text chunks."}
    
    async def generate_stream(
        self,
        task: str,
        use_web_search: bool = False
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming response.
        
        Args:
            task: The generation task description.
            use_web_search: Whether to include web research.
            
        Yields:
            str: JSON-formatted SSE data chunks.
        """
        try:
            # Retrieve style examples
            docs = self.retriever.invoke(task)
            context = format_docs(docs)
            
            # Optional web search
            web_context = ""
            if use_web_search:
                try:
                    query = task[:150]
                    results = await asyncio.to_thread(self.search_tool.run, query)
                    web_context = f"\n\nWEB RESEARCH CONTEXT:\n{results}"
                except Exception as e:
                    web_context = f"\n\nWEB RESEARCH CONTEXT:\n[Search failed: {str(e)}]"
            
            # Create and stream chain
            chain = self.prompt_template | self.llm | StrOutputParser()
            
            async for chunk in chain.astream({
                "task": task,
                "context": context,
                "web_search_context": web_context
            }):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                await asyncio.sleep(0.01)  # Yield control to event loop
                
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    def get_health_status(self) -> dict:
        """Get the health status of the agent.
        
        Returns:
            dict: Health status with model information.
        """
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "embed_model": EMBED_MODEL
        }