import os
import json
import glob
import asyncio
import time
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

# ================= CONFIGURATION =================
APP = FastAPI(title="Style-Aware AI Text Generator", version="1.0.0")

MODEL_NAME = "gemma4"               # Ollama model for generation (Gemma 4)
EMBED_MODEL = "nomic-embed-text"    # Ollama model for embeddings
VECTOR_DB_PATH = "./vector_db"
TRAINING_DIR = "./training_data"
FEEDBACK_DIR = "./feedback"

os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# ================= INIT COMPONENTS =================
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
llm = ChatOllama(model=MODEL_NAME, temperature=0.6)
search_tool = DuckDuckGoSearchRun()

# ================= INGESTION LOGIC =================
def load_documents():
    patterns = ["*.[tT][xX][tT]", "*.[mM][dD]"]
    files = []
    for d in [TRAINING_DIR, FEEDBACK_DIR]:
        for pat in patterns:
            files.extend(glob.glob(os.path.join(d, pat)))
    
    docs = []
    for f in files:
        try:
            docs.extend(TextLoader(f, encoding="utf-8").load())
        except Exception as e:
            print(f"⚠️ Failed to load {f}: {e}")
    return docs

def ingest():
    docs = load_documents()
    if not docs:
        return {"message": "No documents found to ingest."}
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    
    vector_db.reset_collection()  # Clears old data
    vector_db.add_documents(chunks)
    return {"message": f"✅ Successfully ingested {len(chunks)} text chunks."}

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n--- EXAMPLE ---\n\n".join([d.page_content for d in docs])

# ================= PROMPT & CHAIN =================
STYLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert writing assistant. Your ONLY job is to generate content that strictly matches the user's unique writing style, tone, structure, and phrasing.
Analyze the provided examples carefully. Mimic their:
- Sentence length & rhythm
- Vocabulary, idioms, and technical depth
- Tone (formal, casual, authoritative, conversational, etc.)
- Formatting, bullet usage, and paragraph structure
NEVER output generic AI text. If web research is provided, weave it naturally into the established style.
Output ONLY the generated content. No intros, no conclusions, no explanations."""),
    ("human", """TASK: {task}
STYLE EXAMPLES FROM ARCHIVE:
{context}
{web_search_context}
GENERATE:""")
])

# ================= STREAMING GENERATOR =================
async def stream_generator(task: str, use_web_search: bool = False):
    try:
        # Retrieve style examples
        docs = retriever.invoke(task)
        context = format_docs(docs) if docs else "[No style examples found. Generating in neutral tone.]"

        # Optional web search
        web_context = ""
        if use_web_search:
            try:
                query = task[:150]
                # Run sync search in thread to avoid blocking FastAPI
                results = await asyncio.to_thread(search_tool.run, query)
                web_context = f"\n\nWEB RESEARCH CONTEXT:\n{results}"
            except Exception as e:
                web_context = f"\n\nWEB RESEARCH CONTEXT:\n[Search failed: {str(e)}]"

        chain = STYLE_PROMPT | llm | StrOutputParser()
        async for chunk in chain.astream({
            "task": task,
            "context": context,
            "web_search_context": web_context
        }):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            await asyncio.sleep(0.01)  # Yield control to event loop

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

# ================= FASTAPI ENDPOINTS =================
@APP.on_event("startup")
async def startup():
    ingest()
    print("🚀 Bot started. Vector DB loaded.")

@APP.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    for file in files:
        if not file.filename.endswith(('.txt', '.md')):
            continue
        content = await file.read()
        path = os.path.join(TRAINING_DIR, file.filename)
        with open(path, "wb") as f:
            f.write(content)
    return ingest()

@APP.post("/generate")
async def generate(
    task: str = Form(..., description="What to generate (e.g., 'Write a product description for...')"),
    use_web_search: bool = Form(False, description="Enable live web research")
):
    return StreamingResponse(
        stream_generator(task, use_web_search),
        media_type="text/event-stream"
    )

@APP.post("/feedback")
async def save_feedback(
    generated_text: str = Form(...),
    rating: int = Form(..., ge=1, le=5, description="1-5 rating. ≥4 auto-improves style")
):
    if rating >= 4:
        filename = os.path.join(FEEDBACK_DIR, f"approved_{int(time.time())}.md")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Style-Approved Output (Rating: {rating}/5)\n\n{generated_text}")
        return {"message": "✅ Saved. High-rated outputs will improve style matching on next /retrain or restart."}
    return {"message": "📝 Feedback recorded. Only 4+ rated outputs auto-train the model."}

@APP.post("/retrain")
async def retrain():
    return ingest()

@APP.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "embed_model": EMBED_MODEL}

# ================= RUN =================
if __name__ == "__main__":
    uvicorn.run(APP, host="0.0.0.0", port=8000, log_level="info")