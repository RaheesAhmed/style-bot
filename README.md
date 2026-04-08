# Style-Aware AI Text Generator

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-orange.svg)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A bot that learns your writing style, generates human-like content via real-time streaming, and improves over time through a feedback loop.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API Reference](#-api-reference) • [How It Works](#-how-it-works)

</div>

---

## 📋 Overview

This project creates a personalized AI writing assistant that:
- **Learns your unique writing style** from your documents
- **Generates content in your voice** using RAG-powered retrieval
- **Streams responses in real-time** for low-latency output
- **Improves continuously** through user feedback
- **Optionally integrates web research** into your established style

Built with modern AI/ML technologies: **FastAPI**, **LangChain**, **Ollama**, **ChromaDB**, and managed with **uv** for blazing-fast dependency management.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Style Mimicry** | RAG-powered retrieval of your best writing samples with strict prompt enforcement |
| ⚡ **Token Streaming** | Server-Sent Events (SSE) for real-time, low-latency output |
| 🔍 **Live Web Research** | Optional DuckDuckGo search woven into your established style |
| 📈 **Continuous Learning** | 4+ star ratings auto-save to `./feedback` and retrain the style baseline |
| 🐍 **Modern Stack** | FastAPI + LangChain + uv dependency management + Chroma vector cache |
| 🔒 **Privacy-First** | Runs entirely locally with Ollama - no data leaves your machine |

---

## 📦 Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | ≥3.11 | Runtime environment |
| uv | Latest | Blazing-fast package manager |
| Ollama | Latest | Local LLM & embedding engine |

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/raheesahmed/style-bot.git
cd style-bot
```

### Step 2: Install Dependencies with uv

Using `uv` (recommended - 10-100x faster than pip):

```bash
# Install uv if you haven't already
pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
uv add fastapi "uvicorn[standard]" langchain langchain-ollama langchain-chroma langchain-community duckduckgo-search python-multipart chromadb
```

Or sync from `pyproject.toml`:

```bash
uv sync
```

### Step 3: Setup Ollama Models

```bash
# Pull required models
ollama pull gemma4               # Gemma 4 for text generation
ollama pull nomic-embed-text     # Style embedding & retrieval

# Ensure Ollama is running
ollama serve
```

**Available Gemma 4 variants:**
- `gemma4:e2b` - Efficient 2B parameter model (lightweight)
- `gemma4:e4b` - Efficient 4B parameter model (balanced)
- `gemma4:26b` - Full 26B parameter model (most capable)
- `gemma4:31b` - Large 31B parameter model (frontier performance)

Use `ollama run gemma4:26b` for the best balance of performance and efficiency.

### Step 4: Run the Server

```bash
# Using uv (recommended)
uv run python main.py

# Or directly
python main.py
```

✅ **Server starts at:** `http://localhost:8000`  
📖 **Interactive API Docs:** `http://localhost:8000/docs`  
📚 **Alternative Docs:** `http://localhost:8000/redoc`

---

## 📖 Usage

### 1. Upload Training Files

Upload your writing samples to teach the bot your style:

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@my_blog_posts.txt" \
  -F "files=@email_templates.md"
```

📁 **Supported formats:** `.txt` and `.md` files  
📂 **Storage location:** `./training_data/` (auto-indexed)

### 2. Generate Streaming Content

Generate content in your style with real-time streaming:

```bash
curl -N -X POST http://localhost:8000/generate \
  -F "task=Write a 3-part LinkedIn carousel about AI workflow automation" \
  -F "use_web_search=true"
```

🔁 **Output format:** Streams JSON lines: `{"chunk": "text"}\n\n`  
💡 **Tip:** Use `-N` flag with curl to disable buffering

### 3. Submit Feedback (Improves Style)

Rate generated content to improve future outputs:

```bash
curl -X POST http://localhost:8000/feedback \
  -F "generated_text=Full AI output here..." \
  -F "rating=5"
```

⭐ **Auto-save:** Ratings ≥4 save to `./feedback/` and merge into your style archive

### 4. Force Retraining

Manually trigger style retraining:

```bash
curl -X POST http://localhost:8000/retrain
```

### 5. Health Check

Verify the server status:

```bash
curl http://localhost:8000/health
# → {"status": "ok", "model": "gemma4", "embed_model": "nomic-embed-text"}
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload training files (.txt, .md) |
| `/generate` | POST | Generate streaming content with style mimicry |
| `/feedback` | POST | Submit rating for generated content |
| `/retrain` | POST | Force retrain the style model |
| `/health` | GET | Check server health status |

### Detailed API Documentation

Access the interactive API documentation at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## 🔍 How It Works

### Style Adaptation Pipeline

```
┌─────────────────┐
│  Upload Files   │  → .txt/.md files saved to ./training_data/
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Ingestion     │  → Split into chunks, embedded with nomic-embed-text
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ChromaDB       │  → Vector storage for semantic retrieval
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Generation    │  → Top-3 style examples retrieved + LLM generation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Feedback      │  → High-rated outputs saved to ./feedback/
└─────────────────┘
```

### Technical Details

1. **Ingestion**: `.txt`/`.md` files are split into 500-character chunks with 80-character overlap, embedded with `nomic-embed-text`, and stored in `ChromaDB`.

2. **Retrieval**: On each request, the top-3 most relevant style examples are fetched using semantic similarity.

3. **Prompt Enforcement**: A strict system prompt forces the LLM to mimic:
   - Sentence rhythm & length
   - Vocabulary, tone, & technical depth
   - Formatting, bullets, & paragraph structure

4. **Feedback Loop**: High-rated outputs (≥4 stars) are saved to `./feedback/` and re-indexed, gradually shifting the style baseline toward your preferred voice.

---

## 📁 Project Structure

```
style-bot/
├── main.py                 # FastAPI application entry point
├── pyproject.toml          # Project dependencies (uv format)
├── README.md               # This file
├── stream.js               # Frontend streaming handler (optional)
├── training_data/          # Your writing samples (.txt, .md)
├── feedback/               # High-rated generated content
└── vector_db/              # ChromaDB persistent storage
```

---

## ⚙️ Configuration

Key configuration variables in `main.py`:

```python
MODEL_NAME = "gemma4"              # Ollama model for generation (Gemma 4)
EMBED_MODEL = "nomic-embed-text"   # Ollama model for embeddings
VECTOR_DB_PATH = "./vector_db"     # ChromaDB storage path
TRAINING_DIR = "./training_data"   # Training files directory
FEEDBACK_DIR = "./feedback"        # Feedback files directory
```

---

## 🛠️ Development

### Run in Development Mode

```bash
# Install development dependencies
uv add --dev pytest pytest-asyncio httpx

# Run tests
uv run pytest

# Run with auto-reload
uv run uvicorn main:APP --reload
```



## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) - Framework for LLM applications
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
- [Gemma 4](https://ollama.com/library/gemma4) - Google's frontier-level language model

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

<div align="center">

**Made with ❤️ by Rahees Ahmed**

</div>
5. **Web Search (Optional)**: Live research is injected but **always filtered through your style prompt** to prevent generic AI drift.

---

## 📂 Project Structure
```
style-ai-bot/
├── bot.py                  # FastAPI app + LangChain pipeline
├── training_data/          # Your original .txt/.md samples
├── feedback/               # Auto-saved 4+ rated outputs
├── vector_db/              # ChromaDB embeddings cache (auto-created)
├── pyproject.toml          # uv project config & dependencies
└── README.md
```

---

## ⚙️ Customization
| Change | How To |
|--------|--------|
| **Switch LLM** | Edit `MODEL_NAME = "qwen2.5:14b"` in `bot.py` |
| **Change Chunk Size** | Adjust `chunk_size=500, chunk_overlap=80` in `ingest()` |
| **Add Auth** | Use FastAPI middleware or `fastapi-users` for API keys |
| **Multi-Project Styles** | Add `metadata={"project": "brand_a"}` to Chroma docs & filter retriever |
| **Production Deploy** | Wrap in Docker + Gunicorn/Uvicorn workers + reverse proxy |

---

## 🐛 Troubleshooting
| Issue | Fix |
|-------|-----|
| `ConnectionRefusedError: Ollama` | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `ollama pull llama3` and `ollama pull nomic-embed-text` |
| `Upload fails with 422` | Ensure `python-multipart` is installed (`uv add python-multipart`) |
| Slow streaming | Lower `temperature` to `0.4`, increase chunk overlap, or use `qwen2.5:7b` |
| Style feels generic | Add more diverse samples to `./training_data/`, use `rating=5` feedback |

---

## 📜 License
MIT — Free for personal & commercial use. Modify, extend, and deploy as needed.

---

💡 **Pro Tip**: Start with 5-10 high-quality `.md` files that represent your *best* writing. The bot's style fidelity scales directly with training data quality + feedback loop usage.

