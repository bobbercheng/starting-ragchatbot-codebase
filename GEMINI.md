# Course Materials RAG System - Gemini Context

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses. The application is a full-stack web application built with Python/FastAPI backend and a modern JavaScript frontend.

### Main Technologies
- **Backend**: Python 3.13+, FastAPI, ChromaDB, OpenAI API
- **Frontend**: HTML/CSS/JavaScript with modern UI components
- **AI Models**: GPT-OSS-20B via LiteLLM proxy (default), with Anthropic Claude support
- **Vector Storage**: ChromaDB with sentence-transformers embeddings
- **Package Management**: uv (Python package manager)

### Architecture
The system follows a clean architecture with these main components:
1. **Frontend**: User interface for querying course materials
2. **API Layer**: FastAPI endpoints for query processing and course analytics
3. **RAG System**: Main orchestrator coordinating document processing, vector storage, and AI generation
4. **AI Generator**: Handles interactions with AI models via LiteLLM proxy
5. **Vector Store**: ChromaDB integration for semantic search capabilities
6. **Document Processor**: Parses course documents and creates searchable chunks
7. **Tool System**: Custom tool implementation for course search and outline retrieval

## Building and Running

### Prerequisites
- Python 3.13 or higher
- uv (Python package manager)
- LiteLLM proxy server running on port 4000

### Installation
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
uv sync
```

### Configuration
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=sk-6y2nvIEFmgz1HaHkO9DnYA
OPENAI_BASE_URL=http://localhost:4000
OPENAI_TIMEOUT=300.0
```

### Starting the Application
```bash
# Quick start (recommended)
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

### LiteLLM Proxy Setup
The system requires a LiteLLM proxy server to be running:
```bash
litellm --model openai/gpt-oss-20b --port 4000
```

## Development Conventions

### Code Structure
- **Backend**: All Python code in `backend/` directory
- **Frontend**: Static files in `frontend/` directory
- **Documents**: Course materials in `docs/` directory
- **Tests**: Test files in root directory (e.g., `test_openai_integration.py`)

### Python Conventions
- Uses modern Python 3.13+ features
- Type hints for all function signatures
- Pydantic models for data validation
- uv for package management (not pip directly)
- FastAPI for REST API implementation

### Document Processing
Course documents should follow this format:
```
Course Title: Your Course Name
Course Link: https://example.com (optional)
Course Instructor: Instructor Name (optional)

Lesson 0: Introduction
Lesson Link: https://example.com/lesson0 (optional)
Content here...

Lesson 1: Next Topic  
Content here...
```

Supported file formats: `.txt`, `.pdf`, `.docx`

### AI Tool Architecture
The system uses a custom tool implementation without agent frameworks:
- **CourseSearchTool**: Semantic search with optional course/lesson filtering
- **CourseOutlineTool**: Retrieves course structure and lesson information
- Tools are automatically converted from Anthropic format to OpenAI function calling format

### Vector Storage
- Uses ChromaDB for persistent vector storage
- Sentence transformers for text embeddings (`all-MiniLM-L6-v2`)
- Stores both course metadata and content chunks separately
- Supports semantic search with course/lesson filtering

## Key Files and Directories

### Root Directory
- `run.sh`: Main startup script
- `pyproject.toml`: Project dependencies
- `README.md`: Project documentation
- `CLAUDE.md`: Development guidance (also relevant for Gemini)
- `GEMINI.md`: This file

### Backend Directory (`backend/`)
- `app.py`: FastAPI application with endpoints
- `rag_system.py`: Main RAG orchestrator
- `ai_generator.py`: AI model integration
- `vector_store.py`: ChromaDB integration
- `document_processor.py`: Course document parsing
- `search_tools.py`: Custom tool implementation
- `config.py`: Configuration management
- `models.py`: Pydantic data models

### Frontend Directory (`frontend/`)
- `index.html`: Main HTML interface
- `script.js`: Client-side JavaScript
- `style.css`: Modern dark-themed styling

### Documentation
- `query_flow_diagram.md`: System architecture diagram
- `CLAUDE.md`: Development commands and architecture overview