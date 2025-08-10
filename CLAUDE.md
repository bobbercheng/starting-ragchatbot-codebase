# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install all dependencies
uv sync

# Add new dependency
uv add package-name

# Update dependencies
uv lock --upgrade
```

### Development Server
- Web interface: http://localhost:8000
- API docs: http://localhost:8000/docs
- Backend runs on port 8000 with auto-reload

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for querying course materials using a custom implementation without agent frameworks.

### Core Components

**RAGSystem** (`rag_system.py`) - Main orchestrator that coordinates:
- Document processing and chunking
- Vector storage operations  
- AI generation with tool calling
- Session management for conversations

**DocumentProcessor** (`document_processor.py`) - Handles course document parsing:
- Expected format: `Course Title:`, `Course Instructor:`, `Lesson N: Title` markers
- Sentence-based chunking with configurable overlap (800 chars, 100 overlap)
- Context enhancement: prefixes chunks with course/lesson metadata

**VectorStore** (`vector_store.py`) - ChromaDB integration:
- Uses sentence-transformers for embeddings (`all-MiniLM-L6-v2`)
- Supports semantic search with course/lesson filtering
- Stores both course metadata and content chunks separately

**AIGenerator** (`ai_generator.py`) - OpenAI GPT-OSS-20B integration:
- Direct OpenAI client usage via LiteLLM proxy (no LangChain/LlamaIndex)
- Converts Anthropic tool format to OpenAI function calling format
- 5-minute timeout handling for slower model responses
- Conversation history management

**Tool System** (`search_tools.py`) - Custom tool implementation:
- Abstract `Tool` base class for extensibility
- `CourseSearchTool` - semantic search with optional filtering
- `ToolManager` - registers and orchestrates tool execution

### Request Flow

1. **Frontend** (`script.js`) sends query to `/api/query`
2. **FastAPI** (`app.py`) routes to RAG system
3. **RAG System** builds prompt with conversation history  
4. **AI Generator** calls GPT-OSS-20B with available tools (via LiteLLM proxy)
5. **GPT-OSS-20B decides** whether to search based on query type
6. **If search needed**: Tool system executes semantic search via vector store  
7. **Response assembly**: GPT-OSS-20B generates final answer, sources tracked separately
8. **Session update**: Conversation history maintained for context

### Configuration

All settings in `config.py` with environment variable overrides:
- `OPENAI_API_KEY` - Required for GPT-OSS-20B access via LiteLLM proxy (default: sk-6y2nvIEFmgz1HaHkO9DnYA)
- `OPENAI_BASE_URL` - LiteLLM proxy URL (default: http://localhost:4000) 
- `OPENAI_MODEL` - Model name (openai/gpt-oss-20b)
- `OPENAI_TIMEOUT` - Request timeout in seconds (default: 300.0 / 5 minutes)
- `CHUNK_SIZE=800` - Text chunk size for vector storage
- `MAX_RESULTS=5` - Search results limit
- `MAX_HISTORY=2` - Conversation context depth

### Data Models

**Course**: Title (unique ID), instructor, lessons list, optional course link
**Lesson**: Number, title, optional lesson link  
**CourseChunk**: Content, course title, lesson number, chunk index

### Document Processing Pipeline

1. **Parse metadata** from first 3-4 lines (title, instructor, course link)
2. **Segment by lessons** using `Lesson N:` regex markers
3. **Chunk content** with sentence-aware splitting and overlap
4. **Add context** to chunks: `"Course {title} Lesson {N} content: {chunk}"`
5. **Store in vector DB** with metadata for filtering

### Tool Calling Architecture

Uses OpenAI function calling converted from Anthropic format:
- Single tool: `search_course_content` with optional course/lesson filters  
- One search per query maximum to prevent loops
- Anthropic tool definitions automatically converted to OpenAI function format
- Sources tracked separately from AI response for UI display
- 5-minute timeout handling for slower GPT-OSS-20B responses

## Environment Setup

Create `.env` file in root:
```
OPENAI_API_KEY=sk-6y2nvIEFmgz1HaHkO9DnYA
OPENAI_BASE_URL=http://localhost:4000
OPENAI_TIMEOUT=300.0
```

**Prerequisites**: Ensure LiteLLM proxy server is running:
```bash
litellm --model openai/gpt-oss-20b --port 4000
```

## Document Format

Place course documents in `docs/` folder with this structure:
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

Supports `.txt`, `.pdf`, `.docx` files.
- always use uv to run the server or install python package do not use pip directly
- use uv to run Python files/code