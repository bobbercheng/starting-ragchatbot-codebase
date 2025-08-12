# RAG System Architecture

## Overview
This RAG (Retrieval-Augmented Generation) system enables semantic search and AI-powered responses over course materials using ChromaDB for vector storage and OpenAI GPT-OSS-20B for generation.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAG SYSTEM FLOW                                │
└─────────────────────────────────────────────────────────────────────────┘

1. DOCUMENT INGESTION
┌──────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐
│  Course Files    │───▶│ DocumentProcessor│───▶│    Structured Data      │
│  (.txt/.pdf/     │    │                 │    │                         │
│   .docx)         │    │ - Parse metadata│    │ Course:                 │
│                  │    │ - Split lessons │    │  - title (unique ID)    │
│                  │    │ - Chunk text    │    │  - instructor           │
│                  │    │   (800 chars,   │    │  - course_link          │
│                  │    │    100 overlap) │    │  - lessons[]            │
│                  │    │ - Add context   │    │                         │
│                  │    │   prefixes      │    │ CourseChunks:           │
│                  │    │                 │    │  - content (w/ context) │
│                  │    │                 │    │  - course_title         │
│                  │    │                 │    │  - lesson_number        │
│                  │    │                 │    │  - chunk_index          │
└──────────────────┘    └─────────────────┘    └─────────────────────────┘

2. VECTOR STORAGE (ChromaDB)
┌─────────────────────────────────────────────────────────────────────────┐
│                         VectorStore                                     │
│                                                                         │
│  ┌─────────────────────┐        ┌─────────────────────────────────────┐ │
│  │  course_catalog     │        │         course_content             │ │
│  │  collection         │        │         collection                  │ │
│  │                     │        │                                     │ │
│  │ Documents:          │        │ Documents:                          │ │
│  │  - Course titles    │        │  - Chunked content with context     │ │
│  │                     │        │    "Course X Lesson Y content: ..." │ │
│  │ Metadata:           │        │                                     │ │
│  │  - title            │        │ Metadata:                           │ │
│  │  - instructor       │        │  - course_title                     │ │
│  │  - course_link      │        │  - lesson_number                    │ │
│  │  - lessons_json     │        │  - chunk_index                      │ │
│  │  - lesson_count     │        │                                     │ │
│  │                     │        │ IDs:                                │ │
│  │ IDs:                │        │  - {course_title}_{chunk_index}     │ │
│  │  - Course titles    │        │                                     │ │
│  │                     │        │                                     │ │
│  │ Embeddings:         │        │ Embeddings:                         │ │
│  │  - SentenceTransf.  │        │  - SentenceTransformers             │ │
│  │    all-MiniLM-L6-v2 │        │    all-MiniLM-L6-v2                 │ │
│  └─────────────────────┘        └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘

3. QUERY PROCESSING
┌──────────────┐    ┌─────────────┐    ┌──────────────────────────────────┐
│ User Query   │───▶│ RAGSystem   │───▶│        AI Generator              │
│              │    │             │    │                                  │
│              │    │ - Get conv  │    │ - Format prompt with history     │
│              │    │   history   │    │ - Call OpenAI GPT-OSS-20B        │
│              │    │ - Build     │    │   via LiteLLM proxy              │
│              │    │   tools     │    │ - Handle tool calling            │
│              │    │             │    │ - 5min timeout handling          │
└──────────────┘    └─────────────┘    └──────────────────────────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │   ToolManager   │
                    │                 │
                    │ Available Tools:│
                    │ - CourseSearch  │
                    │ - CourseOutline │
                    │                 │
                    │ Executes search │
                    │ when AI decides │
                    │ it's needed     │
                    └─────────────────┘

4. SEARCH EXECUTION (When AI Calls Tools)
┌─────────────────────────────────────────────────────────────────────────┐
│                      Search Process                                     │
│                                                                         │
│ Query: "Tell me about machine learning concepts in CS101"              │
│                                                                         │
│ ┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐   │
│ │ 1. Course Name  │───▶│ 2. Build Filter  │───▶│ 3. Vector Search  │   │
│ │    Resolution   │    │                  │    │                   │   │
│ │                 │    │ Filter options:  │    │ - Embed query     │   │
│ │ - Search catalog│    │ {                │    │ - Compare with    │   │
│ │   for "CS101"   │    │  "course_title": │    │   chunk embeddings│   │
│ │ - Return exact  │    │   "CS101",       │    │ - Return top N    │   │
│ │   course title  │    │  "lesson_number":│    │   matches         │   │
│ │                 │    │   null           │    │ - Include metadata│   │
│ │                 │    │ }                │    │                   │   │
│ └─────────────────┘    └──────────────────┘    └───────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

5. RESPONSE GENERATION
┌─────────────────────────────────────────────────────────────────────────┐
│                      AI Response Synthesis                              │
│                                                                         │
│ Input to AI:                                                            │
│ - Original user query                                                   │
│ - Conversation history (last 2 exchanges)                              │
│ - Search results from tools (if executed)                              │
│                                                                         │
│ AI Process:                                                             │
│ - Analyze query context                                                 │
│ - Decide if search is needed                                           │
│ - If needed, call search tool with appropriate filters                 │
│ - Synthesize response using retrieved content                          │
│ - Generate natural language answer                                     │
│                                                                         │
│ Output:                                                                 │
│ - Generated response text                                               │
│ - Source links (course/lesson URLs)                                    │
│ - Updated conversation history                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Storage Details

### Vector Database Collections

#### 1. `course_catalog` Collection
**Purpose**: Store course metadata for course name resolution
- **Documents**: Course titles (for semantic matching)
- **Metadata**: 
  ```json
  {
    "title": "CS101: Introduction to Computer Science",
    "instructor": "Dr. Smith", 
    "course_link": "https://example.com/cs101",
    "lessons_json": "[{\"lesson_number\": 1, \"lesson_title\": \"Variables\", \"lesson_link\": \"...\"}]",
    "lesson_count": 12
  }
  ```
- **IDs**: Course titles (used as unique identifiers)
- **Embeddings**: Generated from course titles using SentenceTransformers

#### 2. `course_content` Collection  
**Purpose**: Store actual course content chunks for semantic search
- **Documents**: Text chunks with context prefixes
  ```
  "Course CS101: Introduction to Computer Science Lesson 3 content: Variables are containers for storing data values. In Python, you don't need to declare..."
  ```
- **Metadata**:
  ```json
  {
    "course_title": "CS101: Introduction to Computer Science",
    "lesson_number": 3,
    "chunk_index": 42
  }
  ```
- **IDs**: `{course_title_with_underscores}_{chunk_index}`
- **Embeddings**: Generated from full content using SentenceTransformers

### Embedding Model
- **Model**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **Dimension**: 384 dimensions
- **Use Cases**: 
  - Course title semantic matching
  - Content similarity search
  - Cross-lingual support

### Text Chunking Strategy
- **Chunk Size**: 800 characters
- **Overlap**: 100 characters between chunks
- **Method**: Sentence-aware splitting (preserves sentence boundaries)
- **Context Enhancement**: Each chunk prefixed with course and lesson info

## Query Flow Examples

### Example 1: Direct Question
```
User: "What is a variable in programming?"

1. RAGSystem receives query
2. AIGenerator calls GPT-OSS-20B with tools
3. AI decides search is needed
4. Calls CourseSearchTool with query "variable in programming"
5. VectorStore searches course_content collection
6. Returns chunks about variables from multiple courses
7. AI synthesizes response using retrieved content
8. Returns answer with source links
```

### Example 2: Course-Specific Question
```
User: "Tell me about loops in CS101"

1. RAGSystem receives query  
2. AIGenerator identifies course filter needed
3. Calls CourseSearchTool with query "loops" and course_name "CS101"
4. VectorStore:
   - Resolves "CS101" to exact course title via course_catalog
   - Searches course_content with course_title filter
5. Returns CS101-specific content about loops
6. AI generates response with course context
```

### Example 3: Conversation Context
```
User: "What about functions?" (following previous CS101 question)

1. RAGSystem gets conversation history
2. AI uses context to understand this relates to CS101
3. Searches for "functions" in CS101 course
4. Generates response maintaining conversation flow
```

## Key Architecture Benefits

1. **Semantic Search**: Vector embeddings enable meaning-based retrieval beyond keyword matching
2. **Course Resolution**: Fuzzy course name matching via semantic search in catalog
3. **Contextual Chunking**: Each chunk contains course/lesson context for better AI responses  
4. **Conversation Memory**: Session manager maintains context across exchanges
5. **Tool-Based Architecture**: AI decides when search is needed, preventing unnecessary queries
6. **Timeout Handling**: 5-minute timeouts accommodate slower AI model responses
7. **Source Tracking**: Maintains links to original course/lesson content for user reference

## Configuration Parameters

- `CHUNK_SIZE=800`: Character limit per text chunk
- `CHUNK_OVERLAP=100`: Character overlap between chunks  
- `MAX_RESULTS=5`: Maximum search results returned
- `MAX_HISTORY=2`: Number of conversation exchanges to maintain
- `OPENAI_TIMEOUT=300`: 5-minute timeout for AI responses
- `EMBEDDING_MODEL=all-MiniLM-L6-v2`: Sentence transformer model