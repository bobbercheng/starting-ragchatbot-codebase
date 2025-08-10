# RAG System Query Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Frontend<br/>(script.js)
    participant API as FastAPI<br/>(app.py)
    participant RAG as RAG System<br/>(rag_system.py)
    participant Session as Session Manager
    participant AI as AI Generator<br/>(ai_generator.py)
    participant Claude as Claude API
    participant Tools as Tool Manager<br/>(search_tools.py)
    participant Vector as Vector Store<br/>(ChromaDB)

    User->>Frontend: Types query & clicks send
    Frontend->>Frontend: addMessage(query, 'user')
    Frontend->>Frontend: createLoadingMessage()
    
    Frontend->>+API: POST /api/query<br/>{query, session_id}
    
    API->>+RAG: query(query, session_id)
    
    RAG->>Session: get_conversation_history(session_id)
    Session-->>RAG: previous_messages
    
    RAG->>+AI: generate_response(query, history, tools, tool_manager)
    
    AI->>+Claude: messages.create()<br/>with tools & system prompt
    
    Note over Claude: Claude analyzes query<br/>Decides: search needed?
    
    alt Query needs search
        Claude-->>-AI: tool_use response<br/>("search_course_content")
        
        AI->>+Tools: execute_tool("search_course_content",<br/>query, course_name, lesson_number)
        
        Tools->>+Vector: search(query, filters)
        Vector->>Vector: Semantic search<br/>+ filtering
        Vector-->>-Tools: search_results
        
        Tools->>Tools: format_results()<br/>track sources
        Tools-->>-AI: formatted_results
        
        AI->>+Claude: messages.create()<br/>with tool_results
        Claude-->>-AI: final_response
    else General knowledge
        Claude-->>-AI: direct_response
    end
    
    AI-->>-RAG: response_text
    
    RAG->>Tools: get_last_sources()
    Tools-->>RAG: sources[]
    
    RAG->>Session: add_exchange(session_id, query, response)
    
    RAG-->>-API: (response, sources)
    
    API-->>-Frontend: QueryResponse<br/>{answer, sources, session_id}
    
    Frontend->>Frontend: loadingMessage.remove()
    Frontend->>Frontend: addMessage(answer, 'assistant', sources)
    Frontend-->>User: Display response + sources
```

## Flow Breakdown

### 1. **Frontend Layer**
- User interaction handling
- Loading states management  
- HTTP API calls to backend

### 2. **API Layer** 
- FastAPI endpoint routing
- Request/response serialization
- Session management

### 3. **RAG Orchestration**
- Coordinates all components
- Manages conversation history
- Handles response assembly

### 4. **AI Generation**
- Claude API integration
- Tool calling orchestration
- Multi-turn conversation handling

### 5. **Tool System**
- Abstract tool interface
- Search tool execution
- Source tracking for UI

### 6. **Vector Storage**
- Semantic search execution
- Course/lesson filtering
- Document retrieval

## Key Decision Points

- **Tool Usage**: Claude decides whether to search based on query type
- **Search Strategy**: Vector search with optional course/lesson filters
- **Response Assembly**: Final response + separate source tracking
- **Session Continuity**: Conversation history maintained across queries

## Data Flow

```
User Query → Frontend → API → RAG System → AI Generator → Claude API
                ↓                              ↓
            UI Update ← API Response ← Sources ← Tool Execution → Vector Search
```