import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # Model selection flag - "anthropic" or "openai"
    MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "anthropic")
    
    # Anthropic API settings (via LiteLLM proxy)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_BASE_URL: str = os.getenv("ANTHROPIC_BASE_URL", "http://localhost:4000")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "anthropic/claude-sonnet-4-20250514") 
    ANTHROPIC_TIMEOUT: float = float(os.getenv("ANTHROPIC_TIMEOUT", "300.0"))
    
    # OpenAI API settings (for GPT-OSS-20B via LiteLLM proxy)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "http://localhost:4000")
    # OPENAI_MODEL: str = "gpt-oss-20b"
    # The best local model I have tested so far
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "qwen3-4b") #"qwen3-4b"
    # OPENAI_MODEL: str = "deepseek-r1-0528-qwen3-8b"
    OPENAI_TIMEOUT: float = float(os.getenv("OPENAI_TIMEOUT", "900.0"))  # 15 minutes for very slow GPT-OSS-20B
    
    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember
    
    # AI Response settings
    ENABLE_SYNTHESIS_FALLBACK: bool = os.getenv("ENABLE_SYNTHESIS_FALLBACK", "true").lower() == "true"
    SKIP_SYNTHESIS_FOR_ANTHROPIC: bool = os.getenv("SKIP_SYNTHESIS_FOR_ANTHROPIC", "false").lower() == "true"
    
    # Sequential Tool Calling settings
    MAX_TOOL_ROUNDS: int = int(os.getenv("MAX_TOOL_ROUNDS", "2"))
    ENABLE_SEQUENTIAL_TOOLS: bool = os.getenv("ENABLE_SEQUENTIAL_TOOLS", "true").lower() == "true"
    
    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

config = Config()


