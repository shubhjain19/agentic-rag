import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration for the RAG system"""
    
    # OpenRouter API Configuration
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    
    # Meilisearch Configuration
    MEILISEARCH_URL = os.getenv("MEILISEARCH_URL", "http://localhost:7700")
    MEILISEARCH_MASTER_KEY = os.getenv("MEILISEARCH_MASTER_KEY", "")
    
    # Application Configuration
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # RAG Configuration
    INDEX_NAME = "ecommerce_orders"
    MAX_SEARCH_RESULTS = 5  # Reduced for faster processing
    LLM_MODEL = "anthropic/claude-3-haiku"  # Faster model
    
    # Data Configuration
    DATA_FILE = "data/Order Details.csv"
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present"""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is required. Please set it in your .env file. "
                "Get your API key from: https://openrouter.ai/keys"
            )
        
        return True 