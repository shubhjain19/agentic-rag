import logging
from typing import Dict, Any, List
from openai import OpenAI
from openai.types.chat import ChatCompletion

from config import Config

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """Client for interacting with OpenRouter API"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """Initialize OpenRouter client"""
        logger.info("Initializing OpenRouter client")
        self.api_key = api_key or Config.OPENROUTER_API_KEY
        self.base_url = base_url or Config.OPENROUTER_BASE_URL
        
        if not self.api_key:
            logger.error("OpenRouter API key is missing")
            raise ValueError("OpenRouter API key is required")
        
        logger.info(f"Using OpenRouter base URL: {self.base_url}")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={
                "HTTP-Referer": "https://agentic-rag-system.com",
                "X-Title": "Agentic RAG System"
            }
        )
        logger.info("OpenRouter client initialized successfully")
    
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         model: str = None,
                         temperature: float = 0.7,
                         max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate response using OpenRouter API"""
        try:
            model = model or Config.LLM_MODEL
            
            logger.info(f"Generating response with model: {model}")
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=10.0
            )
            
            result = {
                "choices": [
                    {
                        "message": {
                            "content": response.choices[0].message.content
                        }
                    }
                ]
            }
            
            logger.info("Response generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def create_rag_prompt(self, query: str, context: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Create a prompt for RAG system"""
        logger.info(f"Creating RAG prompt for query: {query}")
        logger.info(f"No. of context documents: {len(context)}")
    
        context_text = ""
        for i, doc in enumerate(context, 1):
            context_text += f"{i}. {doc['content']}\n"
        
        logger.info(f"Context text length: {len(context_text)} characters")
        
        system_prompt = """You are an e-commerce data analyst. Answer questions concisely based on the provided order data. Be direct and factual."""
        
        user_prompt = f"""Context Data:
{context_text}

User Question: {query}

Please provide a helpful answer based on the context data above."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        logger.info(f"Created prompt with {len(messages)} messages")
        logger.info(f"Total prompt length: {len(str(messages))} characters")
        
        return messages
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        try:
            models = self.client.models.list()
            
            model_list = []
            for model in models.data:
                model_list.append({
                    "id": model.id,
                    "object": model.object,
                    "created": model.created,
                    "owned_by": model.owned_by
                })
            
            return model_list
            
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test the connection to OpenRouter"""
        try:
            models = self.get_available_models()
            if models:
                logger.info(f"OpenRouter connection successful. Found {len(models)} models")
                return True
            else:
                logger.error("No models found")
                return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

if __name__ == "__main__":
    try:
        client = OpenRouterClient()
        
        if client.test_connection():
            print("OpenRouter connection successful!")
            
            messages = [
                {"role": "user", "content": "Hello! Can you help me with e-commerce data analysis?"}
            ]
            
            response = client.generate_response(messages, max_tokens=100)
            print(f"Test response: {response['choices'][0]['message']['content']}")
        else:
            print("OpenRouter connection failed!")
            
    except Exception as e:
        print(f"Error: {e}") 