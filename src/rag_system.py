import logging
import time
from typing import Dict, Any, List
import json

from config import Config
from data_loader import EcommerceDataLoader
from meilisearch_client import MeilisearchClient
from openrouter_client import OpenRouterClient

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(message)s' 
)
logger = logging.getLogger(__name__)

# Reduce verbosity of other loggers
logging.getLogger('meilisearch_client').setLevel(logging.WARNING)
logging.getLogger('data_loader').setLevel(logging.WARNING)
logging.getLogger('openrouter_client').setLevel(logging.WARNING)

class AgenticRAGSystem:
    """Main RAG system for e-commerce data analysis"""
    
    def __init__(self):
        """Initialize the RAG system"""
        self.data_loader = EcommerceDataLoader()
        self.meilisearch_client = MeilisearchClient()
        self.openrouter_client = OpenRouterClient()
        
        Config.validate_config()
    
    def _smart_search(self, query: str, max_results: int, filters: str = None) -> Dict[str, Any]:
        """Smart search with multiple strategies to find relevant documents"""
        query_lower = query.lower()
        
        # Direct search
        results = self.meilisearch_client.search(query, max_results, filters)
        if results['hits']:
            return results
        
        # Handle top queries specifically
        if 'top' in query_lower:
            # For top queries, search for terms that actually exist in the data
            top_search_terms = ['order', 'item', 'amount', 'profit', 'Furniture', 'Electronics', 'Clothing']
            for term in top_search_terms:
                results = self.meilisearch_client.search(term, max_results * 2, filters)
                if results['hits']:
                    return results
        
        # Extract key terms and search
        key_terms = self._extract_key_terms(query)
        for term in key_terms:
            if len(term) > 2: 
                results = self.meilisearch_client.search(term, max_results, filters)
                if results['hits']:
                    return results
        
        # Search by category if mentioned
        categories = ['Electronics', 'Furniture', 'Clothing']
        for category in categories:
            if category.lower() in query_lower:
                results = self.meilisearch_client.search(category, max_results, filters)
                if results['hits']:
                    return results
        
        # Search for common business terms that actually exist in the data
        business_terms = ['profit', 'amount', 'order', 'item', 'Furniture', 'Electronics', 'Clothing']
        for term in business_terms:
            if term.lower() in query_lower:
                results = self.meilisearch_client.search(term, max_results, filters)
                if results['hits']:
                    return results
        
        # If nothing found, try to get some sample data
        if not results['hits']:
            results = self.meilisearch_client.search("*", max_results, filters)
        
        return results
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for better search"""
        terms = query.split()
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'which', 'where', 'when', 'why', 'how', 'show', 'me', 'find', 'get', 'top', 'best', 'highest', 'lowest', 'average', 'total', 'orders', 'products', 'items'}
        key_terms = [term.lower() for term in terms if term.lower() not in stop_words and len(term) > 2]
        return key_terms
    
    def setup_index(self) -> bool:
        """Set up the Meilisearch index with data"""
        try:
            print("Initializing system...")
            
            # Check if Meilisearch is running
            if not self.meilisearch_client.health_check():
                logger.error("Meilisearch health check failed")
                print("Error: Meilisearch is not running. Please start it first.")
                return False

            self.meilisearch_client.get_or_create_index()
            self.meilisearch_client.configure_search_settings()
            
            stats = self.meilisearch_client.get_index_stats()
            
            if stats.get('numberOfDocuments', 0) > 0:
                logger.info("Index already contains documents, skipping data loading")
                return True
            
            print("Loading data...")
            documents = self.data_loader.process_data()
            
            logger.info(f"Adding {len(documents)} documents to index")
            self.meilisearch_client.add_documents(documents)
            
            print("System ready!")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            print(f"Error: {e}")
            raise
    
    def query(self, 
              user_query: str, 
              max_results: int = None,
              filters: str = None,
              model: str = None) -> Dict[str, Any]:
        """Process a user query using RAG"""
        try:
            start_time = time.time()
            logger.info(f"Processing query: '{user_query}'")
            
            # Search for relevant documents with improved search
            search_results = self._smart_search(user_query, max_results or Config.MAX_SEARCH_RESULTS, filters)
            
            search_time = time.time() - start_time
            
            if not search_results['hits']:
                logger.warning("No relevant documents found for query")
                return {
                    "query": user_query,
                    "answer": "I couldn't find any relevant data to answer your question. Please try rephrasing your query.",
                    "context": [],
                    "search_time": search_time,
                    "llm_time": 0.0,
                    "total_time": time.time() - start_time,
                    "sources": [],
                    "search_stats": {
                        "total_hits": 0,
                        "processing_time_ms": 0
                    }
                }
            
            # Create RAG prompt with context
            context_docs = search_results['hits']
            messages = self.openrouter_client.create_rag_prompt(user_query, context_docs)
            
            # Generate response using LLM
            llm_start_time = time.time()
            llm_response = self.openrouter_client.generate_response(
                messages=messages,
                model=model,
                temperature=0.3,  # Lower temperature for more focused responses
                max_tokens=800
            )
            
            llm_time = time.time() - llm_start_time
            total_time = time.time() - start_time
            
            # Extract the response text
            answer = llm_response['choices'][0]['message']['content']
            
            # Prepare sources
            sources = []
            for doc in context_docs:
                sources.append({
                    "order_id": doc.get("order_id"),
                    "category": doc.get("category"),
                    "sub_category": doc.get("sub_category"),
                    "amount": doc.get("amount"),
                    "profit": doc.get("profit"),
                    "content": doc.get("content")
                })
            
            result = {
                "query": user_query,
                "answer": answer,
                "context": context_docs,
                "search_time": search_time,
                "llm_time": llm_time,
                "total_time": total_time,
                "sources": sources,
                "search_stats": {
                    "total_hits": search_results.get('estimatedTotalHits', 0),
                    "processing_time_ms": search_results.get('processingTimeMs', 0)
                }
            }
            
            logger.info(f"Query completed in {total_time:.2f}s (search: {search_time:.2f}s, LLM: {llm_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and status"""
        try:
            info = {
                "system": "Agentic RAG System",
                "version": "1.0.0",
                "components": {
                    "meilisearch": {
                        "status": "healthy" if self.meilisearch_client.health_check() else "unhealthy",
                        "url": self.meilisearch_client.url
                    },
                    "openrouter": {
                        "status": "healthy" if self.openrouter_client.test_connection() else "unhealthy",
                        "model": Config.LLM_MODEL
                    }
                }
            }
            
            try:
                stats = self.meilisearch_client.get_index_stats()
                info["index_stats"] = stats
            except:
                info["index_stats"] = "Not available"
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    import sys
    
    try:
        rag = AgenticRAGSystem()
        
        if not rag.setup_index():
            print("System initialization failed. Exiting.")
            sys.exit(1)
        
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
            print(f"\nQuery: {query}")
            
            result = rag.query(query)
            print(f"\n{result['answer']}")
            print(f"Response time: {result['total_time']:.1f}s")
            
        else:
            print("\nE-commerce Data Assistant")
            print("Ask me about orders, products, sales, and more!")
            print("Type 'quit' to exit\n")
            
            while True:
                try:
                    query = input("Query: ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        print("Goodbye!")
                        break
                    if not query:
                        continue
                    
                    result = rag.query(query)
                    print(f"\n{result['answer']}")
                    print(f"Response time: {result['total_time']:.1f}s")
                    print()
                    
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    
    except Exception as e:
        print(f"System initialization failed: {e}")
        sys.exit(1) 