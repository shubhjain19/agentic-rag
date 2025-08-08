
import meilisearch
import logging
from typing import List, Dict, Any
import json

from config import Config
logger = logging.getLogger(__name__)

class MeilisearchClient:
    """Client for interacting with Meilisearch"""
    
    def __init__(self, url: str = None, master_key: str = None):
        """Initialize Meilisearch client"""
        logger.info("Initializing Meilisearch client")
        self.url = url or Config.MEILISEARCH_URL
        self.master_key = master_key or Config.MEILISEARCH_MASTER_KEY

        logger.info(f"Connecting to Meilisearch at: {self.url}")
        if self.master_key:
            logger.info("Using master key for authentication")
            self.client = meilisearch.Client(self.url, self.master_key)
        else:
            logger.info("Connecting without master key")
            self.client = meilisearch.Client(self.url)
        
        self.index_name = Config.INDEX_NAME
        logger.info(f"Using index name: {self.index_name}")
        self.index = None
        logger.info("Meilisearch client initialized successfully")
        
    def create_index(self) -> bool:
        """Create the index if it doesn't exist"""
        try:
            logger.info(f"Creating index: {self.index_name}")
            
            self.client.create_index(
                uid=self.index_name,
                options={
                    'primaryKey': 'id'
                }
            )
            
            self.index = self.client.index(self.index_name)
            
            logger.info(f"Index '{self.index_name}' created successfully")
            return True
            
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Index '{self.index_name}' already exists")
                self.index = self.client.index(self.index_name)
                return True
            else:
                logger.error(f"Error creating index: {e}")
                raise
    
    def delete_index(self) -> bool:
        """Delete the index if it exists"""
        try:
            logger.info(f"Deleting index: {self.index_name}")
            self.client.delete_index(self.index_name)
            logger.info(f"Index '{self.index_name}' deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            return False
    
    def get_or_create_index(self):
        """Get existing index or create new one"""
        try:
            self.index = self.client.index(self.index_name)
        except Exception:
            self.create_index()
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the index"""
        try:
            logger.info(f"Adding {len(documents)} documents to index")
            
            if not self.index:
                self.get_or_create_index()
            
            batch_size = 100 
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.index.add_documents(batch)
                import time
                time.sleep(0.1)
            
            logger.info("All batches added, waiting for indexing to complete")
        
            import time
            max_wait = 30 
            wait_time = 0
            while wait_time < max_wait:
                stats = self.get_index_stats()
                if not stats.get('isIndexing', False):
                    logger.info(f"Indexing completed after {wait_time} seconds")
                    break
                time.sleep(1)
                wait_time += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def configure_search_settings(self):
        """Configure search settings for better results"""
        try:
            if not self.index:
                self.get_or_create_index()
            
            self.index.update_searchable_attributes([
                'content',
                'category',
                'sub_category',
                'order_id',
                'amount',
                'profit',
                'quantity'
            ])
            

            self.index.update_filterable_attributes([
                'category',
                'sub_category',
                'amount_range',
                'profit_range',
                'quantity_range'
            ])
            
            self.index.update_sortable_attributes([
                'amount',
                'profit',
                'quantity'
            ])
            
        except Exception as e:
            logger.error(f"Error configuring search settings: {e}")
            raise
    
    def search(self, query: str, limit: int = None, filters: str = None) -> Dict[str, Any]:
        """Search documents in the index"""
        try:
            if not self.index:
                self.get_or_create_index()
            
            opt_params = {}
            
            if limit:
                opt_params['limit'] = limit
            
            if filters:
                opt_params['filter'] = filters
            
            results = self.index.search(query, opt_params)
            
            logger.info(f"Search completed - found {len(results.get('hits', []))} results")
            logger.info(f"Search processing time: {results.get('processingTimeMs', 0)}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            if not self.index:
                self.get_or_create_index()
            
            stats = self.index.get_stats()
            
            # Convert to a simple dictionary format that's JSON serializable
            if hasattr(stats, 'to_dict'):
                return stats.to_dict()
            elif hasattr(stats, '__dict__'):
                # Filter out non-serializable objects
                serializable_stats = {}
                for key, value in stats.__dict__.items():
                    try:
                        json.dumps(value)
                        serializable_stats[key] = value
                    except (TypeError, ValueError):
                        serializable_stats[key] = str(value)
                return serializable_stats
            else:
                return {
                    'numberOfDocuments': getattr(stats, 'number_of_documents', 0),
                    'databaseSize': getattr(stats, 'database_size', 0),
                    'indexSize': getattr(stats, 'index_size', 0),
                    'updateId': getattr(stats, 'update_id', 0)
                }
            
        except Exception as e:
            # Return default stats if index doesn't exist
            logger.warning(f"Could not get index stats: {e}")
            return {
                'numberOfDocuments': 0,
                'databaseSize': 0,
                'indexSize': 0,
                'updateId': 0
            }
    
    def health_check(self) -> bool:
        """Check if Meilisearch is running"""
        try:
            health = self.client.health()
            return True
        except Exception as e:
            logger.error(f"Meilisearch health check failed: {e}")
            return False

if __name__ == "__main__":
    client = MeilisearchClient()
    
    if client.health_check():
        print("Meilisearch is running!")
        
        client.get_or_create_index()
        
        client.configure_search_settings()
        
        stats = client.get_index_stats()
        print(f"Index stats: {json.dumps(stats, indent=2)}")
    else:
        print("Meilisearch is not running. Please start it first.") 