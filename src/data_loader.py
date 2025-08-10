import pandas as pd
import json
from typing import List, Dict, Any
import logging

from config import Config

logger = logging.getLogger(__name__)

class EcommerceDataLoader:
    """Load and process e-commerce order data for RAG system"""
    
    def __init__(self, data_file: str = None):
        """Initialize the data loader"""
        self.data_file = data_file or Config.DATA_FILE
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load CSV data into a pandas DataFrame"""
        try:
            logger.info(f"Loading data from file: {self.data_file}")
            self.data = pd.read_csv(self.data_file)
            logger.info(f"Successfully loaded {len(self.data)} rows from CSV file")
            logger.info(f"Data columns: {list(self.data.columns)}")
            return self.data
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise
    
    def process_data(self) -> List[Dict[str, Any]]:
        """Process the data and convert to documents for indexing"""
        logger.info("Starting data processing for indexing")
        
        if self.data is None:
            logger.info("Data not loaded, loading data first")
            self.load_data()
        
        documents = []
        logger.info(f"Processing {len(self.data)} rows into documents")
        
        for index, row in self.data.iterrows():
            doc = {
                "id": f"order_{index}",
                "order_id": str(row["Order ID"]),
                "amount": float(row["Amount"]),
                "profit": float(row["Profit"]),
                "quantity": int(row["Quantity"]),
                "category": str(row["Category"]),
                "sub_category": str(row["Sub-Category"]),
                "content": self._create_consumer_friendly_content(row),
                "business_content": self._create_business_content(row),
                "amount_range": self._get_amount_range(row["Amount"]),
                "profit_range": self._get_profit_range(row["Profit"]),
                "quantity_range": self._get_quantity_range(row["Quantity"])
            }
            documents.append(doc)
        
        logger.info(f"Successfully processed {len(documents)} documents")
        logger.info(f"Sample document ID: {documents[0]['id'] if documents else 'None'}")
        
        return documents
    
    def _create_consumer_friendly_content(self, row: pd.Series) -> str:
        """Create consumer-friendly content without business metrics"""
        product_name = str(row['Sub-Category'])
        category = str(row['Category'])
        amount = float(row['Amount'])
        quantity = int(row['Quantity'])
        
        price_description = self._get_consumer_price_description(amount)
        quality_description = self._get_consumer_quality_description(amount)
        availability_description = self._get_consumer_availability_description(quantity)
        
        return f"Product: {product_name} from {category} category. Price: ${amount:.2f}, Quantity available: {quantity}. This is a {price_description} item with {quality_description} quality. {availability_description}."
    
    def _create_business_content(self, row: pd.Series) -> str:
        """Create business-focused content with profit metrics"""
        product_name = str(row['Sub-Category'])
        category = str(row['Category'])
        amount = float(row['Amount'])
        profit = float(row['Profit'])
        quantity = int(row['Quantity'])
        
        profit_description = self._get_profit_description(profit)
        amount_range = self._get_amount_range(amount)
        
        return f"Product: {product_name} from {category} category. Price: ${amount:.2f}, Profit: ${profit:.2f}, Quantity available: {quantity}. This is a {amount_range}-priced item with {profit_description}."
            
    def _get_consumer_price_description(self, amount: float) -> str:
        """Get consumer-friendly price description"""
        if amount < 100:
            return "affordable"
        elif amount < 500:
            return "mid-range"
        else:
            return "premium"
    
    def _get_consumer_quality_description(self, amount: float) -> str:
        """Get consumer-friendly quality description based on price"""
        if amount < 100:
            return "good value"
        elif amount < 500:
            return "high quality"
        else:
            return "luxury"
    
    def _get_consumer_availability_description(self, quantity: int) -> str:
        """Get consumer-friendly availability description"""
        if quantity <= 2:
            return "Limited stock available"
        elif quantity <= 5:
            return "Moderate availability"
        else:
            return "Good stock availability"
    
    def _get_amount_range(self, amount: float) -> str:
        """Categorize amount into ranges"""
        if amount < 100:
            return "low"
        elif amount < 500:
            return "medium"
        else:
            return "high"
    
    def _get_profit_range(self, profit: float) -> str:
        """Categorize profit into ranges"""
        if profit < 0:
            return "loss"
        elif profit < 50:
            return "low_profit"
        else:
            return "high_profit"
    
    def _get_quantity_range(self, quantity: int) -> str:
        """Categorize quantity into ranges"""
        if quantity <= 2:
            return "small"
        elif quantity <= 5:
            return "medium"
        else:
            return "large"
    
    def _get_profit_description(self, profit: float) -> str:
        """Get a neutral description of profit status"""
        if profit > 0:
            return "positive profitability"
        elif profit < 0:
            return "negative profitability"
        else:
            return "break-even"
    
if __name__ == "__main__":
    loader = EcommerceDataLoader()
    documents = loader.process_data()
    
    print(f"Processed {len(documents)} documents")
    print("\nSample document:")
    print(json.dumps(documents[0], indent=2)) 