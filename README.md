## Prerequisites

- Python 3.8 or higher
- pip package manager
- Meilisearch instance (local or cloud)
- OpenRouter API key

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shubhjain19/agentic-rag.git
   cd agentic-rag
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` file with your configuration:
   ```
   MEILISEARCH_URL=http://localhost:7700
   MEILISEARCH_MASTER_KEY=your_master_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

## Setup

### Meilisearch Setup

1. **Install Meilisearch**
   ```bash
   # Using Docker
   docker run -p 7700:7700 getmeili/meilisearch:latest
   
   # Or download binary from https://github.com/meilisearch/meilisearch/releases
   ```

2. **Create index and add data**
   ```python
   from src.meilisearch_client import MeilisearchClient
   
   client = MeilisearchClient()
   client.create_index("products")
   client.add_documents("products", your_product_data)
   ```

### Data Preparation

1. **Place your CSV data in the `data/` directory**
2. **Update data loading configuration in `src/config.py`**
3. **Run data loader to process and index your data**
