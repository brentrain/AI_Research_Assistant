import os
from dotenv import load_dotenv
# TODO: 

# Complete these sections in config.py
class Config:
    def __init__(self):
        # 1. Mistral API configuration
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.mistral_api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
       
        # 2. Processing parameters (chunk size, overlap)
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))

        # 3. Directory paths for data and results
        self.DATA_DIR = os.getenv("DATA_DIR", "./data")
        self.Sample_papers_dir = os.getenv("SAMPLE_PAPERS_DIR", "./sample_papers")
        self.processed_dir = os.getenv("PROCESSED_DIR", "data/processed")
        self.results_dir = os.getenv("RESULTS_DIR", "./results")

        # 4. Model parameters (temperature, max tokens)
        self.Model_name = os.getenv("MISTRAL_MODEL_NAME", "mistral-medium")
        self.temperature = float(os.getenv("TEMPERATURE", 0.1))
        self.max_tokens = int(os.getenv("MAX_TOKENS", 1000))

        # 5. Logging configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = os.getenv("LOG_FILE", "research_assistant.log")
        self.LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Validate configuration afer lodaing
        self.validate_config()

    def validate_config(self):
        #Validate api key
        if not self.MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY is required")
        #validate numberic parameters
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0")

        if selfOVERLAP < 0:
            raise ValueError("CHUNK_OVERLAP must be non-negative")
        
        if not 0 <= self.TEMPERATURE <= 2:
            raise ValueError("TEMPERATURE must be between 0 and 2")
        
        if self.MAX_TOKENS <= 0:
            raise ValueError("MAX_TOKENS must be greater than 0")
        
        # Validate directory paths exist or can be created
        self._ensure_directories_exist()
    
    def _ensure_directories_exist(self):
        """Create directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.SAMPLE_PAPERS_DIR,
            self.PROCESSED_DIR,
            self.RESULTS_DIR
        ]
        
        for directory in directories: