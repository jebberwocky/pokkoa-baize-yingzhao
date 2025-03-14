"""
Common configuration settings for the Text Matching System
"""

# Default file paths
DEFAULT_TEXT_DIR = "./txt"
DEFAULT_DB_PATH = "./text_vectors.db"
DEFAULT_PARQUET_DB_PATH = "./text_vectors_parquet.db"
DEFAULT_PARQUET_DIR = "./parquet"

# Default server settings
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 3168

# Logging settings
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"