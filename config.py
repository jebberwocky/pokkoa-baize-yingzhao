"""
Common configuration settings for the Text Matching System
"""
import os

# Default file paths
DEFAULT_TEXT_DIR = "./txt"
DEFAULT_DB_PATH = "./text_vectors.db"
DEFAULT_PARQUET_DB_PATH = "./text_vectors_parquet.db"
DEFAULT_PARQUET_DIR = "./parquet"

# Default server settings
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 3168

# Default gRPC settings
DEFAULT_GRPC_HOST = os.environ.get('GRPC_HOST', '127.0.0.1')
DEFAULT_GRPC_PORT = int(os.environ.get('GRPC_PORT', 50051))

# Logging settings
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"