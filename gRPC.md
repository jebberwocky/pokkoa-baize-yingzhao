# Text Matching gRPC Service

This package provides a gRPC service for text matching functionality, allowing clients to find relevant documents based on text queries.

## Overview

The service uses TF-IDF vectorization and cosine similarity to match input text against a database of documents. It's particularly optimized for Chinese text processing with jieba segmentation.

## Components

1. **Protocol Buffers Definition** (`text_matching.proto`)
   - Defines the service interface and message types

2. **gRPC Server** (`server.py`)
   - Implements the service defined in the proto file
   - Handles client requests and processes them using the TextMatchingSystem

3. **gRPC Client** (`client.py`)
   - Provides a convenient interface for interacting with the service
   - Supports health checks, text matching, and database statistics

4. **Text Matching System** (`text_matching_system.py`)
   - Core functionality for text processing and matching
   - Handles database operations and vector calculations

## Prerequisites

- Python 3.7+
- gRPC and gRPC tools
- Required Python packages: `grpcio`, `grpcio-tools`, `jieba`, `scikit-learn`, `numpy`, `pandas`

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install grpcio grpcio-tools jieba scikit-learn numpy pandas
   ```
3. Generate gRPC code from the proto file:
   ```
   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. text_matching.proto
   ```

## Usage

### Building the Vector Database

Before using the service, you need to build the