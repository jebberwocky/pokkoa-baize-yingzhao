import os
import sqlite3
import logging
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP
from text_matching_system import TextMatchingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('mcp_service')

# Set environment variables with defaults
DEFAULT_DB_PATH = os.environ.get('DB_PATH', 'text_vectors.db')
DEBUG_MODE = os.environ.get('DEBUG_MODE', 'true').lower() == 'true'

# Initialize TextMatchingSystem
text_matching_system = TextMatchingSystem(db_path=DEFAULT_DB_PATH, debug=DEBUG_MODE)

# Initialize FastMCP server
mcp = FastMCP("TextMatchingService")


@mcp.tool()
def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    logger.info("Health check requested")
    return {"status": "ok"}


@mcp.tool()
def match_text(text: str, count: int = 3) -> Dict[str, Any]:
    """Match text against the database

    Args:
        text: The text to match
        count: Number of results to return (default: 3, max: 10)

    Returns:
        Dictionary containing matching results
    """
    try:
        # Validate input
        if not text:
            logger.error("Missing 'text' input")
            return {"error": "Missing 'text' input"}

        # Validate count parameter
        if count > 10:
            count = 10
            logger.info(f"Requested count exceeded maximum. Using max count: 10")

        # Ensure the vectorizer is loaded
        if not text_matching_system.vectorizer:
            logger.info("Loading vectorizer from database")
            if not text_matching_system._load_vectorizer():
                return {"error": "Vectorizer not found in database. Run build script first."}

        # Find relevant texts
        logger.info(f"Processing query: '{text[:50]}...' with count: {count}")
        results = text_matching_system.find_relevant_text(text, top_n=count)

        # Get the actual text content for each result
        enhanced_results = []
        conn = sqlite3.connect(text_matching_system.db_path)
        cursor = conn.cursor()

        for filename, similarity in results:
            cursor.execute('SELECT content FROM document_vectors WHERE filename = ?', (filename,))
            content_row = cursor.fetchone()
            content = content_row[0] if content_row else ""

            enhanced_results.append({
                "filename": filename,
                "similarity": similarity,
                "content": content
            })

        conn.close()

        return {
            "results": enhanced_results,
            "query": text,
            "count": len(enhanced_results),
            "requested_count": count
        }
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
def get_stats() -> Dict[str, Any]:
    """Get database statistics"""
    try:
        logger.info("Getting database stats")
        stats = text_matching_system.get_database_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}")
        return {"error": str(e)}
