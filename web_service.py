import os
import json
from flask import Flask, request, jsonify
from text_matching_system import TextMatchingSystem
import logging
from config import DEFAULT_DB_PATH, DEFAULT_HOST, DEFAULT_PORT, LOG_FORMAT, LOG_DATE_FORMAT

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger('text_matching_service')

# Initialize TextMatchingSystem with environment variables or defaults
DB_PATH = os.environ.get('DB_PATH', DEFAULT_DB_PATH)
DEBUG_MODE = os.environ.get('DEBUG_MODE', 'false').lower() == 'true'

text_matching_system = TextMatchingSystem(db_path=DB_PATH, debug=DEBUG_MODE)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"})


@app.route('/match', methods=['POST'])
def match_text():
    """Endpoint to match text against the database"""
    try:
        # Get request data
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' in request body"}), 400

        query_text = data['text']

        # Get count parameter with default of 3, max of 10
        count = data.get('count', 3)
        if count > 10:
            count = 10
            logger.info(f"Requested count exceeded maximum. Using max count: 10")

        # Ensure the vectorizer is loaded
        if not text_matching_system.vectorizer:
            logger.info("Loading vectorizer from database")
            if not text_matching_system._load_vectorizer():
                return jsonify({"error": "Vectorizer not found in database. Run build script first."}), 500

        # Find relevant texts
        logger.info(f"Processing query: '{query_text[:50]}...' with count: {count}")
        results = text_matching_system.find_relevant_text(query_text, top_n=count)

        # Get the actual text content for each result
        enhanced_results = []
        conn = text_matching_system.db_path
        import sqlite3
        conn = sqlite3.connect(conn)
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

        return jsonify({
            "results": enhanced_results,
            "query": query_text,
            "count": len(enhanced_results),
            "requested_count": count
        })

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Endpoint to get database statistics"""
    try:
        stats = text_matching_system.get_database_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Text Matching Web Service')
    parser.add_argument('--host', default=DEFAULT_HOST, help=f'Host to bind the server to (default: {DEFAULT_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help=f'Port to bind the server to (default: {DEFAULT_PORT})')
    parser.add_argument('--db', default=DEFAULT_DB_PATH,
                        help=f'Database file path (default: {DEFAULT_DB_PATH})')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    # Update global variables
    DB_PATH = args.db
    DEBUG_MODE = args.debug

    # Re-initialize with updated settings
    text_matching_system = TextMatchingSystem(db_path=DB_PATH, debug=DEBUG_MODE)

    # Start the Flask server
    app.run(host=args.host, port=args.port, debug=args.debug)