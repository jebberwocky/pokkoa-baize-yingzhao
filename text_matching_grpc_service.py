import os
import grpc
import sqlite3
import logging
from concurrent import futures
import argparse

import text_matching_pb2
import text_matching_pb2_grpc
from text_matching_system import TextMatchingSystem
from config import DEFAULT_DB_PATH, DEFAULT_GRPC_HOST, DEFAULT_GRPC_PORT, LOG_FORMAT, LOG_DATE_FORMAT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger('text_matching_grpc_service')


class TextMatchingServicer(text_matching_pb2_grpc.TextMatchingServiceServicer):
    def __init__(self, db_path=DEFAULT_DB_PATH, debug=False):
        self.text_matching_system = TextMatchingSystem(db_path=db_path, debug=debug)
        self.db_path = db_path
        self.debug = debug
        logger.info(f"Initialized TextMatchingServicer with database: {db_path}")

    def HealthCheck(self, request, context):
        """Health check implementation"""
        logger.info("Health check requested")
        return text_matching_pb2.HealthCheckResponse(status="ok")

    def MatchText(self, request, context):
        """Match text implementation"""
        try:
            query_text = request.text
            count = request.count if request.count > 0 else 3

            # Cap the count at 10
            if count > 10:
                count = 10
                logger.info(f"Requested count exceeded maximum. Using max count: 10")

            # Ensure the vectorizer is loaded
            if not self.text_matching_system.vectorizer:
                logger.info("Loading vectorizer from database")
                if not self.text_matching_system._load_vectorizer():
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details("Vectorizer not found in database. Run build script first.")
                    return text_matching_pb2.MatchTextResponse(
                        error="Vectorizer not found in database. Run build script first."
                    )

            # Find relevant texts
            logger.info(f"Processing query: '{query_text[:50]}...' with count: {count}")
            results = self.text_matching_system.find_relevant_text(query_text, top_n=count)

            # Get the actual text content for each result
            enhanced_results = []
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for filename, similarity in results:
                cursor.execute('SELECT content FROM document_vectors WHERE filename = ?', (filename,))
                content_row = cursor.fetchone()
                content = content_row[0] if content_row else ""

                enhanced_results.append(
                    text_matching_pb2.MatchResult(
                        filename=filename,
                        similarity=similarity,
                        content=content
                    )
                )

            conn.close()

            return text_matching_pb2.MatchTextResponse(
                results=enhanced_results,
                query=query_text,
                count=len(enhanced_results),
                requested_count=count
            )

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return text_matching_pb2.MatchTextResponse(error=str(e))

    def GetStats(self, request, context):
        """Get database statistics implementation"""
        try:
            stats = self.text_matching_system.get_database_stats()

            return text_matching_pb2.StatsResponse(
                document_count=stats['document_count'],
                total_content_size=stats['total_content_size'],
                avg_content_size=stats['avg_content_size'],
                database_size=stats['database_size'],
                vectorizer_exists=stats['vectorizer_exists'],
                vocabulary_size=stats['vocabulary_size'],
                vectorizer_size=stats['vectorizer_size'],
                filenames=stats['filenames']
            )
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return text_matching_pb2.StatsResponse(error=str(e))


def serve(host, port, db_path, debug):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    text_matching_pb2_grpc.add_TextMatchingServiceServicer_to_server(
        TextMatchingServicer(db_path=db_path, debug=debug), server
    )
    server_address = f"{host}:{port}"
    server.add_insecure_port(server_address)
    server.start()
    logger.info(f"Server started on {server_address}")
    logger.info("Press Ctrl+C to stop the server")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
        logger.info("Server stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text Matching gRPC Service')
    parser.add_argument('--host', default=DEFAULT_GRPC_HOST,
                        help=f'Host to bind the server to (default: {DEFAULT_GRPC_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_GRPC_PORT,
                        help=f'Port to bind the server to (default: {DEFAULT_GRPC_PORT})')
    parser.add_argument('--db', default=DEFAULT_DB_PATH,
                        help=f'Database file path (default: {DEFAULT_DB_PATH})')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    serve(args.host, args.port, args.db, args.debug)