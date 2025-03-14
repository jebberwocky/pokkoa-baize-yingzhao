import grpc
import argparse
import json
import logging

import text_matching_pb2
import text_matching_pb2_grpc
from config import DEFAULT_GRPC_HOST, DEFAULT_GRPC_PORT, LOG_FORMAT, LOG_DATE_FORMAT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger('text_matching_grpc_client')


class TextMatchingClient:
    def __init__(self, host, port):
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = text_matching_pb2_grpc.TextMatchingServiceStub(self.channel)
        logger.info(f"Connected to gRPC server at {host}:{port}")

    def health_check(self):
        """Send a health check request to the server"""
        try:
            response = self.stub.HealthCheck(text_matching_pb2.HealthCheckRequest())
            return response.status
        except grpc.RpcError as e:
            logger.error(f"RPC error: {e.details()}")
            return {"error": e.details()}

    def match_text(self, query_text, count=3):
        """Send a text matching request to the server"""
        try:
            response = self.stub.MatchText(
                text_matching_pb2.MatchTextRequest(text=query_text, count=count)
            )

            # Convert to dictionary for easier handling
            result = {
                "query": response.query,
                "count": response.count,
                "requested_count": response.requested_count,
                "results": []
            }

            for match in response.results:
                result["results"].append({
                    "filename": match.filename,
                    "similarity": match.similarity,
                    "content": match.content
                })

            return result
        except grpc.RpcError as e:
            logger.error(f"RPC error: {e.details()}")
            return {"error": e.details()}

    def get_stats(self):
        """Get database statistics from the server"""
        try:
            response = self.stub.GetStats(text_matching_pb2.StatsRequest())

            # Convert to dictionary for easier handling
            stats = {
                "document_count": response.document_count,
                "total_content_size": response.total_content_size,
                "avg_content_size": response.avg_content_size,
                "database_size": response.database_size,
                "vectorizer_exists": response.vectorizer_exists,
                "vocabulary_size": response.vocabulary_size,
                "vectorizer_size": response.vectorizer_size,
                "filenames": list(response.filenames)
            }

            return stats
        except grpc.RpcError as e:
            logger.error(f"RPC error: {e.details()}")
            return {"error": e.details()}


def main():
    parser = argparse.ArgumentParser(description='Text Matching gRPC Client')
    parser.add_argument('--host', default=DEFAULT_GRPC_HOST,
                        help=f'Server host (default: {DEFAULT_GRPC_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_GRPC_PORT,
                        help=f'Server port (default: {DEFAULT_GRPC_PORT})')
    parser.add_argument('--action', choices=['health', 'match', 'stats'], required=True,
                        help='Action to perform')
    parser.add_argument('--text', help='Text to match (required for match action)')
    parser.add_argument('--count', type=int, default=3,
                        help='Number of results to return (default: 3)')
    parser.add_argument('--output', help='Output file for JSON results')

    args = parser.parse_args()

    client = TextMatchingClient(args.host, args.port)

    if args.action == 'health':
        result = client.health_check()
        print(f"Health check result: {result}")

    elif args.action == 'match':
        if not args.text:
            print("Error: --text is required for match action")
            return

        result = client.match_text(args.text, args.count)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results written to {args.output}")
        else:
            print(json.dumps(result, indent=2))

    elif args.action == 'stats':
        result = client.get_stats()

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Stats written to {args.output}")
        else:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()