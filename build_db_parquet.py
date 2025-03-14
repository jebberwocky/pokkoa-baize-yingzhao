import os
import argparse
from text_matching_system import TextMatchingSystem
from config import DEFAULT_PARQUET_DIR, DEFAULT_PARQUET_DB_PATH

def main():
    parser = argparse.ArgumentParser(description='Build vector database from text files or parquet file')
    parser.add_argument('--dir', '-d', default=DEFAULT_PARQUET_DIR,
                        help=f'Directory containing text files (default: {DEFAULT_PARQUET_DIR})')
    parser.add_argument('--db', default=DEFAULT_PARQUET_DB_PATH,
                        help=f'Database file path (default: {DEFAULT_PARQUET_DB_PATH})')
    parser.add_argument('--parquet', '-p',
                        help='Path to parquet file for building vectors')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    # Create TextMatchingSystem instance
    system = TextMatchingSystem(db_path=args.db, debug=True)

    if args.parquet:
        print(f"Processing parquet file {args.parquet}...")
        system.build_vectors_from_parquet(args.parquet)
    else:
        # Check if directory exists
        if not os.path.isdir(args.dir):
            print(f"Error: Directory '{args.dir}' not found.")
            return 1

        print(f"Processing files from {args.dir}...")
        system.build_vectors_from_parquet(args.dir)

    # Print database stats
    try:
        stats = system.get_database_stats()
        print(f"\nDatabase Statistics:")
        print(f"- Documents: {stats['document_count']}")
        print(f"- Total content size: {stats['total_content_size_mb']:.2f} MB")
        print(f"- Database file size: {stats['database_size_mb']:.2f} MB")
    except Exception as e:
        print(f"Error getting database stats: {str(e)}")

    return 0

if __name__ == "__main__":
    exit(main())
