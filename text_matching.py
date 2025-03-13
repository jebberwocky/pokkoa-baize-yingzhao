import os
import pickle
import sqlite3
import numpy as np
import jieba
import time
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextMatchingSystem:
    def __init__(self, db_path='text_vectors.db', debug=False):
        self.db_path = db_path
        self.vectorizer = None
        self.debug = debug

        # Set up logging
        if self.debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self.logger = logging.getLogger('TextMatchingSystem')

        self._init_database()
        self.debug_print(f"Initialized TextMatchingSystem with database: {db_path}")

    def debug_print(self, message, level='debug'):
        """Print debug messages if debug mode is enabled"""
        if self.debug:
            if level.lower() == 'debug':
                self.logger.debug(message)
            elif level.lower() == 'info':
                self.logger.info(message)
            elif level.lower() == 'warning':
                self.logger.warning(message)
            elif level.lower() == 'error':
                self.logger.error(message)

    def _init_database(self):
        """Initialize the database with required tables"""
        self.debug_print("Initializing database...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table for storing the vectorizer object
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS vectorizer (
            id INTEGER PRIMARY KEY,
            vectorizer BLOB
        )
        ''')

        # Table for storing document vectors
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            content TEXT,
            vector BLOB
        )
        ''')

        conn.commit()
        conn.close()
        self.debug_print("Database initialized successfully")

    def preprocess_text(self, text):
        """Preprocess text with Chinese word segmentation"""
        self.debug_print(f"Preprocessing text (length: {len(text)} chars)")
        start_time = time.time()
        words = jieba.cut(text)
        processed = ' '.join(words)
        end_time = time.time()
        self.debug_print(f"Text preprocessing completed in {end_time - start_time:.2f} seconds")
        return processed

    def build_vectors_from_directory(self, directory_path):
        """Process all text files in a directory and store their vectors"""
        self.debug_print(f"Building vectors from directory: {directory_path}", level='info')
        start_total = time.time()

        # Load all text files
        self.debug_print("Loading text files...")
        texts = {}
        file_count = 0
        total_chars = 0

        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                self.debug_print(f"Reading file: {filename}")

                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        texts[filename] = content
                        file_count += 1
                        total_chars += len(content)
                        self.debug_print(f"  - File size: {len(content)} characters")
                except Exception as e:
                    self.debug_print(f"Error reading file {filename}: {str(e)}", level='error')

        self.debug_print(f"Loaded {file_count} files with total {total_chars} characters")

        # Preprocess all texts
        self.debug_print("Preprocessing all texts...")
        start_preprocess = time.time()
        processed_texts = {}
        for filename, text in texts.items():
            self.debug_print(f"Preprocessing: {filename}")
            processed_texts[filename] = self.preprocess_text(text)

        end_preprocess = time.time()
        self.debug_print(f"Preprocessing completed in {end_preprocess - start_preprocess:.2f} seconds")

        # Create vectorizer and transform all documents
        self.debug_print("Creating TF-IDF vectorizer...")
        start_vectorize = time.time()
        self.vectorizer = TfidfVectorizer()
        all_processed = list(processed_texts.values())
        self.debug_print(f"Fitting vectorizer on {len(all_processed)} documents")
        self.vectorizer.fit(all_processed)

        # Get vocabulary size
        vocab_size = len(self.vectorizer.vocabulary_)
        self.debug_print(f"Vocabulary size: {vocab_size} terms")

        # Store the vectorizer
        self.debug_print("Storing vectorizer in database...")
        self._store_vectorizer()

        # Store each document vector
        self.debug_print("Storing document vectors in database...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for filename, processed_text in processed_texts.items():
            self.debug_print(f"Creating vector for: {filename}")
            vector = self.vectorizer.transform([processed_text])
            vector_blob = pickle.dumps(vector)
            vector_size = len(vector_blob)

            self.debug_print(f"  - Vector serialized size: {vector_size / 1024:.2f} KB")

            cursor.execute('''
            INSERT OR REPLACE INTO document_vectors (filename, content, vector)
            VALUES (?, ?, ?)
            ''', (filename, texts[filename], vector_blob))

        conn.commit()
        conn.close()

        end_vectorize = time.time()
        self.debug_print(f"Vectorizing and storing completed in {end_vectorize - start_vectorize:.2f} seconds")

        end_total = time.time()
        self.debug_print(f"Total processing time: {end_total - start_total:.2f} seconds", level='info')
        self.debug_print(f"Successfully processed and stored vectors for {len(texts)} documents", level='info')

    def _store_vectorizer(self):
        """Store the TF-IDF vectorizer in the database"""
        self.debug_print("Storing vectorizer in database")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        vectorizer_blob = pickle.dumps(self.vectorizer)
        blob_size = len(vectorizer_blob)
        self.debug_print(f"Vectorizer serialized size: {blob_size / 1024:.2f} KB")

        cursor.execute('DELETE FROM vectorizer')
        cursor.execute('INSERT INTO vectorizer (id, vectorizer) VALUES (1, ?)', (vectorizer_blob,))

        conn.commit()
        conn.close()
        self.debug_print("Vectorizer stored successfully")

    def _load_vectorizer(self):
        """Load the TF-IDF vectorizer from the database"""
        self.debug_print("Loading vectorizer from database")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT vectorizer FROM vectorizer WHERE id = 1')
        result = cursor.fetchone()

        conn.close()

        if result:
            start_time = time.time()
            blob_size = len(result[0])
            self.debug_print(f"Vectorizer blob size: {blob_size / 1024:.2f} KB")

            self.vectorizer = pickle.loads(result[0])
            end_time = time.time()

            vocab_size = len(self.vectorizer.vocabulary_)
            self.debug_print(f"Loaded vectorizer with vocabulary size: {vocab_size}")
            self.debug_print(f"Vectorizer loaded in {end_time - start_time:.2f} seconds")
            return True

        self.debug_print("No vectorizer found in database", level='warning')
        return False

    def find_relevant_text(self, user_input, top_n=3):
        """Find the most relevant texts matching the user input"""
        self.debug_print(f"Finding relevant texts for input: '{user_input[:50]}...'", level='info')
        start_time = time.time()

        # Ensure vectorizer is loaded
        if not self.vectorizer:
            self.debug_print("Vectorizer not loaded, attempting to load from database")
            if not self._load_vectorizer():
                error_msg = "Vectorizer not found in database. Run build_vectors_from_directory first."
                self.debug_print(error_msg, level='error')
                raise ValueError(error_msg)

        # Process user input
        self.debug_print("Processing user input")
        processed_input = self.preprocess_text(user_input)
        self.debug_print(f"Processed input: '{processed_input[:50]}...'")

        self.debug_print("Transforming input to vector")
        input_vector = self.vectorizer.transform([processed_input])

        # Retrieve all document vectors from database
        self.debug_print("Retrieving document vectors from database")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM document_vectors')
        doc_count = cursor.fetchone()[0]
        self.debug_print(f"Found {doc_count} documents in database")

        cursor.execute('SELECT id, filename, content, vector FROM document_vectors')
        documents = cursor.fetchall()

        conn.close()

        if not documents:
            self.debug_print("No documents found in database", level='warning')
            return []

        # Calculate similarities
        self.debug_print("Calculating similarities")
        similarities = []

        for doc_id, filename, content, vector_blob in documents:
            doc_vector = pickle.loads(vector_blob)
            similarity = cosine_similarity(input_vector, doc_vector)[0][0]
            similarities.append((doc_id, filename, content, similarity))
            self.debug_print(f"  - {filename}: similarity = {similarity:.4f}")

        # Sort by similarity (descending)
        self.debug_print("Sorting results by similarity")
        similarities.sort(key=lambda x: x[3], reverse=True)

        # Return top N results
        results = []
        for doc_id, filename, content, similarity in similarities[:top_n]:
            if similarity > 0.05:  # Only include if some similarity exists
                results.append({
                    'filename': filename,
                    'similarity': float(similarity),
                    'text': content
                })
                self.debug_print(f"Selected result: {filename} (score: {similarity:.4f})")
            else:
                self.debug_print(f"Skipped low-similarity result: {filename} (score: {similarity:.4f})")

        end_time = time.time()
        self.debug_print(f"Found {len(results)} relevant documents in {end_time - start_time:.2f} seconds",
                         level='info')

        return results

    def add_new_document(self, filename, text):
        """Add a new document to the database"""
        self.debug_print(f"Adding new document: {filename}", level='info')

        if not self.vectorizer:
            self.debug_print("Vectorizer not loaded, attempting to load from database")
            if not self._load_vectorizer():
                error_msg = "Vectorizer not found in database. Run build_vectors_from_directory first."
                self.debug_print(error_msg, level='error')
                raise ValueError(error_msg)

        self.debug_print(f"Document length: {len(text)} characters")
        processed_text = self.preprocess_text(text)

        self.debug_print("Creating vector representation")
        vector = self.vectorizer.transform([processed_text])
        vector_blob = pickle.dumps(vector)

        self.debug_print(f"Vector blob size: {len(vector_blob) / 1024:.2f} KB")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        self.debug_print("Storing document in database")
        cursor.execute('''
        INSERT OR REPLACE INTO document_vectors (filename, content, vector)
        VALUES (?, ?, ?)
        ''', (filename, text, vector_blob))

        conn.commit()
        conn.close()

        self.debug_print(f"Successfully added document: {filename}", level='info')

    def get_database_stats(self):
        """Return statistics about the database contents"""
        self.debug_print("Fetching database statistics", level='info')

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get document count
        cursor.execute('SELECT COUNT(*) FROM document_vectors')
        doc_count = cursor.fetchone()[0]

        # Get total content size
        cursor.execute('SELECT SUM(LENGTH(content)) FROM document_vectors')
        total_content_size = cursor.fetchone()[0] or 0

        # Get individual file stats
        cursor.execute('SELECT filename, LENGTH(content), LENGTH(vector) FROM document_vectors')
        file_stats = cursor.fetchall()

        conn.close()

        # Calculate size statistics
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

        stats = {
            'document_count': doc_count,
            'total_content_size_bytes': total_content_size,
            'total_content_size_mb': total_content_size / (1024 * 1024) if total_content_size else 0,
            'database_size_bytes': db_size,
            'database_size_mb': db_size / (1024 * 1024) if db_size else 0,
            'file_stats': [
                {
                    'filename': filename,
                    'content_size_kb': content_size / 1024,
                    'vector_size_kb': vector_size / 1024
                }
                for filename, content_size, vector_size in file_stats
            ]
        }

        self.debug_print(f"Database contains {doc_count} documents, total size: {db_size / 1024 / 1024:.2f} MB",
                         level='info')
        return stats


# Example usage
if __name__ == "__main__":
    # Enable debug mode
    system = TextMatchingSystem(debug=True)

    # Print welcome message
    print("=" * 60)
    print("Chinese Text Matching System with Debug Mode")
    print("=" * 60)

    # Initial setup option
    setup_choice = input("Do you want to build vectors from text files? (y/n): ").strip().lower()
    if setup_choice == 'y':
        #directory_path = input("Enter the directory path containing text files: ")
        directory_path = "./txt"
        if os.path.isdir(directory_path):
            print(f"Processing files from {directory_path}...")
            system.build_vectors_from_directory(directory_path)
        else:
            print(f"Error: Directory '{directory_path}' not found.")

    # Print database stats
    try:
        stats = system.get_database_stats()
        print(f"\nDatabase Statistics:")
        print(f"- Documents: {stats['document_count']}")
        print(f"- Total content size: {stats['total_content_size_mb']:.2f} MB")
        print(f"- Database file size: {stats['database_size_mb']:.2f} MB")
    except Exception as e:
        print(f"Error getting database stats: {str(e)}")

    # Interactive query mode
    while True:
        print("\n" + "=" * 40)
        user_input = input("Enter your query (or 'exit' to quit): ")

        if user_input.lower() in ('exit', 'quit'):
            break

        try:
            relevant_texts = system.find_relevant_text(user_input)

            # Display results
            if relevant_texts:
                print(f"\nFound {len(relevant_texts)} relevant texts:")
                for i, result in enumerate(relevant_texts, 1):
                    print(f"\n{i}. {result['filename']} (Similarity: {result['similarity']:.4f})")
                    print("-" * 40)
                    # Extract and print a relevant snippet
                    snippet_length = 200
                    print(result['text'][:snippet_length] + "..." if len(result['text']) > snippet_length else result[
                        'text'])
            else:
                print("No relevant texts found.")
        except Exception as e:
            print(f"Error processing query: {str(e)}")

    print("\nThank you for using the Text Matching System!")