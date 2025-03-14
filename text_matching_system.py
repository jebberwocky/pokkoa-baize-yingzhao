import os
import pickle
import sqlite3
import numpy as np
import jieba
import time
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import DEFAULT_TEXT_DIR, DEFAULT_DB_PATH, LOG_FORMAT, LOG_DATE_FORMAT


class TextMatchingSystem:
    def __init__(self, db_path=DEFAULT_DB_PATH, debug=False):
        self.db_path = db_path
        self.vectorizer = None
        self.debug = debug

        # Set up logging
        if self.debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format=LOG_FORMAT,
                datefmt=LOG_DATE_FORMAT
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

    def build_vectors_from_directory(self, directory_path=DEFAULT_TEXT_DIR):
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

        # Continue with the find_relevant_text method
        input_vector = self.vectorizer.transform([processed_input])

        # Get all document vectors
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT filename, vector FROM document_vectors')
        documents = cursor.fetchall()
        conn.close()

        if not documents:
            self.debug_print("No documents found in database", level='warning')
            return []

        # Calculate similarities
        similarities = []
        for filename, vector_blob in documents:
            doc_vector = pickle.loads(vector_blob)
            similarity = cosine_similarity(input_vector, doc_vector)[0][0]
            similarities.append((filename, similarity))

        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top N results
        top_results = similarities[:top_n]

        end_time = time.time()
        self.debug_print(f"Found {len(top_results)} relevant documents in {end_time - start_time:.2f} seconds")

        return top_results

    def get_database_stats(self):
        """Get statistics about the database and vectorizer"""
        self.debug_print("Collecting database statistics", level='info')
        stats = {}

        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get document count
        cursor.execute('SELECT COUNT(*) FROM document_vectors')
        stats['document_count'] = cursor.fetchone()[0]

        # Get total content size
        cursor.execute('SELECT SUM(LENGTH(content)) FROM document_vectors')
        result = cursor.fetchone()[0]
        stats['total_content_size'] = result if result else 0

        # Get average content size
        if stats['document_count'] > 0:
            stats['avg_content_size'] = stats['total_content_size'] / stats['document_count']
        else:
            stats['avg_content_size'] = 0

        # Get database file size
        try:
            stats['database_size'] = os.path.getsize(self.db_path)
        except OSError:
            stats['database_size'] = 0

        # Check if vectorizer exists
        cursor.execute('SELECT COUNT(*) FROM vectorizer')
        stats['vectorizer_exists'] = cursor.fetchone()[0] > 0

        # Get vectorizer info if it exists
        if stats['vectorizer_exists']:
            # Load vectorizer if not already loaded
            if not self.vectorizer:
                self._load_vectorizer()

            if self.vectorizer:
                stats['vocabulary_size'] = len(self.vectorizer.vocabulary_)

                # Get vectorizer blob size
                cursor.execute('SELECT LENGTH(vectorizer) FROM vectorizer')
                stats['vectorizer_size'] = cursor.fetchone()[0]
            else:
                stats['vocabulary_size'] = 0
                stats['vectorizer_size'] = 0
        else:
            stats['vocabulary_size'] = 0
            stats['vectorizer_size'] = 0

        # Get document filenames
        cursor.execute('SELECT filename FROM document_vectors')
        filenames = [row[0] for row in cursor.fetchall()]
        stats['filenames'] = filenames

        conn.close()

        self.debug_print(f"Database stats collected: {stats}", level='info')
        return stats