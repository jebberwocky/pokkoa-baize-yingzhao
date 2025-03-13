# Pokkoa Yingzhao

The **Text Matching System** is a Python-based tool designed to process Chinese text documents, vectorize their content using TF-IDF, and find the most relevant texts matching user queries. It leverages SQLite for storing vectors and provides a simple command-line interface for interactive queries.

![Project Image](https://pukkoa.cc/xianghuo/out-pokkoa-img/shj-yingshao.jpg)

## Dataset

Please place information from ancient Chinese books into the `txt/` directory.

## Features

- **TF-IDF Vectorization**: Uses sklearn's `TfidfVectorizer` to transform text into vectors.
- **Chinese Text Segmentation**: Utilizes `jieba` for Chinese word segmentation.
- **SQLite Storage**: Stores document vectors and the TF-IDF vectorizer in an SQLite database.
- **Similarity Matching**: Computes cosine similarity between query input and document vectors.
- **Interactive CLI**: Allows real-time querying and result display.
- **Debug Mode**: Offers detailed logging for processing steps.

## Installation

Ensure you have Python installed (>= 3.8), then install the necessary dependencies:

```bash
pip install numpy jieba scikit-learn
```

## Usage

1. **Initialize the system:**

```python
from text_matching import TextMatchingSystem

# Enable debug mode for detailed logging
system = TextMatchingSystem(debug=True)
```

2. **Build vectors from a directory of text files:**

Ensure you have a directory (e.g., `./txt`) containing `.txt` files.

```python
system.build_vectors_from_directory('./txt')
```

3. **Find relevant texts for a query:**

```python
results = system.find_relevant_text('你的查询文本', top_n=3)
for result in results:
    print(f"{result['filename']} (Similarity: {result['similarity']:.4f})")
    print(result['text'][:200])
```

4. **Add new documents dynamically:**

```python
system.add_new_document('new_file.txt', '这是新的文档内容。')
```

5. **Get database statistics:**

```python
stats = system.get_database_stats()
print(stats)
```

## Running the CLI

You can run the provided CLI by executing the following command:

```bash
python text_matching.py
```

Follow the prompts to build vectors, check database stats, and query texts interactively.

## Database Structure

The SQLite database (`text_vectors.db`) contains:

- `vectorizer` table: Stores the serialized TF-IDF vectorizer.
- `document_vectors` table: Stores document content and their corresponding vectors.

## Debugging

Enable debug mode for verbose logging by initializing the system with:

```python
system = TextMatchingSystem(debug=True)
```

## License

This project is licensed under the MIT License. Feel free to use and modify it.

---

For any questions or feature requests, please open an issue or reach out!

## About Pokkoa

- Pokkoa website: [pokkoa.com](https://pokkoa.com)
- Linkedin: [Pokkoa LinkedIn](https://www.linkedin.com/company/pokkoa)
- Hugging Face: [Pokkoa on Hugging Face](https://huggingface.co/pokkoa)
- ✉️: [contact@pokkoa.cc](contact@pokkoa.cc)

