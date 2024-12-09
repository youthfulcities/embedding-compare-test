import logging
import os
import sqlite3
from collections import Counter, defaultdict
from io import StringIO

import boto3
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import tiktoken

logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Initialize OpenAI Embeddings and Tokenizer
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
tokenizer = tiktoken.get_encoding("cl100k_base")  # Replace with the tokenizer for your model
logging.info("Tokenizer initialized.")

# Load FAISS index
faiss_index = faiss.read_index("/home/genna/yc/YC_Chatbot/VectorStore/240609_DevLab_157interviews_FAISS_VS/index.faiss")
logging.info(f"FAISS index dimensionality: {faiss_index.d}")

# Embed a sample word
sample_word = "example"
sample_embedding = embeddings_model.embed_query(sample_word)
logging.info(f"Sample embedding dimensionality: {len(sample_embedding)}")

# Initialize SQLite database
db_file = "embeddings_cache.db"
conn = sqlite3.connect(db_file)
c = conn.cursor()

# Create table for embeddings
c.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
    word TEXT PRIMARY KEY,
    embedding BLOB
)
""")
conn.commit()

def get_cached_embedding(word, embedding_model):
    c.execute("SELECT embedding FROM embeddings WHERE word = ?", (word,))
    result = c.fetchone()
    if result:
        embedding = np.frombuffer(result[0], dtype=np.float32)

        if len(embedding) != faiss_index.d:
            # If dimensionality is incorrect, regenerate and update the cache
            logging.warning(f"Embedding for '{word}' has incorrect dimensionality: {len(embedding)}. Regenerating...")
            embedding = embedding_model.embed_query(word)
            c.execute("UPDATE embeddings SET embedding = ? WHERE word = ?", (np.array(embedding, dtype=np.float32).tobytes(), word))
            logging.info(f"Embedding for '{word}' updated and cached with dimensionality: {len(embedding)}")
            conn.commit()
        else:
            logging.info(f"Embedding for '{word}' retrieved from cache with dimensionality: {len(embedding)}")
        return embedding
    else:
        # Generate and cache embedding
        embedding = embedding_model.embed_query(word)
        c.execute("INSERT INTO embeddings (word, embedding) VALUES (?, ?)", (word, np.array(embedding, dtype=np.float32).tobytes()))
        conn.commit()
        logging.info(f"Embedding for '{word}' generated and cached with dimensionality: {len(embedding)}")
        return np.array(embedding, dtype=np.float32)


# Load the dataset from S3
def read_csv_from_s3(bucket_name, file_key):
    """
    Reads a CSV file from S3 and returns a Pandas DataFrame.

    Args:
        bucket_name (str): The S3 bucket name.
        file_key (str): The key (path) to the file in the S3 bucket.

    Returns:
        pd.DataFrame: The CSV content as a DataFrame.
    """
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_content))
    return df

# Define bucket name and file key
bucket_name = "yccleaneddata"
file_key = "DEV/interview/codedSegments/all/interview-cleaned.csv"

# Read the CSV from S3
data = read_csv_from_s3(bucket_name, file_key)

# Filter for non-null codes and segments
filtered_data = data.dropna(subset=['code_1', 'segment'])

# Embed words using cached embeddings
def embed_words(words, embedding_model):
    """
    Embed a list of words using cached embeddings.

    Args:
        words (list): List of words to embed.
        embedding_model (OpenAIEmbeddings): The OpenAI embedding model.

    Returns:
        np.ndarray: Array of embeddings.
    """
    embeddings = []
    for word in words:
        try:
            # Use cached embedding or generate a new one
            embedding = get_cached_embedding(word, embedding_model)
            embeddings.append(embedding)
        except Exception as e:
            logging.error(f"Failed to embed word '{word}': {e}")
            embeddings.append(np.zeros(faiss_index.d, dtype=np.float32))  # Use a zero vector as fallback
    return np.array(embeddings, dtype=np.float32)


# Query FAISS to find similar words
def find_similar_words(words, embeddings, k=5):
    """
    Query FAISS to find similar words for given embeddings.

    Args:
        words (list): List of words to query.
        embeddings (np.ndarray): Embeddings of the words.
        k (int): Number of neighbors to retrieve.

    Returns:
        dict: Mapping of word -> list of similar words.
    """
    distances, indices = faiss_index.search(embeddings, k)
    similar_words = defaultdict(list)
    for i, word in enumerate(words):
        similar_indices = indices[i]
        similar_words[word] = [words[idx] for idx in similar_indices if idx < len(words)]
    return similar_words

# Aggregate word frequencies
def aggregate_frequencies(text, embedding_model):
    """
    Aggregate word frequencies using FAISS-based grouping.

    Args:
        text (str): Input text.
        embedding_model (OpenAIEmbeddings): Embedding model function.

    Returns:
        dict: Aggregated frequencies of grouped words.
    """
    words = text.split()
    embeddings = embed_words(words, embedding_model)

    # Find similar words using FAISS
    similar_groups = find_similar_words(words, embeddings)

    # Count frequencies of grouped words
    freq_dict = Counter(words)
    aggregated = defaultdict(int)
    for word, similar in similar_groups.items():
        aggregated[word] += sum(freq_dict[sim] for sim in similar if sim in freq_dict)
    return aggregated

# Calculate token counts
def count_tokens(text):
    return len(tokenizer.encode(text))

# Process each code and compute frequencies and token counts
results = []
for code, group in filtered_data.groupby('code_1'):
    logging.info(f"Processing code: {code}")
    all_text = " ".join(group['segment'])
    
    # Token count
    token_count = count_tokens(all_text)
    
    # Word frequencies
    frequencies = aggregate_frequencies(all_text, embeddings_model)
    for word, freq in frequencies.items():
        results.append({"Code": code, "Word": word, "Frequency": freq, "Token Count": token_count})

# Save the results to a CSV
output_df = pd.DataFrame(results)
output_df.to_csv("word_frequencies_and_tokens_per_code.csv", index=False)

print("Word frequencies and token counts saved to word_frequencies_and_tokens_per_code.csv")

# Close the database connection when done
conn.close()
