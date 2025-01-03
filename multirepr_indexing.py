import os
import sqlite3
import json
import logging
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import SummaryChain, RetrievalQA
from langchain.retrievers import MultiVectorRetriever
from unstructured.partition.auto import partition

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Function to connect to SQLite database
def connect_db(db_name='documents.db'):
    return sqlite3.connect(db_name)

def create_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        metadata TEXT
    )
    ''')
    conn.commit()

# Function to add documents
def add_document(conn, doc_id, metadata):
    try:
        cursor = conn.cursor()
        metadata_json = json.dumps(metadata)
        cursor.execute('INSERT INTO documents (id, metadata) VALUES (?, ?)', 
                       (doc_id, metadata_json))
        conn.commit()
        logging.info(f"Document {doc_id} added successfully.")
    except Exception as e:
        logging.error(f"Error adding document {doc_id}: {e}")

# Function to add summaries
def add_summary(doc_id, summary, summary_vector_store):
    try:
        summary_vector_store.add_documents([{ "text": summary, "metadata": {"id": doc_id} }])
        logging.info(f"Summary for document {doc_id} added successfully.")
    except Exception as e:
        logging.error(f"Error adding summary for document {doc_id}: {e}")

# Function to process documents in a folder
def process_documents(folder_path, vector_db_path):
    conn = connect_db()
    create_table(conn)

    # Create a Chroma vector store for summaries
    summary_vector_store = Chroma(embeddings, collection_name="summary_vectors", persist_directory=vector_db_path)

    # Create a summary chain
    summary_chain = SummaryChain(llm=OpenAIEmbeddings())

    # Crawl and process documents
    for doc_id, filename in enumerate(os.listdir(folder_path), start=1):
        file_path = os.path.join(folder_path, filename)
        documents = partition(file_path)
        
        for doc in documents:
            content = doc.text
            metadata = {
                'filename': filename,
                'source': file_path,
                'pagenumber': doc.metadata.get('page_number', 0),
                'image_count': len(doc.metadata.get('images', [])),
                'images': {i: img['path'] for i, img in enumerate(doc.metadata.get('images', []))}
            }
            summary = summary_chain.run(content)
            add_document(conn, doc_id, metadata)
            add_summary(doc_id, summary, summary_vector_store)

    conn.close()

# Function to create and run a retrieval chain
def run_retrieval_chain(vector_db_path, query):
    # Reconnect to the SQLite database
    conn = connect_db()

    # Create a Chroma vector store for summaries
    summary_vector_store = Chroma(embeddings, collection_name="summary_vectors", persist_directory=vector_db_path)

    # Create a MultiVectorRetriever
    retriever = MultiVectorRetriever(
        vectorstore=summary_vector_store,
        byte_store=conn,  # Using SQLite as the document store
        id_key="id"
    )

    # Create a retrieval chain
    retrieval_chain = RetrievalQA(
        retriever=retriever,
        llm=OpenAIEmbeddings()
    )

    # Run the query
    result = retrieval_chain.run(query)
    logging.info(f"Query Result: {result}")

    conn.close()

# Main function
def main():
    folder_path = "path/to/documents"  # User-specified folder path
    vector_db_path = "path/to/vector_db"  # User-specified vector DB path

    process_documents(folder_path, vector_db_path)

    # Example query
    query = "What is the content of Document 1?"
    run_retrieval_chain(vector_db_path, query)

if __name__ == "__main__":
    main() 