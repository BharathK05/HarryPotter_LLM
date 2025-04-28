import os
import sys
import pickle
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

os.environ["GOOGLE_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"

def making_vector_db(documents: list, persist_directory: Path, batch_size=256) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"🔵 Starting to embed {len(documents)} documents in batches of {batch_size}...")

    texts = []
    metadatas = []

    for i in tqdm(range(0, len(documents), batch_size), desc="🔵 Preparing texts"):
        batch = documents[i:i + batch_size]
        batch_texts = [doc.page_content for doc in batch]
        batch_metadatas = [doc.metadata for doc in batch]

        texts.extend(batch_texts)
        metadatas.extend(batch_metadatas)

    print(f"✅ Preparation complete. Now building Chroma DB...")

    db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=str(persist_directory)
    )
    
    print("✅ Vector database created and automatically saved!")
    return db



def main():
    current_path = Path(__file__)
    root_path = current_path.parent.parent

    # Set the path for processed data (documents.pkl)
    processed_data_path = root_path / sys.argv[1]

    # Ensure the vector DB path exists
    vector_db_data_path = root_path / 'data' / 'vector_db'
    vector_db_data_path.mkdir(exist_ok=True)

    # Load the documents from the file
    with open(processed_data_path, 'rb') as file:
        documents = pickle.load(file)

    print(f"📦 Loaded {len(documents)} documents.")

    if len(documents) > 0:
        print(f"📄 First document preview:\n{documents[0]}")

    # Pass the documents to the vector store creation function
    db = making_vector_db(documents, vector_db_data_path)

    print(f"🎯 Vector store successfully created and saved at: {vector_db_data_path}")

if __name__ == "__main__":
    main()
