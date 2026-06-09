import asyncio
import os
import pickle

from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

import faiss
import numpy as np

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

INDEX_FILE = "vector_store.index"
DOCS_FILE = "documents.pkl"


async def get_embedding(text: str):
    response = await client.aio.models.embed_content(
        model="models/gemini-embedding-2",
        contents=text
    )

    return response.embeddings[0].values


def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


async def create_vector_store(pdf_path: str):

    # Extract text
    text = extract_pdf_text(pdf_path)

    # Split into chunks
    chunks = chunk_text(text)

    if not chunks:
        raise Exception("No text found in document")

    print(f"Created {len(chunks)} chunks")

    embeddings = []

    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}")

        embedding = await get_embedding(chunk)
        embeddings.append(embedding)

    vectors = np.array(
        embeddings,
        dtype=np.float32
    )

    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    faiss.write_index(
        index,
        INDEX_FILE
    )

    metadata = []

    for idx, chunk in enumerate(chunks):
        metadata.append({
            "chunk_id": idx,
            "source": os.path.basename(pdf_path),
            "text": chunk
        })

    with open(DOCS_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print(
        f"Successfully stored {index.ntotal} chunks"
    )


if __name__ == "__main__":
    asyncio.run(
        create_vector_store(
            "sample.pdf"
        )
    )