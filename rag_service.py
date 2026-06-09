import os
import pickle

import faiss
import numpy as np

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY2")
)

INDEX_FILE = "vector_store.index"
DOCS_FILE = "documents.pkl"

index = faiss.read_index(INDEX_FILE)

with open(DOCS_FILE, "rb") as f:
    documents = pickle.load(f)


async def get_embedding(text: str):

    response = await client.aio.models.embed_content(
        model="models/gemini-embedding-2",
        contents=text
    )

    return response.embeddings[0].values


async def ask_rag(question: str):

    query_embedding = await get_embedding(
        question
    )

    query_vector = np.array(
        [query_embedding],
        dtype=np.float32
    )

    distances, indices = index.search(
        query_vector,
        k=3
    )

    retrieved_chunks = []

    for idx in indices[0]:
        if idx != -1:
            retrieved_chunks.append(
                documents[idx]["text"]
            )

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {
        "question": question,
        "answer": response.text,
        "retrieved_chunks": retrieved_chunks
    }