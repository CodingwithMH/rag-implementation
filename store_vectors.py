import asyncio
import os
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from pinecone import Pinecone

load_dotenv()

# Gemini Client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Pinecone Client
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index = pc.Index(
    os.getenv("PINECONE_INDEX")
)


async def get_embedding(text: str):
    """
    Generate embedding using Gemini
    """

    response = await client.aio.models.embed_content(
        model="models/gemini-embedding-2",
        contents=text
    )

    return response.embeddings[0].values


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract all text from PDF
    """

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
    """
    Split text into overlapping chunks
    """

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
    """
    Extract PDF text and upload embeddings to Pinecone
    """

    text = extract_pdf_text(pdf_path)

    chunks = chunk_text(text)

    if not chunks:
        raise Exception("No text found in document")

    print(f"Created {len(chunks)} chunks")

    vectors_to_upsert = []

    filename = os.path.basename(pdf_path)

    for i, chunk in enumerate(chunks):

        print(f"Embedding chunk {i + 1}/{len(chunks)}")

        embedding = await get_embedding(chunk)

        vectors_to_upsert.append(
            {
                "id": f"{filename}-{i}",
                "values": embedding,
                "metadata": {
                    "source": filename,
                    "chunk_number": i,
                    "text": chunk
                }
            }
        )

    print("Uploading vectors to Pinecone...")

    batch_size = 100

    for i in range(0, len(vectors_to_upsert), batch_size):

        batch = vectors_to_upsert[i:i + batch_size]

        index.upsert(
            vectors=batch
        )

    print(
        f"Successfully stored {len(vectors_to_upsert)} chunks from {filename}"
    )

    return {
        "status": "success",
        "chunks_uploaded": len(vectors_to_upsert),
        "source": filename
    }


if __name__ == "__main__":
    asyncio.run(
        create_vector_store("sample.pdf")
    )