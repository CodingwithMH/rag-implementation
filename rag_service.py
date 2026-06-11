import os
from dotenv import load_dotenv
from google import genai
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


async def ask_rag(question: str):
    """
    Search Pinecone and generate answer
    """

    query_embedding = await get_embedding(question)

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    retrieved_chunks = []

    if results.matches:
        for match in results.matches:
            metadata = match.metadata

            if metadata and "text" in metadata:
                retrieved_chunks.append(
                    metadata["text"]
                )

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": "No relevant context was found in the database.",
            "retrieved_chunks": []
        }

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Answer the question using ONLY the provided context.

If the answer is not present in the context, respond:

"I cannot find the answer in the provided documents."

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