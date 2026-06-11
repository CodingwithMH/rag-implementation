import os
import asyncio
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from dotenv import load_dotenv
from store_vectors import create_vector_store
from rag_service import ask_rag

dotenv_path = Path("/home/owais4/mysite/.env")
load_dotenv(dotenv_path=dotenv_path)

app = Flask(__name__)

CORS(app, origins=["https://ecommerce-frontend-chi-cyan.vercel.app"])

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return "Hello, Flask!"


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        asyncio.run(create_vector_store(filepath))

        return jsonify({
            "message": f"Document '{file.filename}' indexed successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json() or {}
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    result = asyncio.run(ask_rag(question))
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)