import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv
from store_vectors import create_vector_store
from rag_service import ask_rag  # Imported to give you an endpoint for queries!

load_dotenv(find_dotenv())

app = Flask(__name__)
CORS(app, origins=[os.getenv("FRONTEND_URI")])
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "Hello, Flask!"

# --- FIX: Native async handling used instead of asyncio.run ---
@app.route("/upload", methods=["POST"])
async def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        # Calls the function cleanly natively in Flask's worker loop
        await create_vector_store(filepath)
        return jsonify({"message": f"Document '{file.filename}' indexed successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Added Query Endpoint to link everything together seamlessly!
@app.route("/query", methods=["POST"])
async def query():
    data = request.get_json() or {}
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400
        
    result = await ask_rag(question)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)