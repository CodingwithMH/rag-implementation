from flask import Flask, request, jsonify
from store_vectors import create_vector_store
import json,asyncio
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])
UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)
@app.route("/")
def home():
    return "Hello, Flask!"
@app.route("/upload", methods=["POST"])
def upload():

    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    asyncio.run(
        create_vector_store(filepath)
    )

    return jsonify({
        "message": "Document indexed successfully"
    })

if __name__ == "__main__":
    app.run(debug=True)