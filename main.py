from flask import Flask, request, jsonify
from generate_response import proposal_agent
from store_vectors import create_vector_store
from agents import Runner, InputGuardrailTripwireTriggered
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
@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/generate-proposal",methods=["POST"])
def genrate_proposal():
    try:
        data = request.json
        if not data or "job_description" not in data or "link" not in data:
            return jsonify({"error": "job_description required"}), 400
        client_name = data.get("client_name", "")
        result = asyncio.run(Runner.run(
            proposal_agent,
            f"""
Job Description:
{data["job_description"]}
Freelancer Portfolio Link:
{data["link"]}
Client Name:
{client_name if client_name else "Not Provided"}
            """
        ))
        output = result.final_output

        if isinstance(output, str):
            output = output.replace("```json", "")
            output = output.replace("```", "")
            output = output.strip()

            output = json.loads(output)

        return jsonify(output)
    except InputGuardrailTripwireTriggered as e:
        print("InputGuardrail: ", str(e))
        return jsonify({
            "guardrail_error": json.loads(
                e.guardrail_result.output.output_info
            )
        }), 400

    except Exception as e:
        print("error", str(e))
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            return jsonify({
                "error": "Quota exceeded. Please wait before trying again.",
                "type": "rate_limit"
            }), 429
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)