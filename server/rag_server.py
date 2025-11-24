#!/usr/bin/env python3
from flask import Flask, request, jsonify
from flask_cors import CORS

# Create flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Example (Flask)
@app.route('/query', methods=['POST'])
def rag():
    data = request.json
    query = data['query']
    # ... run your RAG pipeline ...
    return jsonify({
        "answer": "Hawaii, USA",
        "retrieved_docs": [["doc123", 0.92], ["doc456", 0.87]]  # 10 items
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
