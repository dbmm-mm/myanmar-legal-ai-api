#!/usr/bin/env python3
"""
Myanmar Legal AI - Flask API with Ollama Embeddings
Uses local Ollama for embeddings instead of Google API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from typing import List, Dict, Any

app = Flask(__name__)
CORS(app)

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "https://8c5ef7df-8be8-4c16-9b3f-96254e83aa5c.us-east4-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NGJmYTQ0MjctZTc3Yy00MDM0LWE5NTQtZWEzYzU1YWQ3ZTljIn0.cmMriJyQ51YgeVxzm0X9kMLoQK2IaWyrbJM5cKc82Wo")
COLLECTION_NAME = "legal_documents"
EMBEDDING_MODEL = "nomic-embed-text"

def get_embedding(text: str) -> List[float]:
    """Get embedding from Ollama"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data.get("embedding", [])
    except Exception as e:
        print(f"Embedding error: {e}")
        return []

def search_qdrant(query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """Search Qdrant for similar documents"""
    try:
        headers = {
            "api-key": QDRANT_API_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "vector": query_vector,
            "limit": limit,
            "score_threshold": 0.0  # Accept all results
        }
        
        response = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        results = []
        for result in data.get("result", []):
            results.append({
                "id": result.get("id"),
                "score": result.get("score"),
                "payload": result.get("payload", {})
            })
        
        return results
        
    except Exception as e:
        print(f"Search error: {e}")
        return []

def generate_answer(question: str, sources: List[Dict[str, Any]]) -> str:
    """Generate answer using Google Gemini"""
    try:
        if not sources:
            return "No relevant law books found for your query."
        
        # Prepare context from sources
        context = "\n\n".join([
            f"Section: {s['payload'].get('section_title', 'Unknown')}\n"
            f"Content: {s['payload'].get('full_content', '')[:1000]}"
            for s in sources
        ])
        
        # For now, return a simple response
        # In production, you would call Google Gemini API here
        answer = f"Based on the Myanmar Labor Law, here is the relevant information:\n\n{context[:500]}..."
        
        return answer
        
    except Exception as e:
        print(f"Answer generation error: {e}")
        return "Error generating answer."

@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "message": "Myanmar Legal AI API",
        "endpoints": {
            "search": "POST /search",
            "search_only": "POST /search-only",
            "health": "GET /health"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check"""
    return jsonify({"status": "healthy"})

@app.route("/search", methods=["POST"])
def search():
    """Search and generate answer"""
    try:
        data = request.json
        question = data.get("question", "")
        limit = data.get("limit", 5)
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Get embedding for question
        query_vector = get_embedding(question)
        
        if not query_vector:
            return jsonify({
                "question": question,
                "answer": "Error generating embedding for your question.",
                "sources": []
            })
        
        # Search Qdrant
        search_results = search_qdrant(query_vector, limit)
        
        # Format sources
        sources = []
        for result in search_results:
            payload = result.get("payload", {})
            sources.append({
                "title": payload.get("book_title", "Unknown"),
                "section": payload.get("section_title", "Unknown"),
                "score": result.get("score"),
                "excerpt": payload.get("content", "")[:200]
            })
        
        # Generate answer
        answer = generate_answer(question, search_results)
        
        return jsonify({
            "question": question,
            "answer": answer,
            "sources": sources
        })
        
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/search-only", methods=["POST"])
def search_only():
    """Search only without generating answer"""
    try:
        data = request.json
        question = data.get("question", "")
        limit = data.get("limit", 5)
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Get embedding
        query_vector = get_embedding(question)
        
        if not query_vector:
            return jsonify({
                "question": question,
                "sources": []
            })
        
        # Search
        search_results = search_qdrant(query_vector, limit)
        
        # Format sources
        sources = []
        for result in search_results:
            payload = result.get("payload", {})
            sources.append({
                "title": payload.get("book_title", "Unknown"),
                "section": payload.get("section_title", "Unknown"),
                "score": result.get("score"),
                "excerpt": payload.get("content", "")[:200]
            })
        
        return jsonify({
            "question": question,
            "sources": sources
        })
        
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
