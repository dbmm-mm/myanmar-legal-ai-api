#!/usr/bin/env python3
"""
Myanmar Legal AI - Flask API
Search Qdrant for legal documents and generate answers with Google Gemini
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
QDRANT_URL = os.getenv('QDRANT_URL', 'https://8c5ef7df-8be8-4c16-9b3f-96254e83aa5c.us-east4-0.gcp.cloud.qdrant.io')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', '')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
COLLECTION_NAME = 'legal_documents'

print("=" * 60)
print("🏛️  Myanmar Legal AI - Flask API")
print("=" * 60)
print(f"Qdrant URL: {QDRANT_URL}")
print(f"Collection: {COLLECTION_NAME}")
print()

# Function to generate embedding
def generate_embedding(text):
    """Generate embedding using Google AI"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={GOOGLE_API_KEY}"
        
        payload = {
            "model": "models/gemini-embedding-001",
            "content": {
                "parts": [{"text": text[:2000]}]
            }
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        data = response.json()
        
        if 'embedding' in data:
            return data['embedding']['values']
        else:
            print(f"Embedding error: {data}")
            return None
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Function to search Qdrant
def search_qdrant(query_embedding, limit=5):
    """Search Qdrant for similar documents"""
    try:
        url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search"
        
        headers = {
            "api-key": QDRANT_API_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "vector": query_embedding,
            "limit": limit,
            "with_payload": True,
            "score_threshold": 0.0  # Accept all results, no minimum threshold
        }
        
        print(f"Searching Qdrant with URL: {url}")
        print(f"API Key present: {bool(QDRANT_API_KEY)}")
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        print(f"Qdrant response status: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            result_list = results.get('result', [])
            print(f"Qdrant returned {len(result_list)} results")
            return result_list
        else:
            print(f"Search error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error searching Qdrant: {e}")
        return []

# Function to generate answer with Gemini
def generate_answer(question, context_chunks):
    """Generate answer using Google Gemini"""
    try:
        # Combine context chunks
        context = "\n\n".join([chunk['payload']['content_chunk'] for chunk in context_chunks])
        
        # Create prompt
        prompt = f"""You are a Myanmar legal expert. Answer the following legal question based on the provided law book excerpts.

Question: {question}

Law Book Excerpts:
{context}

Please provide a clear, helpful answer in English or Burmese (Myanmar language) as appropriate."""
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        data = response.json()
        
        if 'candidates' in data and len(data['candidates']) > 0:
            answer = data['candidates'][0]['content']['parts'][0]['text']
            return answer
        else:
            print(f"Generation error: {data}")
            return "Unable to generate answer"
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Error: {str(e)}"

# Routes
@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "status": "🏛️ Myanmar Legal AI API is running!",
        "version": "1.0",
        "endpoints": {
            "POST /search": "Search for legal documents and get AI-powered answers",
            "GET /health": "Check API health"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "qdrant": QDRANT_URL,
        "collection": COLLECTION_NAME
    })

@app.route('/search', methods=['POST'])
def search():
    """Search for legal documents and generate answer"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' parameter"}), 400
        
        question = data['question']
        limit = data.get('limit', 5)
        
        print(f"\n🔍 Searching for: {question}")
        
        # Generate embedding for question
        question_embedding = generate_embedding(question)
        if not question_embedding:
            return jsonify({"error": "Failed to generate embedding"}), 500
        
        # Search Qdrant
        search_results = search_qdrant(question_embedding, limit)
        
        # Debug: Log search results
        print(f"Search results count: {len(search_results)}")
        if search_results:
            for i, result in enumerate(search_results):
                print(f"  Result {i+1}: score={result.get('score')}, title={result.get('payload', {}).get('title')}")
        
        if not search_results:
            return jsonify({
                "question": question,
                "answer": "No relevant law books found for your query.",
                "sources": []
            })
        
        # Generate answer
        answer = generate_answer(question, search_results)
        
        # Format sources
        sources = []
        for result in search_results:
            sources.append({
                "title": result['payload']['title'],
                "section": result['payload']['section'],
                "score": result['score'],
                "excerpt": result['payload']['content_chunk'][:200] + "..."
            })
        
        return jsonify({
            "question": question,
            "answer": answer,
            "sources": sources,
            "results_count": len(search_results)
        })
    
    except Exception as e:
        print(f"Error in /search: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/search-only', methods=['POST'])
def search_only():
    """Search without generating answer (faster)"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' parameter"}), 400
        
        question = data['question']
        limit = data.get('limit', 5)
        
        print(f"\n🔍 Searching for: {question}")
        
        # Generate embedding for question
        question_embedding = generate_embedding(question)
        if not question_embedding:
            return jsonify({"error": "Failed to generate embedding"}), 500
        
        # Search Qdrant
        search_results = search_qdrant(question_embedding, limit)
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "title": result['payload']['title'],
                "section": result['payload']['section'],
                "score": result['score'],
                "excerpt": result['payload']['content_chunk'][:200] + "..."
            })
        
        return jsonify({
            "question": question,
            "results": results,
            "results_count": len(results)
        })
    
    except Exception as e:
        print(f"Error in /search-only: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
