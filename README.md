# 🏛️ Myanmar Legal AI API

Flask API for searching Myanmar law books using semantic search with Qdrant and Google Gemini.

## Features

- 🔍 Semantic search through law books
- 🤖 AI-powered answer generation with Google Gemini
- 📚 Powered by Qdrant vector database
- 🚀 Deployed on Railway
- 🌐 Public API (anyone can search)

## API Endpoints

### POST /search
Search for legal documents and get AI-powered answers.

**Request:**
```json
{
  "question": "What is the minimum working age in Myanmar?",
  "limit": 5
}
```

**Response:**
```json
{
  "question": "What is the minimum working age in Myanmar?",
  "answer": "According to Myanmar Labor Law...",
  "sources": [
    {
      "title": "Myanmar Labor Law 2011",
      "section": "Chapter 2",
      "score": 0.95,
      "excerpt": "..."
    }
  ],
  "results_count": 1
}
```

### POST /search-only
Search without generating answer (faster).

### GET /health
Check API health.

### GET /
Home endpoint with API information.

## Environment Variables

- `QDRANT_URL`: Qdrant Cloud URL
- `QDRANT_API_KEY`: Qdrant API Key
- `GOOGLE_API_KEY`: Google AI API Key
- `PORT`: Server port (default: 5000)

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env with your credentials

# Run API
python legal_ai_api.py
```

Visit http://localhost:5000

## Deployment

Deployed on Railway: https://your-railway-app.railway.app

## Testing

```bash
# Test with cURL
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the minimum working age in Myanmar?",
    "limit": 5
  }'
```

## License

MIT
