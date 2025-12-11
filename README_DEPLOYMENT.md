# LoneStar AI - Streamlit Deployment Guide

AI-Powered Property Due Diligence for Texas Commercial Real Estate

## Overview

LoneStar AI is a Streamlit app that provides 2-minute property analysis using RAG (Retrieval-Augmented Generation) technology powered by Llama 3.1 8B.

## Features

- Property due diligence analysis in 2 minutes
- AI-powered insights using Texas market data (1,193 properties, 2018-2025)
- Critical issue identification
- Valuation assessment
- Investment recommendations
- Clean white, black, and burnt orange UI

## Local Development

### Prerequisites

- Python 3.9+
- Groq API Key (free at https://console.groq.com)

### Installation

1. Clone the repository:
```bash
cd OIDD255_RealEstate_RAG_Model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run app.py
```

5. Open your browser to `http://localhost:8501`

## Deployment to Streamlit Community Cloud

### Step 1: Prepare Your Repository

1. Make sure all files are committed to GitHub:
   - `app.py` (main application)
   - `requirements.txt` (dependencies)
   - `.streamlit/config.toml` (theme configuration)
   - `README.md` (project documentation)

2. Push to GitHub:
```bash
git add .
git commit -m "Add LoneStar AI Streamlit app"
git push origin main
```

### Step 2: Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)

2. Sign in with GitHub

3. Click "New app"

4. Fill in the deployment form:
   - **Repository**: Your GitHub repo (e.g., `yourusername/OIDD255_RealEstate_RAG_Model`)
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom URL (e.g., `lonestar-ai`)

5. Click "Advanced settings"

6. Add your secrets (API keys):
   - Click "Secrets"
   - Add your Groq API key:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

7. Click "Deploy"

8. Wait 2-5 minutes for deployment to complete

### Step 3: Access Your App

Your app will be live at: `https://your-app-name.streamlit.app`

Example: `https://lonestar-ai.streamlit.app`

## Using the App

### Getting a Groq API Key

1. Visit https://console.groq.com
2. Sign up with Google or GitHub
3. Navigate to API Keys section
4. Create a new API key (starts with `gsk_...`)
5. Copy and paste into the app sidebar

### Analyzing a Property

1. Enter your Groq API key in the sidebar
2. Go to the "Property Analysis" tab
3. Fill in property details:
   - Address
   - Property Type
   - Asking Price
   - Size (acres)
   - Year Built
   - Net Operating Income (NOI)
4. Paste inspection report or property notes
5. Click "Analyze Property"
6. Review the AI-generated analysis:
   - Critical Issues
   - Valuation Assessment
   - Investment Recommendation

## Architecture

### Tech Stack

- **Frontend**: Streamlit
- **LLM**: Llama 3.1 8B (via Groq API)
- **Vector DB**: ChromaDB
- **Framework**: LangChain RAG
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Data**: Texas commercial land sales (2018-2025)

### How It Works

1. **Document Processing**: Property data and domain knowledge are split into chunks
2. **Vector Embeddings**: Text chunks are converted to embeddings using HuggingFace models
3. **Vector Storage**: Embeddings stored in ChromaDB for fast retrieval
4. **RAG Pipeline**: User queries trigger:
   - Semantic search in vector DB
   - Relevant context retrieval
   - LLM generation with context
5. **Response**: AI provides analysis based on real Texas market data

## Limitations

### Free Tier Constraints

**Streamlit Community Cloud:**
- 1 GB RAM per app
- Apps sleep after inactivity (wake up in ~30 seconds when accessed)
- No time limit - app stays live indefinitely

**Groq API (Free):**
- Rate limits apply
- Fast inference (30+ tokens/sec)
- No cost for basic usage

### Current Features

- Property analysis based on user input
- Pre-loaded domain knowledge
- Sample Texas market data (200 properties)

### Future Enhancements

- Upload full 1,193 property dataset
- PDF document upload and parsing
- Multi-property comparison
- Historical trend charts
- Export analysis reports to PDF
- User authentication
- Saved analysis history

## Troubleshooting

### App Won't Start

- Check that `requirements.txt` has all dependencies
- Verify Python version is 3.9+
- Check Streamlit Community Cloud logs for errors

### API Key Issues

- Ensure API key starts with `gsk_`
- Verify key is active at console.groq.com
- Check rate limits haven't been exceeded

### Vector DB Issues

- ChromaDB persists to `./chroma_db` directory
- First run creates the database (takes 30-60 seconds)
- Delete `chroma_db/` folder to reset

### Slow Performance

- First query initializes models (30-60 seconds)
- Subsequent queries are fast (2-5 seconds)
- Groq provides fast inference
- Streamlit caches embeddings model

## Cost Analysis

### Development Costs

- **Free**: Groq API (free tier)
- **Free**: Streamlit Community Cloud
- **Free**: ChromaDB (open source)
- **Free**: HuggingFace embeddings
- **Total**: $0/month

### Production Scaling

For >1000 users/month, consider:
- Streamlit Team plan: $250/month
- Groq paid tier: Pay-per-token
- Hosted vector DB: Pinecone/Weaviate
- Estimated: $50-200/month

## Support

### Resources

- Streamlit Docs: https://docs.streamlit.io
- Groq Docs: https://console.groq.com/docs
- LangChain Docs: https://python.langchain.com/docs/
- ChromaDB Docs: https://docs.trychroma.com/

### Contact

For questions about this project:
- GitHub Issues: Create an issue in your repo
- OIDD 2550 Course Materials

## License

This project was created for OIDD 2550 - Lab 5: LLM Pitch Project (Fall 2024)

## Acknowledgments

- OIDD 2550 Course Staff
- Anthropic (Claude)
- Groq (Llama 3.1 inference)
- Streamlit Community
- Texas real estate data sources
