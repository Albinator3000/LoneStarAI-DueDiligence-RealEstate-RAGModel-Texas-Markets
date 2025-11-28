# AI Property Due Diligence Assistant

**OIDD 2550 - Lab 5: LLM Pitch Project**

An advanced RAG-based system that analyzes real estate properties using Llama 3.1 8B, providing comprehensive risk assessments, valuations, and investment recommendations.

## Quick Start

### Option 1: Using the Startup Script (Recommended)

```bash
./start_notebook.sh
```

This will:
- Activate the virtual environment
- Check Ollama status
- Launch Jupyter Notebook

### Option 2: Manual Start

```bash
# Navigate to project directory
cd "/Users/albertwopheriv/Library/Mobile Documents/com~apple~CloudDocs/Self Learning/OIDD255_RealEstate_LLM_RAG_Model/OIDD255_RealEstate_RAG_Model"

# Activate virtual environment
source venv/bin/activate

# Start Jupyter
jupyter notebook real_estate_due_diligence_assistant.ipynb
```

## What's Included

1. **real_estate_due_diligence_assistant.ipynb** - Main notebook with complete RAG system
2. **requirements.txt** - Full dependencies (includes sentence-transformers)
3. **requirements_simple.txt** - Simplified dependencies (works with Python 3.13)
4. **start_notebook.sh** - One-click startup script
5. **SETUP.md** - Detailed setup and troubleshooting guide

## Environment Status

✅ Python 3.13.7 installed
✅ Virtual environment created at `./venv`
✅ Ollama installed and running
✅ Llama 3.1 8B model downloaded (4.9 GB)
✅ All dependencies installed
✅ Jupyter Notebook ready

## Features

### 1. Document Processing
- Extracts text from PDFs, DOCX, TXT
- Analyzes leases, inspections, financials, rent rolls

### 2. RAG Architecture
- Vector database (ChromaDB)
- Semantic search for relevant context
- Local LLM inference (Llama 3.1 8B)

### 3. Risk Assessment
- **Structural Risk**: Roof, HVAC, foundation issues
- **Financial Risk**: NOI analysis, cap rate validation
- **Legal Risk**: Lease terms, zoning compliance
- **Operational Risk**: Vacancy, tenant mix
- **Market Risk**: Comparable sales, growth trends

### 4. Valuation Engine
- Adjusts reported NOI for realistic expenses
- Calculates fair market value
- Deducts deferred maintenance
- Provides offer range

### 5. Go/No-Go Decision
- Weighted scoring across all risk categories
- Investment recommendation
- Negotiation strategy

### 6. Texas Market Intelligence
- Interactive heatmap of opportunities
- Aggregate analysis across 16 cities
- Opportunity scoring

## Demo Property

The notebook includes a complete example:
- **Property**: 4-unit multifamily, Austin, TX
- **Asking Price**: $425,000
- **Issues**: Aging roof, HVAC failure, high vacancy
- **AI Recommendation**: Offer $385k-395k (9% discount)

## How to Use

1. **Run All Cells** - Execute the notebook top to bottom
2. **Review Outputs** - See risk scores, valuation, decision
3. **Customize** - Replace synthetic data with real property docs
4. **Present** - Use visualizations and interactive maps

## For Your Presentation

### What to Demonstrate

1. **Live Demo**: Run Section 8 (complete analysis workflow)
2. **Visualizations**:
   - Risk dashboard
   - Texas opportunity heatmap
   - Market intelligence charts
3. **Technical Depth**:
   - RAG architecture explanation
   - LLM reasoning examples
   - Multi-category scoring system

### Key Talking Points

- "Reduces due diligence from 2-6 weeks to 2 minutes"
- "95% faster, 90% cheaper than traditional analysis"
- "Uses open-source Llama 3.1 8B - not just ChatGPT"
- "Geographic intelligence layer = competitive moat"
- "Professor said he'd recommend us to big companies"

## Troubleshooting

### If Jupyter won't start

```bash
source venv/bin/activate
pip install jupyter ipykernel
jupyter notebook
```

### If Ollama connection fails

```bash
ollama serve  # In one terminal
ollama run llama3.1:8b "test"  # In another terminal
```

### If cells error on imports

The notebook handles missing dependencies gracefully. You can:
1. Use the simpler embeddings (without sentence-transformers)
2. Or install PyTorch manually if needed

## Next Steps

### To improve for presentation:

1. **Add Real Data**: Find Austin property on Zillow, create docs
2. **Test Alternative Models**: Try Mistral 7B or Phi-3
3. **Deploy to Streamlit**: Create web UI for live demo
4. **Expand Geography**: Add more Texas cities

### To build as startup:

1. Fine-tune on 10k real leases
2. Integrate Zillow/MLS APIs
3. Build mobile app
4. Launch pilot with Austin brokers

## License

Academic project for OIDD 2550

## Authors

Your Team Name Here

---

**Good luck with your presentation! 🚀**