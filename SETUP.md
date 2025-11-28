# AI Property Due Diligence Assistant - Setup Guide

## Quick Start

### 1. Install Ollama (Local LLM Runtime)

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from https://ollama.com/download

**Pull Llama 3.1 8B model:**
```bash
ollama pull llama3.1:8b
```

**Verify installation:**
```bash
ollama run llama3.1:8b "Hello, test"
```

### 2. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
# Launch Jupyter
jupyter notebook real_estate_due_diligence_assistant.ipynb
```

Or use VS Code with the Jupyter extension.

---

## System Requirements

- **Python**: 3.10 or higher
- **RAM**: 8GB minimum (16GB recommended for Llama 3.1 8B)
- **Disk Space**: 5GB for model + 2GB for dependencies
- **OS**: macOS, Linux, or Windows

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INPUT                               │
│  (Leases, Inspections, Financials, Rent Rolls)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               DOCUMENT PROCESSOR                             │
│  - PDF extraction (PyPDF2)                                  │
│  - Text chunking (RecursiveCharacterTextSplitter)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              EMBEDDING LAYER                                 │
│  - Model: sentence-transformers/all-MiniLM-L6-v2           │
│  - Converts text → 384-dim vectors                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           VECTOR DATABASE (ChromaDB)                         │
│  - Stores document embeddings                                │
│  - Semantic search (cosine similarity)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             RAG RETRIEVAL                                    │
│  - Query: "What are the structural issues?"                 │
│  - Retrieves top-k relevant chunks                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          LLM (Llama 3.1 8B via Ollama)                      │
│  - Receives: Query + Retrieved context                      │
│  - Generates: Structured analysis                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            RISK SCORING ENGINE                               │
│  - Structural: /100                                         │
│  - Financial: /100                                          │
│  - Legal: /100                                              │
│  - Operational: /100                                        │
│  - Market: /100                                             │
│  - Weighted Overall Score                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          VALUATION ENGINE                                    │
│  - Adjust NOI (vacancy, CapEx, maintenance)                 │
│  - Calculate fair value (NOI / cap rate)                    │
│  - Deduct deferred maintenance                              │
│  - Generate offer range                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          GO/NO-GO DECISION                                   │
│  - Risk score + pricing analysis                            │
│  - Recommendation: Strong Go / Caution / No Go              │
│  - Negotiation strategy                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                OUTPUT                                        │
│  - Risk assessment report                                   │
│  - Valuation analysis                                       │
│  - Go/No-Go recommendation                                  │
│  - Negotiation talking points                               │
│  - Visual dashboards                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Document Ingestion
- Supports PDF, DOCX, TXT formats
- Extracts structured data from unstructured documents
- Handles leases, inspection reports, rent rolls, financials

### 2. Risk Assessment (5 Categories)
- **Structural** (30%): Roof, HVAC, foundation, major systems
- **Financial** (30%): NOI, cap rate, vacancy, operating expenses
- **Legal** (20%): Lease terms, zoning, permits
- **Operational** (10%): Tenant mix, management, turnover
- **Market** (10%): Comparable sales, rent growth, demographics

### 3. Valuation Engine
- Adjusts reported NOI for realistic expenses
- Calculates fair market value using market cap rates
- Deducts deferred maintenance
- Provides offer range with negotiation buffer

### 4. Go/No-Go Framework
Thresholds:
- **Strong Go**: Overall score ≥ 75, fair pricing
- **Proceed with Caution**: Score 60-75, minor issues
- **High Risk**: Score 45-60, requires deep discounts
- **No Go**: Score < 45, critical problems

### 5. Texas Market Intelligence
- Aggregates property data across cities
- Opportunity scoring: cap rate + growth + quality
- Interactive heatmap visualization
- Identifies undervalued markets

---

## Customization

### Change Base Model

```python
# In the notebook, modify CONFIG:
CONFIG = {
    'model_name': 'mistral:7b',  # or 'phi3:medium', 'llama3.1:70b'
    ...
}
```

### Adjust Risk Weights

```python
# Give more weight to financial risk:
RISK_WEIGHTS = {
    'Structural': 0.20,
    'Financial': 0.40,  # Increased from 0.30
    'Legal': 0.20,
    'Operational': 0.10,
    'Market': 0.10
}
```

### Modify Go/No-Go Thresholds

```python
DECISION_THRESHOLDS = {
    'strong_go': 80,      # More conservative
    'proceed_with_caution': 65,
    'high_risk': 50,
    'no_go': 0
}
```

---

## Troubleshooting

### Issue: "Connection to Ollama failed"

**Solution:**
```bash
# Ensure Ollama is running
ollama serve

# In another terminal, test:
ollama list
```

### Issue: "Out of memory when running Llama 3.1"

**Solution:**
Use a smaller model:
```bash
ollama pull phi3:mini  # Only 2GB
```

Then update CONFIG:
```python
CONFIG['model_name'] = 'phi3:mini'
```

### Issue: "ChromaDB persistence error"

**Solution:**
Delete and recreate the database:
```bash
rm -rf ./chroma_db
```

Then re-run the vector database creation cell in the notebook.

### Issue: "Slow inference times"

**Solutions:**
1. **Use GPU**: Install CUDA-enabled PyTorch
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```
   Then modify embedding config:
   ```python
   embeddings = HuggingFaceEmbeddings(
       model_name=CONFIG['embedding_model'],
       model_kwargs={'device': 'cuda'}
   )
   ```

2. **Use quantized model**: Smaller, faster
   ```bash
   ollama pull llama3.1:8b-q4_0  # 4-bit quantization
   ```

---

## Data Sources (For Production)

To enhance the system with real data, integrate:

1. **Property Listings**: Zillow API, Redfin Data Center
2. **Market Data**: Census Bureau, HUD FMR datasets
3. **Demographics**: American Community Survey
4. **Building Permits**: City/county APIs
5. **Crime Data**: FBI UCR, local police departments
6. **School Ratings**: GreatSchools API
7. **Flood Risk**: FEMA flood maps

See project doc "DATA" section for specific URLs.

---

## Next Steps

1. **Run the demo**: Execute all cells in order
2. **Upload your own property docs**: Replace synthetic data
3. **Present to class**: Use visualizations and live demo
4. **Post-class**: Consider building a Streamlit UI

---

## Support

For issues or questions:
- Review the notebook markdown documentation
- Check Ollama docs: https://ollama.com/docs
- LangChain docs: https://python.langchain.com/docs

---

**Good luck with your presentation! 🚀**
