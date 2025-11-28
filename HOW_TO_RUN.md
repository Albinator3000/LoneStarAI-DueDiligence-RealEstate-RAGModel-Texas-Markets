# How to Run Your AI Property Due Diligence Assistant

## ✅ Environment is Ready!

Everything is installed and configured. You're ready to go!

---

## 🚀 Method 1: One-Click Start (Easiest)

```bash
cd "/Users/albertwopheriv/Library/Mobile Documents/com~apple~CloudDocs/Self Learning/OIDD255_RealEstate_LLM_RAG_Model/OIDD255_RealEstate_RAG_Model"

./start_notebook.sh
```

This single command will:
1. Activate your virtual environment
2. Check that Ollama is running
3. Launch Jupyter Notebook
4. Open the notebook in your browser

---

## 🔧 Method 2: Manual Start

If the script doesn't work, run these commands:

```bash
# Step 1: Navigate to project directory
cd "/Users/albertwopheriv/Library/Mobile Documents/com~apple~CloudDocs/Self Learning/OIDD255_RealEstate_LLM_RAG_Model/OIDD255_RealEstate_RAG_Model"

# Step 2: Activate virtual environment
source venv/bin/activate

# Step 3: Start Jupyter Notebook
jupyter notebook real_estate_due_diligence_assistant.ipynb
```

Your browser will open automatically with the notebook.

---

## 📓 Running the Notebook

### First Time Setup (in the notebook):

1. **Run cells in order** - Click "Cell" → "Run All" or press Shift+Enter repeatedly
2. **Wait for downloads** - First run downloads the embedding model (~90MB)
3. **Watch for outputs** - Each cell will show progress

### Expected Runtime:

- **Setup cells** (1-4): ~30 seconds (one-time download)
- **Document processing**: ~5 seconds
- **Risk assessment**: ~30-60 seconds (LLM inference)
- **Valuation**: ~10 seconds
- **Visualizations**: ~5 seconds

**Total first run**: ~2-3 minutes
**Subsequent runs**: ~1 minute

---

## 🎯 What You'll See

The notebook will generate:

1. **Risk Assessment Report**
   ```
   Structural: 35/100
   Financial: 55/100
   Legal: 90/100
   Operational: 75/100
   Market: 95/100

   OVERALL: 62/100 - PROCEED WITH CAUTION
   ```

2. **Valuation Analysis**
   ```
   Asking Price: $425,000
   Fair Value: $395,000
   Recommended Offer: $385,000-$395,000
   Discount: $30k-40k (7-9%)
   ```

3. **Go/No-Go Decision**
   ```
   DECISION: PROCEED WITH CAUTION
   - Offer below asking price
   - Request $10k seller credit for repairs
   - Budget $25k-35k for deferred maintenance
   ```

4. **Visualizations**
   - Risk score bar charts
   - Texas market heatmap (interactive!)
   - Market intelligence dashboard

---

## 🔥 For Your Presentation

### Live Demo Flow:

1. **Open the notebook**
2. **Run Section 8: "Demo: End-to-End Property Analysis"**
3. **Show the output** - it generates a full report
4. **Scroll to visualizations** - show the Texas map
5. **Explain the decision** - "AI recommends 9% discount due to deferred maintenance"

### Pro Tips:

- **Pre-run the notebook before presenting** so you're not waiting for LLM inference
- **Keep the Texas heatmap open** in a browser tab for quick demo
- **Have the README.md open** to show the architecture diagram
- **Mention**: "This runs 100% locally on my laptop with Llama 3.1 8B"

---

## ⚠️ Troubleshooting

### Problem: "ModuleNotFoundError"

**Solution:**
```bash
source venv/bin/activate
pip install -r requirements_simple.txt
```

### Problem: "Cannot connect to Ollama"

**Solution:**
```bash
# Open a new terminal
ollama serve

# Then re-run the notebook
```

### Problem: Cells taking too long

**Solution:**
- Llama 3.1 8B inference takes 10-30 seconds per query
- This is normal for local LLM inference
- If too slow, consider using `phi3:mini` (faster, smaller model):
  ```bash
  ollama pull phi3:mini
  ```
  Then change `CONFIG['model_name'] = 'phi3:mini'` in the notebook

### Problem: Out of memory

**Solution:**
- Close other applications
- Or use a smaller model: `ollama pull phi3:mini`
- Or reduce the number of documents being processed

---

## 📊 Optional: Note that you are missing sentence-transformers

The notebook is configured to work WITHOUT sentence-transformers (since PyTorch doesn't support Python 3.13 yet).

It uses ChromaDB's built-in embeddings instead, which work great for the demo.

If you want to add sentence-transformers later:
1. Downgrade to Python 3.11: `pyenv install 3.11 && pyenv local 3.11`
2. Reinstall with: `pip install -r requirements.txt`

But this is **not necessary** - the current setup works perfectly!

---

## 🎓 Understanding the Output

### Risk Scores:
- **80-100**: Low risk, good investment
- **60-79**: Moderate risk, proceed with caution
- **40-59**: High risk, deep discount needed
- **0-39**: Critical risk, avoid

### Sample Property Scores:
- **Structural: 35** - Major issues (roof, HVAC)
- **Financial: 55** - High vacancy, delinquency
- **Legal: 90** - Clean leases
- **Operational: 75** - Manageable
- **Market: 95** - Austin is hot!

**Overall: 62** → Proceed with caution, negotiate hard

---

## 🚀 You're All Set!

Your environment is completely configured and ready to run.

To start:
```bash
./start_notebook.sh
```

Or manually:
```bash
source venv/bin/activate && jupyter notebook real_estate_due_diligence_assistant.ipynb
```

**Good luck with your presentation!** 🏆
