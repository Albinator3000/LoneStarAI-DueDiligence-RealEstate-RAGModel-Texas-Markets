# Running in Google Colab (Cloud Alternative to Local Setup)

## Why Use Google Colab?

**Advantages:**
- ✅ No local installation needed
- ✅ Free GPU access (T4)
- ✅ Easy to share with teammates
- ✅ Works on any computer (even Chromebook!)
- ✅ No model downloads (runs in cloud)

**Disadvantages:**
- ❌ Smaller models available (quality vs Llama 3.1 8B)
- ❌ Session timeout after ~90 minutes of inactivity
- ❌ Need API keys for best performance

---

## Quick Start: 3 Options

### Option 1: HuggingFace Phi-2 (Free, No API Key) ⭐ Recommended for Demo

**Pros:** Completely free, no setup
**Cons:** Lower quality than Llama 3.1

1. Upload `real_estate_due_diligence_COLAB.ipynb` to Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells
4. Uses Microsoft Phi-2 (2.7B parameters)

**Best for:** Quick demo, sharing with team

---

### Option 2: Groq API (Free, Fast) ⭐ Best Quality Free Option

**Pros:** Uses Llama 3.1 70B, very fast, free tier
**Cons:** Requires API key (30 seconds to get)

1. Get free API key: https://console.groq.com
2. In Colab: Secrets icon (🔑) → Add new secret
   - Name: `GROQ_API_KEY`
   - Value: your API key
3. Uncomment "Option 3" code in notebook
4. Run all cells

**Best for:** Presentation (best quality + free)

---

### Option 3: OpenAI API (Paid, Best Quality)

**Pros:** GPT-3.5/4, excellent results
**Cons:** Costs money (~$0.10 per analysis)

1. Get API key: https://platform.openai.com
2. In Colab: Add secret `OPENAI_API_KEY`
3. Uncomment "Option 2" code
4. Run all cells

**Best for:** Final presentation if budget allows

---

## Step-by-Step: Groq Setup (Recommended)

### Step 1: Get Groq API Key (Free)

1. Go to https://console.groq.com
2. Sign up with Google/GitHub
3. Click "Create API Key"
4. Copy the key (starts with `gsk_...`)

### Step 2: Upload Notebook to Colab

1. Go to https://colab.research.google.com
2. File → Upload notebook
3. Select `real_estate_due_diligence_COLAB.ipynb`

### Step 3: Add API Key to Colab

1. Click 🔑 icon (Secrets) in left sidebar
2. Click "+ Add new secret"
3. Name: `GROQ_API_KEY`
4. Value: paste your API key
5. Toggle "Notebook access" ON

### Step 4: Configure Runtime

1. Runtime → Change runtime type
2. Hardware accelerator: **T4 GPU**
3. Save

### Step 5: Run the Notebook

1. Find the cell with "OPTION 3: Use Groq"
2. **Uncomment all the lines** (remove the `#` at the start)
3. Runtime → Run all (Ctrl+F9)
4. Wait ~3-5 minutes for model to load

### Step 6: View Results

Scroll through the notebook to see:
- Risk assessment scores
- Valuation analysis
- Visualizations
- Final recommendation

---

## Comparison: Local vs Colab

| Feature | Local (Ollama) | Colab (Phi-2) | Colab (Groq) |
|---------|---------------|---------------|--------------|
| **Setup Time** | 10 minutes | 30 seconds | 2 minutes |
| **Cost** | Free | Free | Free |
| **Model Quality** | ⭐⭐⭐⭐⭐ (Llama 3.1 8B) | ⭐⭐⭐ (Phi-2 2.7B) | ⭐⭐⭐⭐⭐ (Llama 3.1 70B) |
| **Speed** | Fast (local) | Medium (GPU) | Very Fast (API) |
| **Requires Internet** | No | Yes | Yes |
| **API Key** | No | No | Yes (free) |
| **Session Timeout** | Never | 90 min | Never |
| **Best For** | Development | Quick demo | Presentation |

---

## Recommendation for Your Presentation

### Best Strategy: Use Both!

1. **Local (Ollama)** for development and testing
   - Better quality outputs
   - No timeout issues
   - Full control

2. **Colab (Groq)** as backup for presentation
   - In case laptop issues
   - Easy to share with professor
   - Can demo from any computer

---

## Troubleshooting Colab

### Problem: "Runtime disconnected"

**Solution:** Runtime → Reconnect, then re-run cells

### Problem: "Out of memory"

**Solution:**
1. Runtime → Factory reset runtime
2. Change to T4 GPU (not CPU)
3. Use 8-bit quantization (already enabled)

### Problem: API key not working

**Solution:**
1. Check secret name is exact: `GROQ_API_KEY` or `OPENAI_API_KEY`
2. Toggle "Notebook access" ON
3. Restart runtime

### Problem: Model loading too slow

**Solution:**
- First run takes 5-10 minutes (downloads model)
- Subsequent runs are faster
- Or switch to Groq API (instant)

---

## What Works in Colab vs Local

### ✅ Works Exactly the Same:
- Document processing
- Vector database (ChromaDB)
- RAG architecture
- Risk scoring logic
- Visualizations
- All the core functionality

### ⚠️ Different in Colab:
- LLM backend (HuggingFace/API instead of Ollama)
- Response quality (depends on model choice)
- Need to reconnect after timeout

### ❌ Not Available in Colab:
- Texas market heatmap (needs local files)
- Can add by uploading data as CSV

---

## Quick Test

To verify Colab setup works, run this single cell:

```python
from transformers import pipeline

generator = pipeline('text-generation', model='microsoft/phi-2', device_map='auto')
result = generator("Analyze this property: 4-unit multifamily in Austin", max_length=100)
print(result[0]['generated_text'])
```

If you see text output, you're good to go!

---

## For Your Presentation

### Option A: Demo from Local

**Pros:** Best quality, full features
**Cons:** Requires your laptop

Run: `./start_notebook.sh`

### Option B: Demo from Colab

**Pros:** Works from any computer, easy backup
**Cons:** Slightly lower quality (unless using Groq)

1. Open Colab notebook
2. Runtime → Run all
3. Show outputs

### Option C: Hybrid (Recommended!)

1. **Develop locally** - use Ollama for best outputs
2. **Take screenshots** of best results
3. **Upload to Colab** as backup
4. **Use Colab** if laptop issues during presentation

---

## Cost Analysis

### Groq (Free Tier):
- 14,400 requests/day
- 7,000 requests/minute
- More than enough for your presentation

### OpenAI (Paid):
- GPT-3.5-turbo: ~$0.10 per property analysis
- GPT-4: ~$0.50 per property analysis
- For 10 test runs: ~$1-5 total

**Recommendation:** Use Groq (free + fast + excellent quality)

---

## Summary

**For your class presentation:**

1. **Primary:** Run locally with Ollama (best quality)
2. **Backup:** Colab with Groq API (free, almost as good)
3. **Emergency:** Colab with Phi-2 (free, lower quality but works)

**For team collaboration:**

- Share Colab notebook (everyone can edit)
- Each teammate can run with their own Groq key
- No setup required

**For demo/pitch:**

- Groq API in Colab = professional quality
- No laptop dependency
- Fast inference (~2 seconds per query)

---

You're all set! The local setup is ready, and now you have a cloud backup too. 🚀
