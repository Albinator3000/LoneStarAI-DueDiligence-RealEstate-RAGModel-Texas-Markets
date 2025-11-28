#!/bin/bash

# Startup script for AI Property Due Diligence Assistant

echo "======================================"
echo "AI Property Due Diligence Assistant"
echo "======================================"
echo ""

# Navigate to project directory
cd "/Users/albertwopheriv/Library/Mobile Documents/com~apple~CloudDocs/Self Learning/OIDD255_RealEstate_LLM_RAG_Model/OIDD255_RealEstate_RAG_Model"

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if Ollama is running
echo "🤖 Checking Ollama status..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Ollama is not running. Starting Ollama in background..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
else
    echo "✅ Ollama is running"
fi

# Verify Llama 3.1 8B model is available
echo "📦 Checking for Llama 3.1 8B model..."
if ollama list | grep -q "llama3.1:8b"; then
    echo "✅ Llama 3.1 8B model is ready"
else
    echo "❌ Llama 3.1 8B model not found. Please run: ollama pull llama3.1:8b"
    exit 1
fi

echo ""
echo "======================================"
echo "🚀 Starting Jupyter Notebook..."
echo "======================================"
echo ""
echo "The notebook will open in your browser."
echo "To stop: Press Ctrl+C in this terminal"
echo ""

# Launch Jupyter Notebook
jupyter notebook real_estate_due_diligence_assistant.ipynb
