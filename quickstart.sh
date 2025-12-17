#!/bin/bash
# ============================================================================
# ByNoemie RAG Chatbot - Quick Start Setup
# ============================================================================
# Run this script after extracting the zip file:
#   chmod +x quickstart.sh
#   ./quickstart.sh
# ============================================================================

set -e

echo "=============================================="
echo "  ByNoemie RAG Chatbot - Quick Start Setup"
echo "=============================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python3 --version || { echo "❌ Python 3 not found. Please install Python 3.8+"; exit 1; }
echo "✅ Python found"
echo ""

# Create virtual environment
echo "2. Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "3. Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Install dependencies
echo "4. Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Setup environment variables
echo "5. Setting up environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your API key!"
    echo "   Get FREE Groq API key at: https://console.groq.com"
    echo ""
else
    echo "✅ .env file already exists"
fi
echo ""

# Run environment check
echo "6. Checking configuration..."
python3 env_setup.py --check
echo ""

echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Add your API key to .env file:"
echo "     GROQ_API_KEY=gsk_xxxxxxxxxxxx"
echo ""
echo "  2. Run the Streamlit demo:"
echo "     source venv/bin/activate"
echo "     streamlit run app.py"
echo ""
echo "  3. Or process products from CSV:"
echo "     python scripts/process_products.py --csv data/products/sample_products.csv"
echo ""
echo "  4. Or run interactive search:"
echo "     python scripts/process_products.py --interactive"
echo ""
echo "=============================================="
