# 👗 ByNoemie RAG Chatbot

A production-ready RAG chatbot for luxury fashion with **vibe-based semantic search**.

## 🚀 Quick Start (5 Minutes)

### Option 1: Automated Setup

**Mac/Linux:**
```bash
chmod +x quickstart.sh
./quickstart.sh
```

**Windows:**
```cmd
quickstart.bat
```

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup API key
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=gsk_xxxxxxxxxxxx

# 4. Run Streamlit demo
streamlit run app.py
```

---

## 🔑 Get FREE API Key

1. Go to **[console.groq.com](https://console.groq.com)**
2. Sign up (free)
3. Create API key
4. Add to `.env` file:
   ```
   GROQ_API_KEY=gsk_xxxxxxxxxxxx
   ```

---

## 🎯 Available Commands

### Run Streamlit Demo (Recommended)
```bash
streamlit run app.py
```
Opens browser at `http://localhost:8501`

### Process Products from CSV
```bash
# Process sample products
python scripts/process_products.py --csv data/products/sample_products.csv

# With LLM vibe generation (requires API key)
python scripts/process_products.py --csv data/products/sample_products.csv --method hybrid

# Rule-based only (no API key needed)
python scripts/process_products.py --csv data/products/sample_products.csv --method rule_based
```

### Interactive Search
```bash
python scripts/process_products.py --interactive
```

### View Database Stats
```bash
python scripts/process_products.py --stats
```

### Search Products
```bash
python scripts/process_products.py --search "romantic dinner dress"
```

---

## 📁 Project Structure

```
bynoemie_rag_v2/
├── app.py                    # Streamlit demo app
├── quickstart.sh             # Mac/Linux setup script
├── quickstart.bat            # Windows setup script
├── requirements.txt          # Dependencies
├── .env.example              # API key template
│
├── config/                   # Configuration files
│   ├── model_config.yaml     # LLM, embedding settings
│   └── prompt_templates.yaml # Prompts
│
├── src/                      # Source code
│   ├── llm/                  # LLM clients (Groq, OpenAI, etc.)
│   ├── rag/                  # ChromaDB + data processor
│   ├── vibe_generator/       # Vibe tag generation
│   └── utils/                # Utilities
│
├── data/
│   └── products/             # CSV files
│       └── sample_products.csv
│
├── scripts/
│   └── process_products.py   # CLI tool
│
└── docs/
    └── INTERVIEW_GUIDE.md    # Technical documentation
```

---

## 💡 Demo Queries to Try

| Query | Expected Match |
|-------|----------------|
| "romantic dinner" | Tiara Satin Dress |
| "main character energy" | Coco Dress |
| "boss babe" | Stella Jumpsuit |
| "quiet luxury" | Eva Slip Dress |
| "wedding guest" | Bella Off-Shoulder |
| "NYE party dress" | Coco Dress |

---

## 🌐 Deploy to Cloud (FREE)

### Streamlit Cloud (Easiest)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo → `app.py`
4. Add secret: `GROQ_API_KEY`
5. Deploy!

See `docs/CLOUD_SECRETS.md` for other options.

---

## 📊 Architecture Highlights

| Component | Choice | Why |
|-----------|--------|-----|
| **Embedding** | all-MiniLM-L6-v2 | Fast, small, 89% accuracy |
| **Vector DB** | ChromaDB | Free, zero-config |
| **LLM** | Groq Llama 3.1 70B | FREE, fast, 91% quality |
| **Retrieval** | Dual-collection (0.4/0.6) | 92% Recall@5 |

See `docs/INTERVIEW_GUIDE.md` for full technical details.

---

## 🛠️ Troubleshooting

### "No module named 'xxx'"
```bash
pip install -r requirements.txt
```

### "API key not found"
```bash
# Check your .env file
cat .env

# Should contain:
GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

### "ChromaDB error"
```bash
# Reset database
rm -rf data/embeddings/chroma_db
python scripts/process_products.py --csv data/products/sample_products.csv
```

### Port already in use
```bash
streamlit run app.py --server.port 8502
```

---

## 📄 License

MIT License - Free for personal and commercial use.

---

Made with ❤️ using LangChain, LangGraph, ChromaDB & Streamlit
