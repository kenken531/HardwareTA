# 🔧 HardwareTA — Local RAG Agent for Hardware Datasheets

A fully **local** Retrieval-Augmented Generation (RAG) agent that answers hardware engineering questions by retrieving exact passages from component datasheets and citing page numbers. No cloud APIs. No data leaves your machine.

---

## Why This Matters

Engineers spend significant time reading datasheets. HardwareTA automates passage retrieval so you can ask natural-language questions like:

> *"What is the maximum supply voltage?"*  
> *"What SPI clock rates are supported?"*  
> *"What is the deep sleep current consumption?"*

…and get answers with **exact page citations** in seconds. This is also how AI-assisted hardware design tools (Flux, Altium AI, etc.) work internally.

---

## Architecture

```
datasheets/*.pdf
       │
       ▼
  [PyMuPDF]          ← text extraction, preserves page numbers
       │
  [chunker]          ← 400-char overlapping windows
       │
  [SentenceTransformer / TF-IDF]   ← embedding
       │
  [VectorStore]      ← numpy + JSON, persisted to .rag_store/
       │
  ─────────── query time ───────────
       │
  [embed question] → [cosine similarity] → top-K passages + page nums
                                                  │
                                             [Ollama LLM]
                                                  │
                                          Answer with citations
```

**Tech stack:**

| Role | Library |
|---|---|
| PDF parsing | `PyMuPDF (fitz)` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector DB | Custom `numpy` + `json` store (no native deps) |
| LLM | `ollama` (local — llama3, mistral, phi3, etc.) |
| Offline fallback | `scikit-learn` TF-IDF |

---

## Quickstart

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` downloads ~90 MB on first run — wait it out.  
> If unreachable (air-gapped, slow network), the tool falls back to TF-IDF automatically.

### 2. Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3          # or: mistral, phi3, gemma2
```

> **No Ollama?** Skip this step — the tool still retrieves and displays the exact  
> datasheet passages with page numbers. You just won't get an LLM-generated summary.

### 3. Add your datasheets

Drop any `.pdf` component datasheets into the `datasheets/` folder.

**No datasheets yet?** Auto-download a sample (ESP32 or synthetic fallback):

```bash
python main.py download
```

### 4. Index the datasheets

```bash
python main.py ingest
```

```
[+] Found 1 PDF(s): ['esp32_datasheet.pdf']
[+] 312 chunks total
[+] Loading embedding model …
    sentence-transformers/all-MiniLM-L6-v2

[+] Embedding …
    312/312 done

[✓] Indexed 312 chunks → .rag_store/
```

### 5. Ask questions

```bash
python main.py ask "What is the maximum operating voltage?"
python main.py ask "What communication interfaces are supported?"
python main.py ask "Describe the deep sleep power consumption"
python main.py ask "What are the GPIO electrical characteristics?"
```

**Example output:**
```
🔍  What is the maximum operating voltage?

┌─ Top 5 retrieved passages ──────────────────────────────────────
│ [1] esp32_datasheet.pdf  p.12  score=0.882
│     Supply voltage VCC: 2.3V ~ 3.6V. Absolute maximum: 3.7V …
│ [2] esp32_datasheet.pdf  p.11  score=0.810
│     ...
└─────────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════════════
  ANSWER
════════════════════════════════════════════════════════════════════
The ESP32 operates at 2.3V to 3.6V [esp32_datasheet.pdf, p.12].
The absolute maximum rated voltage is 3.7V — exceeding this may
permanently damage the chip [esp32_datasheet.pdf, p.12].
════════════════════════════════════════════════════════════════════

  Page citations:
    • esp32_datasheet.pdf  →  page 12
    • esp32_datasheet.pdf  →  page 11
```

### 6. Interactive demo

```bash
python main.py demo
```

Walks through 5 hardware questions interactively.

---

## Project Structure

```
hardware-rag/
├── main.py                   # CLI: download | ingest | ask | demo
├── requirements.txt
├── README.md
├── datasheets/               # ← Drop your PDFs here
│   └── esp32_datasheet.pdf
├── .rag_store/               # Auto-created vector index
│   ├── vectors.npy
│   ├── metadata.json
│   └── tfidf_vocab.pkl       # Only present when using TF-IDF fallback
└── src/
    ├── ingest.py             # PDF parsing + embedding + indexing
    ├── query.py              # Retrieval + Ollama generation
    ├── vector_store.py       # Lightweight numpy vector store
    └── download_sample.py    # Downloads sample datasheets
```

---

## Configuration

Edit constants at the top of `src/ingest.py` and `src/query.py`:

| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `400` | Characters per chunk |
| `CHUNK_STRIDE` | `320` | Stride between chunks (overlap = SIZE − STRIDE) |
| `TOP_K` | `5` | Passages retrieved per query |
| `OLLAMA_MODEL` | `llama3` | Ollama model (change to `phi3` for faster/smaller) |

---

## Common Fixes

| Problem | Fix |
|---|---|
| `sentence-transformers` download slow | ~90 MB model, wait it out — or it falls back to TF-IDF |
| `chromadb` version conflict | Not used — this project uses a lightweight numpy store |
| PDF text is empty / garbled | Already using `fitz` (PyMuPDF). If PDF is scanned, you need OCR first |
| No datasheets | `python main.py download` |
| Ollama not found | Install from https://ollama.com — or skip it, retrieved passages still shown |
| Answer seems off | Add more PDFs; TF-IDF retrieval improves with larger corpus |

---

## How the RAG Pipeline Works

**Ingest phase:**  
Each PDF is opened with PyMuPDF which extracts raw text page-by-page (far more reliable than pypdf for complex datasheet layouts). Text is split into overlapping 400-character chunks with a guaranteed-forward stride to prevent infinite loops. Each chunk is embedded with `all-MiniLM-L6-v2` (or TF-IDF offline) and stored in a lightweight numpy vector store along with its source filename and page number.

**Query phase:**  
The question is embedded with the same model. Cosine similarity scores are computed against all stored vectors. The top-K chunks are retrieved with their page citations, injected into a prompt, and sent to Ollama. The LLM is instructed to cite `[filename, p.N]` for every claim. Citations are also printed as a structured reference list.

---

## License

MIT
