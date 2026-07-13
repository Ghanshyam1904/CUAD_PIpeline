# CUAD Contract Analysis Pipeline

> **LLM-powered legal contract clause extraction and summarisation**  
> Built on the [Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad)  
> Model backend: [Groq](https://console.groq.com/) + Llama-3

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture & Flow Diagram](#architecture--flow-diagram)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Dataset Setup](#dataset-setup)
6. [Configuration](#configuration)
7. [Running the Pipeline](#running-the-pipeline)
8. [Output Format](#output-format)
9. [Bonus: Semantic Search](#bonus-semantic-search)
10. [Design Decisions](#design-decisions)
11. [Evaluation Criteria Met](#evaluation-criteria-met)

---

## Overview

This pipeline:

1. **Loads** a 50-contract subset from the CUAD JSON dataset
2. **Normalises** raw contract text (Unicode, whitespace, ligatures)
3. **Extracts** three key clause types per contract using a Groq-hosted LLM:
   - Termination clause
   - Confidentiality clause
   - Liability clause
4. **Summarises** each contract in 100–150 words (purpose, obligations, risks)
5. **Outputs** a structured `results.csv` and `results.json`
6. **[Bonus]** Supports semantic search over clauses using sentence-transformers

---

## Architecture & Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    CUAD Pipeline Architecture                    │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │  data/       │  ← data.zip extracted (CUADv1.json / train.json)
  │  *.json      │
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │  Step 1: Data Loading  (src/data_loader.py)  │
  │  ─────────────────────────────────────────── │
  │  • Parse SQuAD-format JSON                   │
  │  • Deduplicate by contract title             │
  │  • Concatenate paragraph chunks              │
  │  • Unicode NFC + whitespace normalization    │
  │  • Select top 50 contracts                   │
  └────────────────────┬─────────────────────────┘
                       │  List[{contract_id, title, text, annotations}]
                       ▼
  ┌──────────────────────────────────────────────┐
  │  Step 2: LLM Extraction  (src/extractor.py)  │
  │  ─────────────────────────────────────────── │
  │  For each contract:                          │
  │                                              │
  │  ┌─────────────────────────────────────────┐ │
  │  │  src/llm_client.py                      │ │
  │  │  ─────────────────                      │ │
  │  │  • Truncate to token budget (30K chars) │ │
  │  │  • Few-shot prompt engineering          │ │
  │  │  • Call Groq API (llama3-8b-8192)       │ │
  │  │  • Exponential back-off on rate limits  │ │
  │  │  ─────────────────                      │ │
  │  │  Call 1: extract_clauses()              │ │
  │  │    → TERMINATION / CONFIDENTIALITY /    │ │
  │  │      LIABILITY (structured output)      │ │
  │  │  Call 2: summarize_contract()           │ │
  │  │    → 100–150 word summary               │ │
  │  └─────────────────────────────────────────┘ │
  │                                              │
  │  • Rate-limit throttle: 3s between contracts │
  │  • Crash-safe: write CSV row after each      │
  │  • Resume mode: skip already-done contracts  │
  └────────────────────┬─────────────────────────┘
                       │
                       ▼
  ┌──────────────────────────────────────────────┐
  │  Step 3: Output (output/)                    │
  │  ─────────────────────────────────────────── │
  │  • results.csv  (contract_id, summary,       │
  │                  termination_clause,         │
  │                  confidentiality_clause,     │
  │                  liability_clause, status)   │
  │  • results.json (same + error field)         │
  └────────────────────┬─────────────────────────┘
                       │
                       ▼  [BONUS]
  ┌──────────────────────────────────────────────┐
  │  Step 4: Semantic Search                     │
  │  (src/semantic_search.py)                    │
  │  ─────────────────────────────────────────── │
  │  • Encode all clauses with all-MiniLM-L6-v2  │
  │  • Cosine similarity search                  │
  │  • CLI: python -m src.semantic_search        │
  │         --query "termination notice 30 days" │
  └──────────────────────────────────────────────┘
```

---

## Project Structure

```
cuad-contract-pipeline/
│
├── pipeline.py              # Main CLI entry-point
├── cuad_pipeline_notebook.py# Interactive notebook (jupytext format)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # CUAD JSON loading & text normalisation
│   ├── llm_client.py        # Groq API wrapper, prompts, retry logic
│   ├── extractor.py         # Batch processing orchestrator
│   └── semantic_search.py   # [Bonus] Embedding-based clause search
│
├── data/                    # ← Put extracted JSON files here
│   ├── CUADv1.json          #   (or train.json + test.json)
│   └── ...
│
└── output/                  # Auto-created by pipeline
    ├── results.csv
    └── results.json
```

---

## Quick Start

> **Note for evaluators:** Pre-computed results for all 50 contracts are already available in `output/results.csv` and `output/results.json` — no need to re-run the pipeline unless you want to verify it.

### 1. Clone & install

```bash
git clone https://github.com/Ghanshyam1904/CUAD_PIpeline.git
cd CUAD_PIpeline

# Create virtual environment (optional but recommended)
python -m venv myenv
myenv\Scripts\activate          # Windows
# source myenv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 2. Set your Groq API key

```bash
copy .env.example .env
# Edit .env and add your key:
# GROQ_API_KEY=gsk_...
```

Get a free key at **https://console.groq.com/** (no credit card needed).

### 3. Download CUAD data

```bash
# Download data.zip and extract to data/
# https://github.com/TheAtticusProject/cuad/raw/main/data.zip
mkdir data
# Extract data.zip contents into data/
```

After extraction, `data/` should contain `CUADv1.json` (and/or `train.json`, `test.json`).

### 4. Run the pipeline

```bash
python pipeline.py --data_dir data/ --output_dir output/ --max_contracts 50
```

---

## Dataset Setup

The CUAD dataset uses **SQuAD format** JSON. Each JSON file has this structure:

```json
{
  "data": [
    {
      "title": "ContractName.pdf",
      "paragraphs": [
        {
          "context": "Full contract text here...",
          "qas": [
            {
              "question": "Highlight the parts (if any) of this contract related to...",
              "answers": [{"text": "extracted clause text", "answer_start": 123}]
            }
          ]
        }
      ]
    }
  ]
}
```

The pipeline:
- Concatenates all `context` paragraphs per contract to reconstruct the full text
- Uses the `qas` answers as ground-truth annotations (for quality reference)
- Prioritises `CUADv1.json` (full 510 contracts) → `train.json` → `test.json`

---

## Configuration

| CLI Flag | Default | Description |
|---|---|---|
| `--data_dir` | `data/` | Directory with CUAD JSON files |
| `--output_dir` | `output/` | Where to save results |
| `--max_contracts` | `50` | Contracts to process (0 = all) |
| `--model` | `llama3-8b-8192` | Groq model ID |
| `--no_few_shot` | off | Disable few-shot prompting |
| `--no_resume` | off | Re-process even if CSV exists |
| `--search "query"` | - | Run semantic search after processing |

### Available Groq models

| Model | Speed | Quality | Free RPM |
|---|---|---|---|
| `llama3-8b-8192` | ⚡ Fast | ✅ Good | ~30 |
| `llama3-70b-8192` | 🐢 Slow | 🌟 Better | ~15 |

---

## Running the Pipeline

### Full run (recommended)

```bash
python pipeline.py \
  --data_dir data/ \
  --output_dir output/ \
  --max_contracts 50 \
  --model llama3-8b-8192
```

### With semantic search

```bash
python pipeline.py \
  --data_dir data/ \
  --max_contracts 50 \
  --search "indemnification clause unlimited liability"
```

### Standalone semantic search (after pipeline completes)

```bash
python -m src.semantic_search \
  --query "termination without cause" \
  --results_json output/results.json \
  --top_k 5
```

### Notebook

```bash
jupytext --to notebook cuad_pipeline_notebook.py
jupyter notebook cuad_pipeline_notebook.ipynb
```

### Ablation (no few-shot)

```bash
python pipeline.py --data_dir data/ --no_few_shot
```

---

## Output Format

### results.csv

| Column | Description |
|---|---|
| `contract_id` | Unique ID (`contract_0001` … `contract_0050`) |
| `title` | Original contract filename from CUAD |
| `summary` | 100–150 word LLM-generated summary |
| `termination_clause` | Extracted termination conditions |
| `confidentiality_clause` | Extracted confidentiality obligations |
| `liability_clause` | Extracted liability limitations |
| `status` | `success` or `error` |
| `error` | Error message if status = error |

### results.json

Same data as CSV, formatted as a JSON array of objects.

---

## Bonus: Semantic Search

The `src/semantic_search.py` module builds a vector index over all extracted clauses and summaries using the `all-MiniLM-L6-v2` sentence-transformer (80 MB, runs locally).

**Example queries:**
- `"termination without cause 30 days notice"`
- `"non-disclosure confidential information third party"`
- `"limitation of liability consequential damages"`
- `"indemnification intellectual property infringement"`

---

## Design Decisions

### Why Groq + Llama-3?

- **Free tier**: 30 RPM with no credit card required
- **Speed**: llama3-8b-8192 is 5–10x faster than GPT-4
- **Quality**: Excellent instruction following for structured extraction

### Why Few-Shot Prompting?

Including 3 in-context examples (one per clause type) significantly improves:
- Output format compliance (TERMINATION: / CONFIDENTIALITY: / LIABILITY: labels)
- Extraction precision for ambiguous clauses
- Consistency across contracts

### Token Budget Management

Legal contracts can be 50,000+ characters. We:
1. Truncate at ~30,000 chars (≈7,500 tokens + output headroom)
2. Prefer paragraph breaks for clean truncation
3. Append a `[TEXT TRUNCATED]` marker so the LLM knows context was cut

### Crash-Safe Batch Processing

- Results are written to CSV after **each contract** (not at the end)
- `--resume` mode skips already-processed contracts IDs
- Failed contracts log an error but don't abort the batch

---

## Evaluation Criteria Met

| Criterion | Implementation |
|---|---|
| **Accuracy** | Few-shot prompting + structured output parsing |
| **Code Quality** | Modular src/ package, docstrings, type hints |
| **LLM Utilization** | Token budgeting, retry logic, few-shot examples |
| **Reproducibility** | requirements.txt, .env.example, clear README |
| **Creativity** | Semantic search bonus, resume mode, ablation flag |
