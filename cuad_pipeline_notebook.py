"""
notebook.ipynb  ← See cuad_pipeline_notebook.py for the .py version.

cuad_pipeline_notebook.py
--------------------------
Jupyter-style notebook as a plain Python script.
Convert to .ipynb with:  jupytext --to notebook cuad_pipeline_notebook.py

Run the full pipeline interactively, inspect results, and run semantic search.
"""

# %% [markdown]
# # CUAD Contract Analysis Pipeline
# ### LLM-Powered Clause Extraction & Summarisation
# **Dataset**: Contract Understanding Atticus Dataset (CUAD)  
# **Model**  : Groq / Llama-3  
# **Bonus**  : Semantic search with sentence-transformers

# %% [markdown]
# ## 0. Setup

# %%
import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()                          # Load GROQ_API_KEY from .env
assert os.getenv("GROQ_API_KEY"), "Set GROQ_API_KEY in your .env file!"

DATA_DIR   = Path("data")             # Folder with CUADv1.json / train.json
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

print("✓ Environment ready")

# %% [markdown]
# ## 1. Data Loading & Preprocessing

# %%
from src.data_loader import load_all_cuad_data

contracts = load_all_cuad_data(DATA_DIR, max_contracts=50)
print(f"Loaded {len(contracts)} contracts")

# Inspect first contract
c = contracts[0]
print(f"\nContract ID : {c['contract_id']}")
print(f"Title       : {c['title']}")
print(f"Text length : {len(c['text'])} chars")
print(f"\nFirst 500 chars:\n{c['text'][:500]}")

# %% [markdown]
# ## 2. LLM-Powered Extraction & Summarisation

# %% [markdown]
# ### 2a. Single-contract demo (fast sanity-check)

# %%
from src.llm_client import extract_clauses, summarize_contract

demo_contract = contracts[0]
print(f"Processing: {demo_contract['title']}\n")

clauses = extract_clauses(demo_contract["text"], few_shot=True)
print("=== CLAUSES ===")
for k, v in clauses.items():
    print(f"[{k.upper()}]\n{v}\n")

summary = summarize_contract(demo_contract["text"])
print("=== SUMMARY ===")
print(summary)

# %% [markdown]
# ### 2b. Full batch processing (all 50 contracts)
# ⚠️ This takes ~5–10 minutes for 50 contracts due to Groq rate limits.

# %%
from src.extractor import process_batch

results = process_batch(
    contracts,
    model="llama3-8b-8192",
    few_shot=True,
    output_dir=OUTPUT_DIR,
    resume=True,           # Skip already-processed contracts
)

print(f"\nTotal processed: {len(results)}")
success = sum(1 for r in results if r.get("status") == "success")
print(f"Success: {success} / {len(results)}")

# %% [markdown]
# ## 3. Explore Results

# %%
df = pd.read_csv(OUTPUT_DIR / "results.csv")
df.head(3)

# %%
# Summary statistics
print(df["status"].value_counts())
print(f"\nAverage summary length: {df['summary'].str.len().mean():.0f} chars")

# %%
# Display one result nicely
row = df.iloc[0]
print(f"Contract  : {row['contract_id']} – {row['title'][:60]}")
print(f"\nSUMMARY:\n{row['summary']}")
print(f"\nTERMINATION:\n{row['termination_clause']}")
print(f"\nCONFIDENTIALITY:\n{row['confidentiality_clause']}")
print(f"\nLIABILITY:\n{row['liability_clause']}")

# %% [markdown]
# ## 4. (Bonus) Semantic Search over Clauses

# %%
# Install if needed: pip install sentence-transformers
from src.semantic_search import build_index, search, print_search_results

# Build embedding index from results
index = build_index(results)
print(f"Index built with {len(index['texts'])} clause segments")

# %%
# Example searches
queries = [
    "termination without cause 30 days notice",
    "indemnification and liability cap",
    "non-disclosure of confidential information",
]

for q in queries:
    hits = search(q, index, top_k=3)
    print_search_results(q, hits)

# %% [markdown]
# ## 5. Save Final Output

# %%
df.to_csv(OUTPUT_DIR / "results.csv", index=False)
print(f"Saved {len(df)} rows to {OUTPUT_DIR / 'results.csv'}")

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved JSON to {OUTPUT_DIR / 'results.json'}")
