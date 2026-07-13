# extractor.py
# Runs the LLM extraction + summarization for a batch of contracts.
# Handles progress logging, error recovery, and saving results to CSV/JSON.

import csv
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

from src.llm_client import extract_and_summarize, DEFAULT_MODEL

logger = logging.getLogger(__name__)

# Groq free tier allows ~30 RPM, so we add a small delay between contracts
# to avoid hitting the limit too hard. 3s seems to work fine.
INTER_REQUEST_DELAY = 3.0


def process_contract(contract: Dict[str, Any], model: str = DEFAULT_MODEL, few_shot: bool = True) -> Dict[str, Any]:
    """Process a single contract - extract clauses and generate summary."""
    cid = contract["contract_id"]
    title = contract["title"]
    text = contract["text"]

    result = {
        "contract_id": cid,
        "title": title,
        "summary": "",
        "termination_clause": "",
        "confidentiality_clause": "",
        "liability_clause": "",
        "status": "success",
        "error": "",
    }

    try:
        logger.info(f"  [{cid}] Extracting clauses + summary (1 API call)...")
        combined = extract_and_summarize(text, model=model, few_shot=few_shot)
        result.update(combined)

    except Exception as exc:
        logger.error(f"  [{cid}] FAILED: {exc}")
        result["status"] = "error"
        result["error"] = str(exc)
        # fill in defaults so CSV row is still complete
        for key in ("termination_clause", "confidentiality_clause", "liability_clause", "summary"):
            if not result[key]:
                result[key] = "Extraction failed."

    return result


def process_batch(
    contracts: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    few_shot: bool = True,
    output_dir: Path = Path("output"),
    resume: bool = True,
) -> List[Dict[str, Any]]:
    """
    Process a list of contracts and save results as we go (crash-safe).
    If resume=True, skips contracts that are already in the CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"

    # check which contracts are already done
    done_ids: set = set()
    existing_results: List[Dict] = []

    if resume and csv_path.exists():
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done_ids.add(row["contract_id"])
                existing_results.append(row)
        logger.info(f"Resume mode: {len(done_ids)} contracts already processed.")

    pending = [c for c in contracts if c["contract_id"] not in done_ids]
    logger.info(f"Processing {len(pending)} contracts ({len(done_ids)} already done, {len(contracts)} total).")

    all_results = list(existing_results)

    fieldnames = [
        "contract_id", "title", "summary",
        "termination_clause", "confidentiality_clause", "liability_clause",
        "status", "error",
    ]

    write_header = not csv_path.exists() or not done_ids
    csv_file = open(csv_path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    try:
        for i, contract in enumerate(pending, 1):
            logger.info(f"\n[{i}/{len(pending)}] Processing: {contract['title']} ({contract['contract_id']})")

            result = process_contract(contract, model=model, few_shot=few_shot)
            all_results.append(result)

            # write immediately so we don't lose progress on crash
            writer.writerow({k: result.get(k, "") for k in fieldnames})
            csv_file.flush()

            # save JSON snapshot every 5 contracts
            if i % 5 == 0 or i == len(pending):
                _save_json(all_results, json_path)
                logger.info(f"  -> Progress saved ({len(all_results)} records).")

            if i < len(pending):
                time.sleep(INTER_REQUEST_DELAY)

    finally:
        csv_file.close()

    _save_json(all_results, json_path)
    logger.info(f"\n✓ Done. Results saved to:\n  CSV : {csv_path}\n  JSON: {json_path}")

    return all_results


def _save_json(results: List[Dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
