# data_loader.py
# Handles loading contracts from CUAD JSON dataset files.
# CUAD uses SQuAD-format JSON, so we need to parse it and pull out contract text + annotations.

import json
import re
import unicodedata
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Clean up raw contract text - fix unicode, remove junk, collapse whitespace."""
    if not text:
        return ""

    # normalize unicode
    text = unicodedata.normalize("NFC", text)

    # replace smart quotes, dashes, ligatures etc.
    replacements = {
        "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
        "\u00a0": " ", "\ufb01": "fi", "\ufb02": "fl",
        "\u2022": "*",
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)

    # strip control characters (keep newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)

    # max 2 blank lines in a row
    text = re.sub(r"\n{3,}", "\n\n", text)

    # collapse spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def _parse_squad_json(squad_data: dict) -> Dict[str, Dict]:
    """
    Parse SQuAD-format CUAD JSON into a dict of {title: contract_info}.
    Each article in CUAD is one contract with multiple paragraphs and QA annotations.
    """
    contracts: Dict[str, Dict] = {}

    for article in squad_data.get("data", []):
        title = article.get("title", "unknown")
        paragraphs = article.get("paragraphs", [])

        text_chunks = []
        annotations: Dict[str, List[str]] = {}

        for para in paragraphs:
            ctx = para.get("context", "")
            if ctx:
                text_chunks.append(ctx)

            # collect ground-truth clause annotations from QA pairs
            for qa in para.get("qas", []):
                question = qa.get("question", "")
                answers = [a["text"] for a in qa.get("answers", []) if a.get("text")]
                if question and answers:
                    if question not in annotations:
                        annotations[question] = []
                    annotations[question].extend(answers)

        full_text = "\n\n".join(text_chunks)

        if title not in contracts:
            contracts[title] = {
                "title": title,
                "raw_text": full_text,
                "annotations": annotations,
            }
        else:
            # same contract split across multiple articles - merge them
            contracts[title]["raw_text"] += "\n\n" + full_text
            for q, ans in annotations.items():
                if q not in contracts[title]["annotations"]:
                    contracts[title]["annotations"][q] = []
                contracts[title]["annotations"][q].extend(ans)

    return contracts


def load_cuad_json(json_path: Path) -> Dict[str, Dict]:
    """Load a single CUAD JSON file."""
    logger.info(f"Loading CUAD JSON: {json_path}")
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    contracts = _parse_squad_json(data)
    logger.info(f"  -> {len(contracts)} contracts loaded from {json_path.name}")
    return contracts


def load_all_cuad_data(data_dir: Path, max_contracts: Optional[int] = 50) -> List[Dict]:
    """
    Load all CUAD JSON files from data_dir, deduplicate contracts by title,
    normalize the text, and return up to max_contracts entries.

    Returns a list of dicts with: contract_id, title, text, raw_text, annotations
    """
    data_dir = Path(data_dir)
    json_files = sorted(data_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {data_dir}. "
            "Download and extract data.zip from https://github.com/TheAtticusProject/cuad"
        )

    all_contracts: Dict[str, Dict] = {}

    # load CUADv1.json first since it has the full dataset
    priority = ["CUADv1.json", "train.json", "test.json"]
    ordered = sorted(
        json_files,
        key=lambda p: priority.index(p.name) if p.name in priority else 99,
    )

    for jf in ordered:
        contracts = load_cuad_json(jf)
        for title, contract in contracts.items():
            if title not in all_contracts:
                all_contracts[title] = contract
        # stop early if we already have more than we need
        if max_contracts and len(all_contracts) >= max_contracts * 2:
            break

    # build final list with normalized text
    contract_list = []
    for idx, (title, c) in enumerate(all_contracts.items()):
        normalized = normalize_text(c["raw_text"])
        if len(normalized) < 100:
            continue  # skip basically empty contracts
        contract_list.append({
            "contract_id": f"contract_{idx + 1:04d}",
            "title": title,
            "text": normalized,
            "raw_text": c["raw_text"],
            "annotations": c.get("annotations", {}),
        })

    logger.info(f"Total unique contracts after deduplication: {len(contract_list)}")

    if max_contracts:
        contract_list = contract_list[:max_contracts]

    logger.info(f"Using {len(contract_list)} contracts for processing.")
    return contract_list
