# pipeline.py
# Main script to run the CUAD contract analysis pipeline.
# Usage: python pipeline.py --data_dir data/ --output_dir output/ --max_contracts 50

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def setup_logging(log_file="pipeline.log"):
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="CUAD Contract Analysis Pipeline - extracts clauses and summaries using Groq LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", default="data",
                        help="Folder with CUAD JSON files.")
    parser.add_argument("--output_dir", default="output",
                        help="Where to save results.csv and results.json.")
    parser.add_argument("--max_contracts", type=int, default=50,
                        help="How many contracts to process (0 = all).")
    parser.add_argument("--model", default="llama-3.1-8b-instant",
                        help="Groq model to use.")
    parser.add_argument("--no_few_shot", action="store_true",
                        help="Turn off few-shot examples in prompts.")
    parser.add_argument("--no_resume", action="store_true",
                        help="Ignore existing CSV and reprocess everything.")
    parser.add_argument("--search", metavar="QUERY", default=None,
                        help="Run semantic search on results after processing.")
    parser.add_argument("--log_file", default="pipeline.log",
                        help="Log file path.")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_file)
    logger = logging.getLogger("pipeline")

    logger.info("=" * 70)
    logger.info("CUAD Contract Analysis Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Data dir      : {args.data_dir}")
    logger.info(f"  Output dir    : {args.output_dir}")
    logger.info(f"  Max contracts : {args.max_contracts}")
    logger.info(f"  Model         : {args.model}")
    logger.info(f"  Few-shot      : {not args.no_few_shot}")
    logger.info(f"  Resume mode   : {not args.no_resume}")
    logger.info("")

    # Step 1 - load contracts from CUAD dataset
    from src.data_loader import load_all_cuad_data

    logger.info("Step 1 - Loading and preprocessing contracts...")
    t0 = time.time()

    max_c = args.max_contracts if args.max_contracts > 0 else None
    contracts = load_all_cuad_data(Path(args.data_dir), max_contracts=max_c)

    logger.info(f"  Loaded {len(contracts)} contracts in {time.time()-t0:.1f}s.\n")

    if not contracts:
        logger.error("No contracts loaded. Check your --data_dir path.")
        sys.exit(1)

    # Step 2 - run LLM extraction and summarization
    from src.extractor import process_batch

    logger.info("Step 2 - Running LLM extraction and summarisation...")
    logger.info("  (This may take several minutes for 50 contracts)")
    t1 = time.time()

    results = process_batch(
        contracts,
        model=args.model,
        few_shot=not args.no_few_shot,
        output_dir=Path(args.output_dir),
        resume=not args.no_resume,
    )

    elapsed = time.time() - t1
    success = sum(1 for r in results if r.get("status") == "success")
    logger.info(f"\n  Completed in {elapsed:.0f}s. {success}/{len(results)} contracts processed successfully.")

    # Step 3 (optional) - semantic search
    if args.search:
        logger.info(f"\nStep 3 - Semantic search: '{args.search}'")
        try:
            from src.semantic_search import build_index, search, print_search_results
            index = build_index(results)
            hits = search(args.search, index, top_k=5)
            print_search_results(args.search, hits)
        except ImportError as e:
            logger.warning(f"Semantic search skipped: {e}")

    logger.info("\n✓ Pipeline complete.")
    logger.info(f"  Output CSV : {Path(args.output_dir) / 'results.csv'}")
    logger.info(f"  Output JSON: {Path(args.output_dir) / 'results.json'}")


if __name__ == "__main__":
    main()
