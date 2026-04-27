"""
Main pipeline: scrape → preprocess → train ML → train DL
Run this once before launching the Streamlit app.
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.scraper import run_scraper
from src.preprocess import preprocess, balance_data
from src.ml_pipeline import train_and_evaluate
from src.dl_pipeline import train_deep_models


def run_pipeline(scrape=False, n_pages=10):
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # ── STEP 1: Scrape ──────────────────────────────────────────────────────────
    if scrape:
        print("=" * 50)
        print("STEP 1: Scraping PolitiFact...")
        print("=" * 50)
        run_scraper(n_pages=n_pages, output_path="data/politifact_data.csv")
    else:
        print("Skipping scrape. Using existing data/politifact_data.csv")

    # ── STEP 2: Preprocess ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 2: Preprocessing...")
    print("=" * 50)
    raw = pd.read_csv("data/politifact_data.csv")
    data = preprocess(raw)
    data_balanced = balance_data(data)
    data_balanced.to_csv("data/processed_data.csv", index=False)
    print(f"Processed data shape: {data_balanced.shape}")

    # ── STEP 3: ML Pipeline ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 3: Training ML Models...")
    print("=" * 50)
    best_model, results_df = train_and_evaluate(data_balanced)
    print("\nModel Results:")
    print(results_df)

    # ── STEP 4: Deep Learning Pipeline ──────────────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 4: Training Deep Learning Models...")
    print("=" * 50)
    ann, rnn, tokenizer = train_deep_models(data_balanced)

    print("\n" + "=" * 50)
    print("Pipeline Complete! Models saved to models/")
    print("Run:  streamlit run app.py")
    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fake News Detection Pipeline")
    parser.add_argument("--scrape", action="store_true",
                        help="Run the scraper (slow, ~300 pages)")
    parser.add_argument("--pages", type=int, default=10,
                        help="Number of pages to scrape (default: 10)")
    args = parser.parse_args()

    run_pipeline(scrape=args.scrape, n_pages=args.pages)
