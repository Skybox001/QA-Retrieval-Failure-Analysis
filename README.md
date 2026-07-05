# QA Retrieval Baseline & Failure Analysis

A small experiment comparing lexical (TF-IDF) vs. semantic (sentence embeddings)
retrieval on a QA dataset, with a focus on understanding *when and why* each
approach fails.

## What this does

- **Part A:** Builds a TF-IDF retrieval baseline over a subset of SQuAD, evaluated
  with Recall@k.
- **Part B:** Tests the hypothesis that retrieval failures cluster around questions
  with low lexical overlap with their source paragraph, then compares TF-IDF
  against a sentence-embedding retriever to see whether semantic similarity
  closes that gap.

See `writeup.md` for the full analysis and findings.

## Setup

```bash
pip install -r requirements.txt
```

## How to run

Run these in order from the project root:

```bash
python src/load_data.py           # sanity-check: loads and inspects raw data
python src/build_corpus.py        # dedupes contexts into a search corpus
python src/retriever_tfidf.py     # sanity-check: TF-IDF on one example
python src/evaluate.py            # baseline Recall@1/3/5/10
python src/lexical_overlap.py     # buckets questions by lexical overlap, checks Recall@5 per bucket
python src/inspect_failures.py    # prints concrete low-overlap failure examples
python src/compare_retrievers.py  # TF-IDF vs. embeddings, side by side, per bucket
```

Note: `compare_retrievers.py` downloads a small (~90MB) sentence-transformers
model on first run — this is cached afterward.

## Project structure