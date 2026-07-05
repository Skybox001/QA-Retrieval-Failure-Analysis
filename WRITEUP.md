# QA Retrieval Baseline & Failure Analysis
Research & Solution Development Intern — Take-Home

## Setup

I used a subset of SQuAD's validation set (1,500 question–context pairs,
deduplicated down to 195 unique paragraphs) as a small but realistic corpus
to search over. I chose a subset mainly for iteration speed SQuAD's
validation split is ordered by source article, so a naive small slice was
almost all repeat questions on ~15 paragraphs. I pulled a bigger slice and
deduplicated contexts to get a corpus size that's actually representative
of a search problem.

## Baseline (Part A)

For Part A I built a TF-IDF retriever (word-overlap + cosine similarity) —
deliberately simple, since the point of a baseline is correctness and
reproducibility, not sophistication.

| Metric | Score |
|---|---|
| Recall@1 | 0.615 |
| Recall@5 | 0.871 |
| Recall@10 | 0.921 |

The gap between Recall@1 and Recall@10 was the first interesting signal
it suggested the retriever often finds the right paragraph but doesn't rank
it as the top choice, which pointed me toward looking at *when* it ranks
things wrong rather than just *how often*.

## Hypothesis (Part B)

I suspected the failures weren't random, but concentrated on questions that
reuse very little of the paragraph's actual wording i.e., paraphrased
questions, since TF-IDF can only match surface words. I measured this
directly: for each question, what fraction of its (non-trivial) words also
appear in the correct paragraph, then bucketed questions into low/medium/high
overlap and checked Recall@5 in each bucket.

| Overlap bucket | n | Recall@5 |
|---|---|---|
| Low (<0.3) | 143 | 0.427 |
| Medium (0.3–0.6) | 705 | 0.864 |
| High (>0.6) | 652 | 0.975 |

The result was a clear, monotonic trend more than a 2x difference in
accuracy between the extremes, on about 1 in 10 questions in the set.

## What actually turned out to be going on

When I pulled real examples from the failing low-overlap bucket, the
mechanism wasn't quite what I expected. Most of these weren't paraphrase
problems (different words, same meaning) they were cases where *multiple
paragraphs shared the same proper nouns and topic* (e.g. several different
paragraphs about "the Panthers" or "Super Bowl 50," each covering a
different specific fact season record vs. defensive stats vs. game
play-by-play). TF-IDF can't tell these apart because the overlapping words
are the same across all the candidates; it has no way to know which
specific fact each paragraph is about.

## Comparison against a semantic retriever

To test whether this really was a "lexical vs. meaning" problem, I built a
second retriever using sentence embeddings (`all-MiniLM-L6-v2`) and re-ran
the same bucketed evaluation:

| Bucket | TF-IDF | Embeddings |
|---|---|---|
| Low overlap | 0.427 | 0.755 |
| Medium overlap | 0.864 | 0.881 |
| High overlap | 0.975 | 0.923 |

Two things stood out. First, embeddings recovered a big chunk of the
low-overlap failures (+32.8 points), which supports the theory — semantic
similarity can disambiguate paragraphs that share vocabulary but differ in
meaning. Second, and more surprising: TF-IDF actually *beat* embeddings on
high-overlap questions (0.975 vs 0.923). My read is that when a question
shares specific, rare terms (names, numbers, exact phrases) with its
paragraph, literal matching is a stronger and more precise signal than
semantic similarity the embedding model's generalization can blur
distinctions that exact matching gets right for free.

## Why this matters

There isn't a single "best" retriever here the right choice depends on
the query. That's a real, practically relevant takeaway: a production
retrieval system probably shouldn't commit to one method globally, but
could route between lexical and semantic retrieval using a signal like
lexical overlap itself.

## Next step

With more time, I'd build a simple hybrid retriever combine TF-IDF and
embedding similarity scores (or route between them using the overlap score
as a switch) and test whether it beats both individual methods across
all three buckets, not just on average.