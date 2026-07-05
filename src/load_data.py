from datasets import load_dataset

def load_squad_sample(n=1500):
    """
    Loads a slice of the SQuAD validation set.

    Note: SQuAD's validation split is ordered by source article, so a small
    slice like the first 300 rows only covers ~15 unique articles/contexts,
    with ~20 questions per context. That's unrealistically easy/narrow for
    a retrieval task. We pull a larger slice (1500) to get a more realistic
    number of unique contexts to search across.
    """
    dataset = load_dataset("rajpurkar/squad", split=f"validation[:{n}]")
    return dataset

if __name__ == "__main__":
    data = load_squad_sample()
    print(f"Loaded {len(data)} examples")