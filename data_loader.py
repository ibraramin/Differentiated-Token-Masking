from datasets import load_dataset

def load_curation_datasets(sample_limit=20000, tracks=None):
    if tracks is None:
        tracks = ["semantic", "logical"]

    semantic_data = []
    logical_data = []

    if "semantic" in tracks:
        print("Loading Semantic Track (Alpaca-Cleaned)...")
        semantic_raw = load_dataset("yahma/alpaca-cleaned", split="train")
        for row in semantic_raw.select(range(min(sample_limit, len(semantic_raw)))):
            prompt = f"{row['instruction']}\n\nInput:\n{row['input']}" if row.get("input") else row['instruction']
            semantic_data.append({"prompt": prompt, "target": row["output"]})

    if "logical" in tracks:
        print("Loading Logical Track (GSM8K)...")
        logical_raw = load_dataset("openai/gsm8k", "main", split="train")
        for row in logical_raw.select(range(min(sample_limit, len(logical_raw)))):
            logical_data.append({"prompt": row["question"], "target": row["answer"]})

    return semantic_data, logical_data

def load_held_out_perplexity_corpus():
    """
    Loads a pure, raw text corpus exclusively for testing Backward Transfer (BWT).
    This is completely isolated from the curation and training phases.
    """
    print("Isolating raw text hold-out set for BWT evaluation...")
    wiki_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    clean_text = [row['text'] for row in wiki_raw if len(row['text'].strip()) > 50]
    return clean_text[:1000]
