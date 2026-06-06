import os
import json
import logging
import pandas as pd
from scipy import stats
from tqdm import tqdm
import random
from masking_metrics import calculate_din, calculate_dout

logging.basicConfig(filename='curation_errors.log', level=logging.WARNING)

def generate_control_subset(data_list, output_name, target_size=5000):
    """[FIX D.3]: Generates random baseline model subset."""
    subset = random.sample(data_list, min(target_size, len(data_list)))
    pd.DataFrame(subset).to_json(f"curated_5k_{output_name}.json", orient='records', lines=True)
    print(f"[{output_name}] Curated random control subset.")

def process_and_score_dataset(model, tokenizer, dataset, track_type, strategy, output_file, nlp_spacy=None):
    start_index = 0
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            start_index = sum(1 for _ in f)
        print(f"Resuming {strategy} ({track_type}) from sample {start_index}...")

    with open(output_file, 'a') as f:
        for i in tqdm(range(start_index, len(dataset)), desc=f"Scoring {strategy}"):
            sample = dataset[i]
            prompt, target = sample["prompt"], sample["target"]
            try:
                if strategy == "Din":
                    score = calculate_din(model, tokenizer, prompt, target, nlp_spacy, track_type)
                elif strategy == "Dout":
                    score = calculate_dout(model, tokenizer, prompt, target)
                
                f.write(json.dumps({"prompt": prompt, "target": target, "raw_score": score}) + '\n')
                f.flush() 
            except Exception as e:
                # [FIX R.1]: Log exceptions instead of silent swallow
                logging.warning(f"Sample {i} failed ({strategy}): {str(e)}")
                continue 

    scored_data = []
    with open(output_file, 'r') as f:
        for line in f:
            scored_data.append(json.loads(line))
    return scored_data

def curate_optimal_subset(scored_data, strategy_name, target_size=5000):
    """
    [FIX D.1]: Statistically precise IQR filtering instead of positional slicing.
    """
    df = pd.DataFrame(scored_data)
    df['z_score'] = stats.zscore(df['raw_score'])
    
    # Calculate Interquartile Range
    q1 = df['z_score'].quantile(0.25)
    q3 = df['z_score'].quantile(0.75)
    
    iqr_subset = df[(df['z_score'] >= q1) & (df['z_score'] <= q3)].copy()
    
    # [FIX R.2]: Variance/Pool size check
    if len(iqr_subset) < target_size:
        print(f"⚠️ Warning: IQR pool ({len(iqr_subset)}) is smaller than target_size ({target_size}). Extracting available.")
        optimal_subset = iqr_subset
    else:
        optimal_subset = iqr_subset.sample(n=target_size, random_state=42)
    
    output_filename = f"curated_5k_{strategy_name}.json"
    optimal_subset[['prompt', 'target']].to_json(output_filename, orient='records', lines=True)
    
    print(f"[{strategy_name}] IQR Curated {len(optimal_subset)} samples. Z-range: [{q1:.3f}, {q3:.3f}]")
    return output_filename