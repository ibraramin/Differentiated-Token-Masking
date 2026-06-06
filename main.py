import argparse
import spacy
import torch
import random
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_loader import load_curation_datasets
from curation import process_and_score_dataset, curate_optimal_subset, generate_control_subset
from train_qlora import execute_qlora_training
from evaluate import execute_evaluation_matrix

# [FIX E.2]: Global deterministic seeding for reproducibility
random.seed(42)
torch.manual_seed(42)

def setup_curation_env():
    print("Initializing Qwen-2.5-0.5B and SpaCy for Phase 3 Curation...")
    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # [FIX E.1]: Inject true mask token and pad token
    special_tokens_dict = {'mask_token': '<|mask|>'}
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = '<|pad|>'
    
    tokenizer.add_special_tokens(special_tokens_dict)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    ).eval()
    
    # [FIX E.1]: Resize token embeddings to accommodate the new mask token cleanly
    model.resize_token_embeddings(len(tokenizer))
    
    nlp = spacy.load("en_core_web_sm")
    return model, tokenizer, nlp

def main():
    parser = argparse.ArgumentParser(description="Qwen-2.5-0.5B Thesis Pipeline Orchestrator")
    parser.add_argument("--curate", action="store_true", help="Execute Phase 3: Data Curation")
    parser.add_argument("--train", action="store_true", help="Execute Phase 4: QLoRA Fine-Tuning")
    parser.add_argument("--evaluate", action="store_true", help="Execute Phase 5: vLLM Evaluation")
    args = parser.parse_args()

    if not any([args.curate, args.train, args.evaluate]):
        print("\n⚠️ No execution flags provided.")
        parser.print_help()
        return

    if args.curate:
        print("\n" + "="*50 + "\nPHASE 3: DATA SCORING AND CURATION\n" + "="*50)
        model, tokenizer, nlp = setup_curation_env()
        semantic_data, logical_data = load_curation_datasets(20000)
        
        print("\n--- Curating Semantic Track ---")
        # [FIX D.3]: Generate Control Models
        generate_control_subset(semantic_data, "Sem_Control")
        sem_din = process_and_score_dataset(model, tokenizer, semantic_data, "semantic", "Din", "scores_sem_din.jsonl", nlp)
        curate_optimal_subset(sem_din, "Sem_Input_Masked")
        sem_dout = process_and_score_dataset(model, tokenizer, semantic_data, "semantic", "Dout", "scores_sem_dout.jsonl")
        curate_optimal_subset(sem_dout, "Sem_Output_Masked")
        
        print("\n--- Curating Logical Track ---")
        generate_control_subset(logical_data, "Log_Control")
        log_din = process_and_score_dataset(model, tokenizer, logical_data, "logical", "Din", "scores_log_din.jsonl", nlp)
        curate_optimal_subset(log_din, "Log_Input_Masked")
        log_dout = process_and_score_dataset(model, tokenizer, logical_data, "logical", "Dout", "scores_log_dout.jsonl")
        curate_optimal_subset(log_dout, "Log_Output_Masked")
        
        del model
        torch.cuda.empty_cache()

    if args.train:
        print("\n" + "="*50 + "\nPHASE 4: QLORA FINE-TUNING\n" + "="*50)
        # Train Control Models
        execute_qlora_training("curated_5k_Sem_Control.json", "Sem_Control")
        execute_qlora_training("curated_5k_Log_Control.json", "Log_Control")
        # Train Masked Models
        execute_qlora_training("curated_5k_Sem_Input_Masked.json", "Sem_InputMasked")
        execute_qlora_training("curated_5k_Sem_Output_Masked.json", "Sem_OutputMasked")
        execute_qlora_training("curated_5k_Log_Input_Masked.json", "Log_InputMasked")
        execute_qlora_training("curated_5k_Log_Output_Masked.json", "Log_OutputMasked")
        torch.cuda.empty_cache()

    if args.evaluate:
        print("\n" + "="*50 + "\nPHASE 5: VLLM BENCHMARKING & BWT\n" + "="*50)
        execute_evaluation_matrix()

if __name__ == "__main__":
    main()