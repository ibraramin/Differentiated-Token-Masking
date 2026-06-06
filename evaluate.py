import torch
import math
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import lm_eval

def calculate_bwt_perplexity(base_model_id, adapter_path=None, device="cuda"):
    """
    Calculates pure perplexity on the raw Wikitext-2 holdout set.
    Uses native Transformers (FA2) for mathematically exact CrossEntropy loss.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.bfloat16, 
        device_map=device, 
        attn_implementation="flash_attention_2"
    )
    
    # Dynamically load LoRA adapter if testing a fine-tuned model
    if adapter_path and adapter_path != base_model_id:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        
    model.eval()
    
    wiki_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    clean_text = [row['text'] for row in wiki_raw if len(row['text'].strip()) > 50][:1000]
    
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in clean_text:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            if inputs.input_ids.shape[1] < 2: continue
            outputs = model(inputs.input_ids, labels=inputs.input_ids)
            total_nll += (outputs.loss * inputs.input_ids.shape[1]).item()
            total_tokens += inputs.input_ids.shape[1]
            
    # Clean up memory before passing the baton to vLLM
    del model
    del tokenizer
    torch.cuda.empty_cache()
            
    return math.exp(total_nll / total_tokens)

def execute_evaluation_matrix():
    """
    Runs the zero-shot benchmarks (via vLLM) and absolute BWT differential analysis.
    """
    base_model_id = "Qwen/Qwen2.5-0.5B"
    
    print("Calculating Baseline Perplexity on Wikitext Holdout...")
    baseline_ppl = calculate_bwt_perplexity(base_model_id)
    print(f"Zero-Shot Baseline Perplexity: {baseline_ppl:.4f}")
    
    models_to_eval = {
        "Semantic_Control": ("./final_models/Sem_Control", "semantic"),
        "Semantic_Input_Masked": ("./final_models/Sem_InputMasked", "semantic"),
        "Semantic_Output_Masked": ("./final_models/Sem_OutputMasked", "semantic"),
        "Logical_Control": ("./final_models/Log_Control", "logical"),
        "Logical_Input_Masked": ("./final_models/Log_InputMasked", "logical"),
        "Logical_Output_Masked": ("./final_models/Log_OutputMasked", "logical"),
    }
    
    master_results = {"Baseline_PPL": baseline_ppl}
    
    for model_name, (adapter_path, track) in models_to_eval.items():
        print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")
        
        # MMLU included for semantic matrix completeness
        tasks = ["arc_challenge", "hellaswag", "mmlu"] if track == "semantic" else ["gsm8k", "math"]
        
        # --- NEW vLLM INTEGRATION ---
        # Construct model arguments to load the base model and the LoRA adapter dynamically
        vllm_args = f"pretrained={base_model_id},dtype=bfloat16,gpu_memory_utilization=0.7"
        if adapter_path != base_model_id:
            vllm_args += f",peft={adapter_path}"
            
        print(f"Booting vLLM engine for tasks: {tasks}...")
        task_results = lm_eval.simple_evaluate(
            model="vllm",
            model_args=vllm_args,
            tasks=tasks,
            num_fewshot=0,
            batch_size="auto"
        )['results']
        # ----------------------------
        
        tuned_ppl = calculate_bwt_perplexity(base_model_id, adapter_path)
        bwt_score = tuned_ppl - baseline_ppl
        
        master_results[model_name] = {
            "benchmarks": task_results, 
            "raw_perplexity": tuned_ppl, 
            "bwt_score": bwt_score
        }
        
        print(f"Raw PPL: {tuned_ppl:.4f} | BWT Score: {bwt_score:.4f}")

    with open("final_evaluation_matrix.json", "w") as f:
        json.dump(master_results, f, indent=4)
    print("\nEvaluation Matrix Complete. Results saved to final_evaluation_matrix.json")

if __name__ == "__main__":
    # execute_evaluation_matrix()
    pass
