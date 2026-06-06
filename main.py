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

random.seed(42)
torch.manual_seed(42)

CURATION_TASKS = [
    ("semantic", "control",      "ctrl", None,      None,    None,   None,  "Sem_Control"),
    ("semantic", "input_masked", "din",  "semantic", "Din",   "scores_sem_din.jsonl",  True,  "Sem_Input_Masked"),
    ("semantic", "output_masked","dout", "semantic", "Dout",  "scores_sem_dout.jsonl", False, "Sem_Output_Masked"),
    ("logical",  "control",      "ctrl", None,      None,    None,   None,  "Log_Control"),
    ("logical",  "input_masked", "din",  "logical",  "Din",  "scores_log_din.jsonl",  True,  "Log_Input_Masked"),
    ("logical",  "output_masked","dout", "logical",  "Dout", "scores_log_dout.jsonl", False, "Log_Output_Masked"),
]

TRAIN_TASKS = [
    ("semantic", "control",      "curated_5k_Sem_Control.json",         "Sem_Control"),
    ("semantic", "input_masked",  "curated_5k_Sem_Input_Masked.json",   "Sem_InputMasked"),
    ("semantic", "output_masked", "curated_5k_Sem_Output_Masked.json",  "Sem_OutputMasked"),
    ("logical",  "control",      "curated_5k_Log_Control.json",         "Log_Control"),
    ("logical",  "input_masked",  "curated_5k_Log_Input_Masked.json",   "Log_InputMasked"),
    ("logical",  "output_masked", "curated_5k_Log_Output_Masked.json",  "Log_OutputMasked"),
]

EVAL_MAP = {
    ("semantic", "control"):      ("Semantic_Control",      "./final_models/Sem_Control"),
    ("semantic", "input_masked"):  ("Semantic_Input_Masked",  "./final_models/Sem_InputMasked"),
    ("semantic", "output_masked"): ("Semantic_Output_Masked", "./final_models/Sem_OutputMasked"),
    ("logical",  "control"):      ("Logical_Control",      "./final_models/Log_Control"),
    ("logical",  "input_masked"):  ("Logical_Input_Masked",  "./final_models/Log_InputMasked"),
    ("logical",  "output_masked"): ("Logical_Output_Masked", "./final_models/Log_OutputMasked"),
}

def _active(args, track, strategy):
    track_ok = args.track in ("all", track)
    strat_ok = args.strategy in ("all", strategy)
    return track_ok and strat_ok


def setup_curation_env():
    print("Initializing Qwen-2.5-0.5B and SpaCy for Curation...")
    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    special_tokens_dict = {'mask_token': '<|mask|>'}
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = '<|pad|>'
    tokenizer.add_special_tokens(special_tokens_dict)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()

    model.resize_token_embeddings(len(tokenizer))

    nlp = spacy.load("en_core_web_sm")
    return model, tokenizer, nlp


def main():
    parser = argparse.ArgumentParser(description="Qwen-2.5-0.5B Thesis Pipeline Orchestrator")
    parser.add_argument("--curate",   action="store_true", help="Phase 1: Data Curation")
    parser.add_argument("--train",    action="store_true", help="Phase 2: QLoRA Fine-Tuning")
    parser.add_argument("--evaluate", action="store_true", help="Phase 3: vLLM Evaluation")
    parser.add_argument("--track",    choices=["all", "semantic", "logical"], default="all",
                        help="Restrict to one cognitive track")
    parser.add_argument("--strategy", choices=["all", "control", "input_masked", "output_masked"], default="all",
                        help="Restrict to one masking strategy (or control)")
    args = parser.parse_args()

    if not any([args.curate, args.train, args.evaluate]):
        print("\nNo execution flags provided.")
        parser.print_help()
        return

    # ── Phase 1: Curation ──────────────────────────────────────────
    if args.curate:
        print("\n" + "=" * 50 + "\nPHASE 1: DATA SCORING & CURATION\n" + "=" * 50)
        print(f"  Track: {args.track}  |  Strategy: {args.strategy}")

        needs_model = any(
            _active(args, t, s) and kind in ("din", "dout")
            for t, s, kind, *_ in CURATION_TASKS
        )
        model = tokenizer = nlp = None
        if needs_model:
            model, tokenizer, nlp = setup_curation_env()

        semantic_data, logical_data = load_curation_datasets(20000)
        data_bank = {"semantic": semantic_data, "logical": logical_data}
        score_cache = {}

        for (track, strat, kind, tt, st, out_f, needs_nlp, out_name) in CURATION_TASKS:
            if not _active(args, track, strat):
                continue
            print(f"\n--- {track.title()} / {strat.replace('_',' ').title()} ---")
            if kind == "ctrl":
                generate_control_subset(data_bank[track], out_name)
            elif kind in ("din", "dout"):
                scores = process_and_score_dataset(
                    model, tokenizer, data_bank[track],
                    tt, st, out_f, nlp if needs_nlp else None
                )
                score_cache[out_name] = scores
                curate_optimal_subset(scores, out_name)
            else:
                continue

        if model is not None:
            del model
        torch.cuda.empty_cache()

    # ── Phase 2: Training ──────────────────────────────────────────
    if args.train:
        print("\n" + "=" * 50 + "\nPHASE 2: QLORA FINE-TUNING\n" + "=" * 50)
        print(f"  Track: {args.track}  |  Strategy: {args.strategy}")

        for track, strat, dataset_path, output_dir in TRAIN_TASKS:
            if not _active(args, track, strat):
                continue
            print(f"\n--- {track.title()} / {strat.replace('_',' ').title()} ---")
            execute_qlora_training(dataset_path, output_dir)

        torch.cuda.empty_cache()

    # ── Phase 3: Evaluation ──────────────────────────────────────────
    if args.evaluate:
        print("\n" + "=" * 50 + "\nPHASE 3: VLLM BENCHMARKING & BWT\n" + "=" * 50)
        print(f"  Track: {args.track}")

        models_to_eval = {}
        for (track, strat), (name, path) in EVAL_MAP.items():
            if _active(args, track, strat):
                models_to_eval[name] = (path, track)

        execute_evaluation_matrix(models_to_eval)


if __name__ == "__main__":
    main()
