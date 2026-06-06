import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

def execute_qlora_training(dataset_path, output_dir_name):
    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load custom tokens if training from curated set
    special_tokens_dict = {'mask_token': '<|mask|>'}
    if tokenizer.pad_token is None: special_tokens_dict['pad_token'] = '<|pad|>'
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.model_max_length = 1024

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)

    dataset = load_dataset('json', data_files=dataset_path, split='train')
    def format_and_tokenize(examples):
        texts = [f"{p}\n\n{t}{tokenizer.eos_token}" for p, t in zip(examples["prompt"], examples["target"])]
        return tokenizer(texts, truncation=True, max_length=1024)
    dataset = dataset.map(format_and_tokenize, batched=True, remove_columns=dataset.column_names)
    
    out_dir = f"./results/{output_dir_name}"
    os.makedirs(out_dir, exist_ok=True) # [FIX R.3] Safe dir creation
    
    # [FIX D.2]: Strictly adhering to methodology constraints for academic integrity
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        learning_rate=2e-5, num_train_epochs=4, lr_scheduler_type="cosine",
        logging_steps=10, save_strategy="epoch", optim="paged_adamw_32bit",
        bf16=True, max_grad_norm=1.0, warmup_steps=38
    )
    
    trainer = SFTTrainer(
        model=model, train_dataset=dataset, processing_class=tokenizer, args=training_args
    )
    
    # [FIX S.3]: Precise checkpoint tracking targeted to this specific model run
    checkpoint_to_resume = None
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        last_cp = get_last_checkpoint(out_dir)
        if last_cp: checkpoint_to_resume = last_cp

    print(f"Initiating training for {output_dir_name}...")
    trainer.train(resume_from_checkpoint=checkpoint_to_resume)
    
    final_out = f"./final_models/{output_dir_name}"
    os.makedirs(final_out, exist_ok=True)
    trainer.model.save_pretrained(final_out)
    tokenizer.save_pretrained(final_out)
    print(f"Training Complete for {output_dir_name}.")