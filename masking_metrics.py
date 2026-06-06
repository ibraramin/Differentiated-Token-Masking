import torch
import torch.nn.functional as F
import random

def _get_target_logprobs(model, tokenizer, prompt, target):
    """
    [FIX E.4]: Combines text before tokenizing to preserve natural BPE subword boundaries,
    then isolates the target log probabilities natively.
    """
    full_text = prompt + " " + target
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)
    prompt_ids = tokenizer(prompt + " ", return_tensors="pt").input_ids.to(model.device)
    
    target_len = full_ids.shape[1] - prompt_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(full_ids)
        
    logits = outputs.logits
    # [FIX E.6]: Batched slicing instead of hardcoded [0, ...]
    shift_logits = logits[:, -target_len-1:-1, :]
    shift_labels = full_ids[:, -target_len:]
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    target_log_probs = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    return target_log_probs[0] # Return the 1D tensor for this single sequence

def calculate_din(model, tokenizer, prompt, target, nlp_spacy, track_type="semantic"):
    base_logprobs = _get_target_logprobs(model, tokenizer, prompt, target)
    
    doc = nlp_spacy(prompt)
    target_tokens = []
    
    if track_type == "semantic":
        target_tokens = [t for t in doc if t.pos_ in ['NOUN', 'VERB', 'ADJ']]
    elif track_type == "logical":
        math_keywords = {"calculate", "derive", "solve", "find", "compute", "how"}
        target_tokens = [t for t in doc if t.pos_ == 'NUM' or t.text.lower() in math_keywords]
        if len(target_tokens) < 2:
             target_tokens.extend([t for t in doc if t.pos_ in ['NOUN', 'VERB']])

    num_to_mask = max(1, int(len(target_tokens) * 0.20))
    masked_prompt = prompt
    mask_token_str = tokenizer.mask_token
    
    if target_tokens:
        tokens_to_mask = random.sample(target_tokens, min(num_to_mask, len(target_tokens)))
        # [FIX E.3]: Sort descending by index and slice to prevent corrupting substrings
        tokens_to_mask = sorted(tokens_to_mask, key=lambda x: x.idx, reverse=True)
        for t in tokens_to_mask:
            masked_prompt = masked_prompt[:t.idx] + mask_token_str + masked_prompt[t.idx + len(t.text):]
            
    masked_logprobs = _get_target_logprobs(model, tokenizer, masked_prompt, target)
    
    # Handle length mismatches gracefully if subword tokenization shifts
    min_len = min(base_logprobs.shape[0], masked_logprobs.shape[0])
    return torch.mean(base_logprobs[:min_len] - masked_logprobs[:min_len]).item()

def calculate_dout(model, tokenizer, prompt, target):
    """
    [FIX E.5]: M_out(y) masks target tokens with a true mask_token_id, not EOS.
    """
    base_logprobs = _get_target_logprobs(model, tokenizer, prompt, target)
    
    full_text = prompt + " " + target
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)
    prompt_ids = tokenizer(prompt + " ", return_tensors="pt").input_ids.to(model.device)
    target_len = full_ids.shape[1] - prompt_ids.shape[1]
    
    masked_full_ids = full_ids.clone()
    prompt_len = prompt_ids.shape[1]
    seq_len = masked_full_ids.shape[1]
    
    block_size = random.randint(3, 5)
    # Uses the injected mask token, avoiding semantic EOS termination
    mask_token_id = tokenizer.mask_token_id 
    
    for i in range(prompt_len + block_size, seq_len - block_size, block_size * 3):
        end_idx = min(i + block_size, seq_len)
        masked_full_ids[:, i:end_idx] = mask_token_id
        
    with torch.no_grad():
        outputs = model(masked_full_ids)
        
    logits = outputs.logits
    shift_logits = logits[:, -target_len-1:-1, :]
    shift_labels = full_ids[:, -target_len:] 
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    masked_logprobs = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1)[0]
    
    min_len = min(base_logprobs.shape[0], masked_logprobs.shape[0])
    return torch.mean(base_logprobs[:min_len] - masked_logprobs[:min_len]).item()