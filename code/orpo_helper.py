import torch
from torch.nn.functional import log_softmax

def get_prompt_cache(model, prompt_inputs):
    """Get the last logits and past key values for the prompt inputs."""
    
    training  = model.training

    model.eval()
    with torch.no_grad():
        out = model(**prompt_inputs, return_dict=True)

    if training:
        model.train()

    return out.logits[:, -1:, :], out.past_key_values

def answer_logits(model, prompt_inputs, chosen_ids, chosen_mask, rejected_ids, rejected_mask):
    """Get logits for the chosen and rejected answers, aligned with the prompt inputs."""

    last_logits, prompt_kv = get_prompt_cache(model, prompt_inputs)

    # raw logits when we feed the full answers
    raw_chosen = model(
        input_ids=chosen_ids,
        attention_mask=chosen_mask,
        past_key_values=prompt_kv
    ).logits          # (B,N,V)

    raw_rejected = model(
        input_ids=rejected_ids,
        attention_mask=rejected_mask,
        past_key_values=prompt_kv
    ).logits          # (B,N,V)

    # align: prepend last_prompt_logits and drop the last timestep
    chosen_logits = torch.cat([last_logits, raw_chosen[:, :-1, :]],  dim=1)
    rejected_logits = torch.cat([last_logits, raw_rejected[:, :-1, :]], dim=1)

    return chosen_logits, rejected_logits

def token_logp(logits, ids):
    """Get log probabilities for the given token IDs from the logits."""
    logp = log_softmax(logits, dim=-1)
    return logp.gather(2, ids.unsqueeze(-1)).squeeze(-1)    # (B,N)

def log_prob(logs, mask):
    """Get the average log probability for the given logs and mask."""
    return (logs * mask).sum(dim=-1) / mask.sum(dim=-1)

def log_odds(log_prob):
    """Convert log probability to log odds."""
    return log_prob - torch.log1p(-torch.exp(log_prob))

def loss_orpo(chosen_logits, rejected_logits, chosen_ids, rejected_ids, chosen_mask, rejected_mask, lam):
    """Calculate the ORPO loss for the chosen and rejected logits."""
    chosen_logits = token_logp(chosen_logits, chosen_ids)   # (B,N)
    rejected_logits = token_logp(rejected_logits, rejected_ids)  # (B,N)

    chosen_logp = log_prob(chosen_logits, chosen_mask)  # (B,)
    rejected_logp = log_prob(rejected_logits, rejected_mask)  # (B,)
    
    log_odds_chosen = log_odds(chosen_logp)  # (B,)
    log_odds_rejected = log_odds(rejected_logp)  # (B,)

    L_sft = -chosen_logp.mean()  # supervised fine-tuning loss
    L_or = -torch.log(
        torch.sigmoid(log_odds_chosen - log_odds_rejected)
    ).mean()


    return L_sft + lam * L_or, L_sft, L_or