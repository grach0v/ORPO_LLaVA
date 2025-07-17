import contextlib
import torch 

@contextlib.contextmanager
def temporary_padding_side(tokenizer, side):
    """Temporarily change padding side (left/right) inside a `with` block."""
    original = tokenizer.padding_side
    tokenizer.padding_side = side
    try:
        yield
    finally:
        tokenizer.padding_side = original


def build_prompt_inputs(images, questions, processor, DEVICE):
    """Tokenise the (question + image placeholder) prompt with left‑padding."""
    conversations = [
        [{"role": "user", "content": [{"type": "text", "text": q}, {"type": "image"}]}]
        for q in questions
    ]
    prompts = [processor.apply_chat_template(c, add_generation_prompt=True) for c in conversations]
    encoded = processor(images=images, text=prompts, padding=True, return_tensors="pt")
    return {k: v.to(DEVICE) for k, v in encoded.items()}


def tokenize_answers(texts, max_length, TOKENIZER, EOS_ID, DEVICE):
    """Right‑pad assistant answers and append EOS."""

    # Reason for the right pad tokenization -
    # later I will concatenate prompt tokens and potential answer tokens,
    # to get logits in one go, without writing a loop.
    # Having pad tokens in the middle seems very confusing
    # and can be misleading and couse errors in the future.

    with temporary_padding_side(TOKENIZER, "right"):
        encoded = TOKENIZER(
            texts,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    eos_column = torch.full((ids.size(0), 1), EOS_ID, dtype=ids.dtype)
    ids = torch.cat([ids, eos_column], dim=1)
    mask = torch.cat([mask, torch.ones_like(eos_column)], dim=1)

    if max_length is not None:
        # Trim if longer than max_length
        ids = ids[:, :max_length]
        mask = mask[:, :max_length]

    return ids.to(DEVICE), mask.to(DEVICE)


def collate_fn(batch, processor, DEVICE, MAX_ANSWER_TOKENS, TOKENIZER, EOS_ID):
    images = [item["image"] for item in batch]
    questions = [item["question"] for item in batch]
    chosen_texts = [item["chosen"] for item in batch]
    rejected_texts = [item["rejected"] for item in batch]

    prompt_inputs = build_prompt_inputs(images, questions, processor, DEVICE)
    chosen_ids, chosen_mask = tokenize_answers(chosen_texts, MAX_ANSWER_TOKENS, TOKENIZER, EOS_ID, DEVICE)
    rejected_ids, rejected_mask = tokenize_answers(rejected_texts, MAX_ANSWER_TOKENS, TOKENIZER, EOS_ID, DEVICE)

    return prompt_inputs, chosen_ids, chosen_mask, rejected_ids, rejected_mask