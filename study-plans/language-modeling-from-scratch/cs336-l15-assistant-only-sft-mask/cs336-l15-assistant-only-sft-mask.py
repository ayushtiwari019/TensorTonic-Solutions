import torch

def assistant_only_sft_mask(input_ids, role_ids, attention_mask, assistant_role):
    attended = attention_mask.bool()

    # Target at position t+1 is eligible if it is attended and has the assistant role
    target_ok = attended[:, 1:] & (role_ids[:, 1:] == assistant_role)

    # The source position t must also be attended
    keep = attended[:, :-1] & target_ok

    # Start with every label ignored and an all-False mask (new tensors, inputs untouched)
    labels = torch.full_like(input_ids, -100)
    loss_mask = torch.zeros_like(attended, dtype=torch.bool)

    # Fill only the eligible source positions with the next token's id
    labels[:, :-1] = torch.where(keep, input_ids[:, 1:], labels[:, :-1])
    loss_mask[:, :-1] = keep

    return {"labels": labels, "loss_mask": loss_mask}