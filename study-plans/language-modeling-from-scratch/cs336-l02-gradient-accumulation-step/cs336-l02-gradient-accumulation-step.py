import torch

def gradient_accumulation_step(
   param, microbatch_inputs, microbatch_targets, lr):
    # Work on a detached copy so the caller's param and its .grad stay untouched
    w = param.detach().clone().requires_grad_(True)

    # Total number of examples across all microbatches
    N = sum(X.shape[0] for X in microbatch_inputs)

    for X, y in zip(microbatch_inputs, microbatch_targets):
        residual = X @ w - y                   # shape (Nm,)
        loss = (residual ** 2).sum() / N       # divide by total N, not Nm
        loss.backward()                        # gradients accumulate in w.grad

    full_grad = w.grad.detach().clone()

    # Exactly one SGD update
    with torch.no_grad():
        new_param = param.detach() - lr * full_grad

    return {"new_param": new_param, "full_grad": full_grad}

