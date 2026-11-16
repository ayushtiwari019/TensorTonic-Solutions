import torch

def muon_spectral_update(parameter, gradient, previous_momentum,
                         momentum_coefficient, learning_rate):
    W = torch.as_tensor(parameter)
    if not W.is_floating_point():
        W = W.to(torch.float32)
    dtype, device = W.dtype, W.device

    G = torch.as_tensor(gradient, dtype=dtype, device=device)
    B_prev = torch.as_tensor(previous_momentum, dtype=dtype, device=device)

    # 1. Momentum: B_t = mu * B_{t-1} + G_t
    B = momentum_coefficient * B_prev + G

    # 2. Compact SVD (in float32/float64 for stability)
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    U, S, Vh = torch.linalg.svd(B.to(compute_dtype), full_matrices=False)

    # 3. Set the positive singular values to one: O = U V^T
    tol = max(B.shape) * torch.finfo(compute_dtype).eps * S.max()
    keep = S > tol
    O = (U[:, keep] @ Vh[keep, :]).to(dtype)

    # 4. Parameter update
    W_new = W - learning_rate * O

    return {
        "new_parameter": W_new,
        "new_momentum": B,
        "orthogonalized_update": O,
    }