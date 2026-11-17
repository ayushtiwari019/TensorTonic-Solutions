import torch

def rmsnorm(x: torch.Tensor, g: torch.Tensor, epsilon: float) -> torch.Tensor:
    x32 = x.to(torch.float32)
    g32 = g.to(torch.float32)

    # Mean of squares over the final dimension, then add epsilon and take the root
    ms = x32.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(ms + epsilon)

    # Normalize, then scale each feature by g
    out = x32 * inv_rms * g32

    return out.to(x.dtype)
    """
    Returns the RMS-normalized tensor with the same shape and dtype as x.
    """
    pass
