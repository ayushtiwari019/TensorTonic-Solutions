import torch

def vision_transformer_patchify(images, patch_height, patch_width):
    B, C, H, W = images.shape
    gh, gw = H // patch_height, W // patch_width      # patch-grid size

    # Split H and W into (grid, within-patch) axes
    x = images.reshape(B, C, gh, patch_height, gw, patch_width)

    # Reorder to (batch, grid row, grid col, channel, local row, local col)
    x = x.permute(0, 2, 4, 1, 3, 5)

    # Flatten the grid into tokens and each patch into a vector
    tokens = x.reshape(B, gh * gw, C * patch_height * patch_width)

    # Patch-grid coordinates, row first then column
    rows = torch.arange(gh, dtype=torch.int64, device=images.device)
    cols = torch.arange(gw, dtype=torch.int64, device=images.device)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    coordinates = torch.stack([grid_r, grid_c], dim=-1).reshape(-1, 2)

    return {"tokens": tokens, "coordinates": coordinates}