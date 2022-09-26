from PIL import Image
import numpy as np
import torch
import os

from .model import FamNetOriginalCountRegressor, FamNetSimilarity

device = "cuda:0"

COUNTER_MODEL = f"{os.path.dirname(__file__)}/weights.pth"

model_density = None
model_similarity = None


@torch.no_grad()
def run(image: Image.Image, tlbr: np.ndarray) -> np.ndarray:
    global model_density, model_similarity

    if model_density is None:
        model_density = FamNetSimilarity()
        model_density.to(device=device)
    if model_similarity is None:
        model_similarity = FamNetOriginalCountRegressor(6)
        model_similarity.load_state_dict(torch.load(COUNTER_MODEL))
        model_similarity.to(device=device)

    try:
        density = model_density(image, tlbr)
        similarity = model_similarity(density)
    except Exception as e:
        raise e

    return similarity.cpu().detach().squeeze(0).squeeze(0).numpy()
