import os
import torch
from rt.griffin_float_embedder import create_pretrained_griffin_float_embedder, GriffinFloatEmbedder

EMBED_DIM  = 64
HIDDEN_DIM = 256
NUM_STEPS  = 10000
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR   = "./pretrained_weights"
SAVE_FILE  = "float_embedder.pt"


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, SAVE_FILE)

    embedder = create_pretrained_griffin_float_embedder(
        dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
        num_pretrain_steps=NUM_STEPS, device=DEVICE, save_path=save_path,
    )

    test_embedder = GriffinFloatEmbedder(dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, pretrained_path=save_path)
    print(f"Weights saved: {save_path}  ({os.path.getsize(save_path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
