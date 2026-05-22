

import os
import torch
from db_transformer.nn.embedder.columns.griffin_float_embedder import create_pretrained_griffin_float_embedder


EMBED_DIM = 64
HIDDEN_DIM = 256
NUM_STEPS = 10000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = './pretrained_weights'
SAVE_FILENAME = 'float_embedder.pt'

def main():
    print("="*60)
    print("Griffin Float Embedder Pre-training")
    print("="*60)
    print(f"Embedding dimension: {EMBED_DIM}")
    print(f"Hidden dimension: {HIDDEN_DIM}")
    print(f"Training steps: {NUM_STEPS}")
    print(f"Device: {DEVICE}")
    print("="*60)
    print()


    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"✓ Save directory created/verified: {SAVE_DIR}")


    save_path = os.path.join(SAVE_DIR, SAVE_FILENAME)
    print(f"✓ Weights will be saved to: {save_path}")
    print()


    print("Starting pre-training...")
    print("-"*60)
    embedder = create_pretrained_griffin_float_embedder(
        dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_pretrain_steps=NUM_STEPS,
        device=DEVICE,
        save_path=save_path,
    )

    print("-"*60)
    print()
    print("="*60)
    print("✓ Pre-training completed successfully!")
    print("="*60)
    print(f"Weights saved to: {save_path}")
    print(f"File size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
    print()


    print("Testing weight loading...")
    from db_transformer.nn.embedder.columns.griffin_float_embedder import GriffinFloatEmbedder
    test_embedder = GriffinFloatEmbedder(
        dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        pretrained_path=save_path
    )
    print("✓ Weights loaded successfully!")
    print()

    print("="*60)
    print("Pre-training Complete!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. The weights are ready to use")
    print(f"2. Pass this path to your model: --griffin-float-weights {save_path}")
    print("3. Example usage:")
    print(f"   python main_relbench.py rel-f1 driver-top3 --use-griffin --griffin-float-weights {save_path}")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print()
        print("="*60)
        print("ERROR during pre-training!")
        print("="*60)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        exit(1)