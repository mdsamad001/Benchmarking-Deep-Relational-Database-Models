import argparse, os
import numpy as np
import datasets as hds
from datasets import Dataset
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--hiddim", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init", type=str, default="normal", choices=["normal", "zeros"])
    ap.add_argument("--batch_read", type=int, default=4096)
    args = ap.parse_args()

    ds = hds.load_from_disk(args.feat_path)

    text_cols = [c for c in ds.column_names if "Griffin_text_" in c]
    if len(text_cols) == 0:
        raise SystemExit("No Griffin_text_ columns found.")


    max_id = -1
    it = ds.to_iterable_dataset()
    for ex in tqdm(it, desc="Scanning max text id"):
        for c in text_cols:
            v = ex.get(c, None)
            if v is None:
                continue

            try:
                iv = int(v)
            except:
                continue
            if iv > max_id:
                max_id = iv

    if max_id < 0:
        raise SystemExit("Could not find any valid text ids.")

    vocab_size = max_id + 1
    print(f"Found {len(text_cols)} Griffin_text_ cols, max_id={max_id}, vocab_size={vocab_size}")

    rng = np.random.default_rng(args.seed)
    if args.init == "normal":
        emb = rng.normal(loc=0.0, scale=0.02, size=(vocab_size, args.hiddim)).astype(np.float32)
    else:
        emb = np.zeros((vocab_size, args.hiddim), dtype=np.float32)

    out = Dataset.from_dict({"emb": [e for e in emb]})
    os.makedirs(args.out_path, exist_ok=True)
    out.save_to_disk(args.out_path)
    print("Saved vocab textemb to:", args.out_path)

if __name__ == "__main__":
    main()
