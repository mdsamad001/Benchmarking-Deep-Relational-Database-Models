import os, argparse
import numpy as np
import datasets as hds
from datasets import Dataset
from sentence_transformers import SentenceTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_path", required=True, help=".../node/<nodetype>/feat")
    ap.add_argument("--out_path", required=True, help=".../node/<nodetype>/textemb")
    ap.add_argument("--hiddim", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    ds = hds.load_from_disk(args.feat_path)


    text_cols = [c for c in ds.column_names if "Griffin_text_" in c]
    if len(text_cols) == 0:
        raise SystemExit("No Griffin_text_ columns found in feat dataset.")

    print("Text cols:", len(text_cols))


    enc = SentenceTransformer(args.model, device="cuda")


    def row_to_text(example):
        parts = []
        for c in text_cols:
            v = example.get(c, "")
            if v is None:
                v = ""
            parts.append(f"{c.split('___')[-1]}: {str(v)}")
        return {"_joined_text": " | ".join(parts)}

    ds2 = ds.map(row_to_text, num_proc=1)

    texts = ds2["_joined_text"]
    embeddings = enc.encode(
        texts,
        batch_size=args.batch,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)


    d = embeddings.shape[1]
    if d != args.hiddim:
        print(f"[WARN] encoder dim={d}, requested hiddim={args.hiddim}. Adjusting...")
        if d > args.hiddim:
            embeddings = embeddings[:, : args.hiddim]
        else:
            pad = np.zeros((embeddings.shape[0], args.hiddim - d), dtype=np.float32)
            embeddings = np.concatenate([embeddings, pad], axis=1)

    out = Dataset.from_dict({"emb": [e for e in embeddings]})
    os.makedirs(args.out_path, exist_ok=True)
    out.save_to_disk(args.out_path)
    print("Saved to:", args.out_path)

if __name__ == "__main__":
    main()
