import os
os.environ["WANDB_MODE"] = "disabled"

import math
import strictfire

from rt.main import main
from rt.tasks import all_tasks


def run_finetune_subsample(
    dataset: str = "rel-f1",
    task: str = "driver-position",
    float_ckpt_path: str = None,
    load_ckpt_path: str = None,
    embedding_model: str = "all-MiniLM-L12-v2",
    d_text: int = 384,
    batch_size: int = 64,
    lr: float = 1e-4,
    target_rows: int = 30000,
    epochs: int = 1,
    max_hop: int = 2,
    fanout_per_hop: str = "100,30",
    seq_len: int = 256,
    max_bfs_width: int = 256,
    num_workers: int = 4,
    compile_: bool = False,
):
    selected_task = None
    for item in all_tasks:
        db_name, table_name, target_col, leakage_cols = item
        if db_name == dataset and table_name == task:
            selected_task = item
            break
    if selected_task is None:
        raise ValueError(f"Task '{dataset}/{task}' not found in rt/tasks.py")

    steps = int(math.ceil(max(target_rows, 1) / max(batch_size, 1)))
    print(f"[SUBSAMPLE] target_rows={target_rows} batch_size={batch_size} => max_steps={steps}")

    main(
        project="rt_subsample",
        eval_splits=["val"],
        max_eval_steps=300,
        load_ckpt_path=load_ckpt_path,
        save_ckpt_dir="ckpts",
        compile_=compile_,
        seed=0,
        train_tasks=[selected_task],
        eval_tasks=[selected_task],
        batch_size=batch_size,
        num_workers=num_workers,
        max_bfs_width=max_bfs_width,
        lr=lr,
        wd=0.0,
        lr_schedule=False,
        max_grad_norm=1.0,
        embedding_model=embedding_model,
        d_text=d_text,
        seq_len=128,
        num_blocks=2,
        d_model=256,
        num_heads=8,
        d_ff=512,
        float_ckpt_path=float_ckpt_path,
        max_steps=steps,
        max_epochs=epochs,
        max_hop=max_hop,
        fanout_per_hop=fanout_per_hop,
    )


if __name__ == "__main__":
    strictfire.StrictFire(run_finetune_subsample)
