import os
os.environ["WANDB_MODE"] = "disabled"

from rt.main import main
from rt.tasks import all_tasks
import strictfire


def run_finetune(
    dataset: str = "rel-f1",
    task: str = "driver-top3",
    load_ckpt_path: str = None,
    embedding_model: str = "all-MiniLM-L12-v2",
    d_text: int = 384,
    batch_size: int = 16,
    lr: float = 1e-4,
    float_ckpt_path: str = None,
    epochs: int = None,
    steps: int = None,
    seed: int = 0,
    results_root: str = "results",
    max_hop: int = -1,
    fanout_per_hop: str = "",
    num_workers: int = 8,
    max_bfs_width: int = 64,
    seq_len: int = 256,
):
    selected_task = None
    for item in all_tasks:
        db_name, table_name, target_col, leakage_cols = item
        if db_name == dataset and table_name == task:
            selected_task = item
            break

    if selected_task is None:
        raise ValueError(f"Task '{dataset}/{task}' not found in tasks.py")

    if epochs is None and steps is None:
        steps = 32000

    max_epochs = epochs
    max_steps  = steps if epochs is None else None

    main(
        project="rt",
        eval_splits=("val",),
        max_eval_steps=30,
        train_steps_per_epoch=1000,
        load_ckpt_path=load_ckpt_path,
        save_ckpt_dir="ckpts",
        compile_=False,
        seed=seed,
        results_root=results_root,
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
        seq_len=seq_len,
        num_blocks=12,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        float_ckpt_path=float_ckpt_path,
        max_steps=max_steps,
        max_epochs=max_epochs,
        max_hop=max_hop,
        fanout_per_hop=fanout_per_hop,
    )


if __name__ == "__main__":
    strictfire.StrictFire(run_finetune)
