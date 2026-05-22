import os, shutil, glob, argparse, yaml
from pathlib import Path
import subprocess

def load_yaml(p): 
    with open(p, "r") as f: 
        return yaml.safe_load(f)

def save_yaml(obj, p):
    with open(p, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def copytree_exist_ok(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        s = item
        d = dst / item.name
        if s.is_dir():
            copytree_exist_ok(s, d)
        else:
            shutil.copy2(s, d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to full joint-v65 root (contains node/ edge/ task/ metanode.yaml ...)")
    ap.add_argument("--db_prefix", required=True, help='Prefix to keep, e.g. "rel-f1-" or "amazon-"')
    ap.add_argument("--task", required=True, help='Task name, e.g. "rel-f1-driver-position"')
    ap.add_argument("--out", default="datasets/filtered", help="Where to write filtered dataset root")
    ap.add_argument("--run_name", default="run_filtered", help="log/run name prefix")
    ap.add_argument("--savepath", default="checkpoints/filtered", help="checkpoint folder")
    ap.add_argument("--extra_args", nargs=argparse.REMAINDER, help="extra args passed to hmaintask_combine.py")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    out_db = out / f"{args.db_prefix.strip('-')}_only"
    if out_db.exists():
        shutil.rmtree(out_db)
    out_db.mkdir(parents=True)


    metanode = load_yaml(src / "metanode.yaml")
    metaadj  = load_yaml(src / "metaadj.yaml")
    metatask = load_yaml(src / "metatask.yaml")

    keep_nodes = {k for k in metanode.keys() if k.startswith(args.db_prefix)}
    keep_tasks = {k for k in metatask.keys() if k.startswith(args.db_prefix)}

    if args.task not in metatask:
        raise SystemExit(f"Task {args.task} not in metatask.yaml")
    if not args.task.startswith(args.db_prefix):
        raise SystemExit(f"Task {args.task} does not match db_prefix {args.db_prefix}")

    metanode_f = {k: v for k, v in metanode.items() if k in keep_nodes}
    metaadj_f  = {k: v for k, v in metaadj.items() if k in keep_nodes}
    metatask_f = {k: v for k, v in metatask.items() if k in keep_tasks}

    save_yaml(metanode_f, out_db / "metanode.yaml")
    save_yaml(metaadj_f,  out_db / "metaadj.yaml")
    save_yaml(metatask_f, out_db / "metatask.yaml")


    for fn in ["edgenameemb.pt", "featnameemb.pt", "tasknameemb.pt"]:
        shutil.copy2(src / fn, out_db / fn)


    (out_db / "node").mkdir()
    for nodetype in keep_nodes:
        src_node = src / "node" / nodetype
        dst_node = out_db / "node" / nodetype
        copytree_exist_ok(src_node, dst_node)


    (out_db / "edge").mkdir()
    for nodetype in keep_nodes:
        src_edge = src / "edge" / nodetype
        dst_edge = out_db / "edge" / nodetype
        if src_edge.exists():
            copytree_exist_ok(src_edge, dst_edge)


    (out_db / "task").mkdir()
    for t in keep_tasks:
        src_task = src / "task" / t
        dst_task = out_db / "task" / t
        copytree_exist_ok(src_task, dst_task)


    dataset_root = str(out_db)
    run = f"{args.run_name}_{args.task}"
    logdir = f"logs/{run}"

    cmd = [
        "accelerate", "launch", "--num_processes", "1",
        "hmaintask_combine.py",
        dataset_root, logdir, run,
        "--savepath", args.savepath,
        "--tasks", args.task,
    ]
    if args.extra_args:
        cmd += args.extra_args

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
