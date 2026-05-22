import os
import os.path as osp
import glob
import warnings
from typing import Optional, Iterable, Union, Literal, Any

import yaml
import torch
import datasets as hds
from datasets import load_dataset
import numpy as np

INF = 100000000
MAXINT64 = 1 << 62
TIMESTAMPADJNAME = "___TIMESTAMP"


def _list_arrow_files(folder: str) -> list[str]:
    return sorted(glob.glob(osp.join(folder, "data-*-of-*.arrow")))


def load_arrow_folder(folder: str) -> hds.Dataset:
    files = _list_arrow_files(folder)
    if len(files) == 0:
        raise FileNotFoundError(f"No arrow shards found in {folder}")
    return load_dataset("arrow", data_files=files, split="train")


def load_hf_folder(folder: str) -> hds.Dataset:
    info = osp.join(folder, "dataset_info.json")
    state = osp.join(folder, "state.json")
    if osp.exists(info) and osp.exists(state):
        return hds.load_from_disk(folder)


    shard_dirs = sorted(
        [p for p in glob.glob(osp.join(folder, "shard_*")) if osp.isdir(p)]
    )
    if len(shard_dirs) > 0:

        datasets = []
        for sd in shard_dirs:
            sd_info = osp.join(sd, "dataset_info.json")
            sd_state = osp.join(sd, "state.json")
            if osp.exists(sd_info) and osp.exists(sd_state):
                ds = hds.load_from_disk(sd)
            else:

                ds = load_arrow_folder(sd)
            datasets.append(ds)

        if len(datasets) == 1:
            return datasets[0]
        return hds.concatenate_datasets(datasets)


    return load_arrow_folder(folder)


def _to_hf_index(idx: Any):
    if isinstance(idx, range):
        return list(idx)

    if isinstance(idx, int):
        return idx

    if isinstance(idx, (list, tuple)):
        return list(idx)

    if isinstance(idx, np.ndarray):
        if idx.ndim == 0:
            return int(idx.item())
        return idx.astype(np.int64).tolist()

    if torch.is_tensor(idx):
        if idx.ndim == 0:
            return int(idx.item())
        return idx.detach().cpu().to(torch.long).tolist()

    raise TypeError(f"Unsupported idx type: {type(idx)}")


class Node:
    meta: dict
    feat: hds.Dataset
    textemb: Union[hds.Dataset, None]
    adj: Union[hds.Dataset, None]

    def __init__(self, meta, feat, textemb, adj) -> None:
        self.meta = meta
        self.feat = feat
        self.textemb = textemb
        self.adj = adj


        assert len(self.feat) == int(self.meta["num"]), (
            f"[Node:{self.meta.get('name','?')}] feat has {len(self.feat)} while meta says {self.meta['num']} nodes "
            f"(feat path maybe not loaded correctly?)"
        )
        if self.adj is not None:
            assert len(self.adj) == int(self.meta["num"]), (
                f"[Node:{self.meta.get('name','?')}] adj has {len(self.adj)} while meta says {self.meta['num']} nodes"
            )

    def __len__(self):
        return int(self.meta["num"])

    @property
    def is_target(self):
        return bool(self.meta["is_target"])

    @property
    def featlist(self):
        return self.meta["feat"]

    def getfeat(self, idx: Union[int, Iterable[int], range, torch.Tensor, np.ndarray], floatemb) -> torch.Tensor:
        idx_hf = _to_hf_index(idx)
        data: dict = self.feat[idx_hf]


        needs_text = any(("Griffin_text_" in col) for col in self.featlist)
        if needs_text and self.textemb is None:
            raise RuntimeError(
                "Node has Griffin_text_ features but textemb is missing. "
                "Check datasets/.../node/<nodetype>/textemb"
            )

        def unique_query_textemb(text_idx: torch.Tensor):
            unique_idx, inv = torch.unique(text_idx, return_inverse=True)
            return self.textemb[_to_hf_index(unique_idx)]["emb"][inv]

        def unique_float_emb(val: torch.Tensor):
            unique_val, inv = torch.unique(val, return_inverse=True)
            return floatemb(unique_val)[inv]

        out = torch.stack(
            [
                unique_query_textemb(data[col]) if "Griffin_text_" in col else unique_float_emb(data[col])
                for col in self.featlist
            ],
            dim=1,
        )
        return out

    def getedge(self, idx: torch.LongTensor, fanout: int = INF, timestamp: list[int] = None):


        if self.adj is None:
            return {}

        num_q = len(idx)
        subadj: dict = self.adj[_to_hf_index(idx)]
        subadj.pop("number", None)

        hastimestamp = timestamp is not None
        if hastimestamp:
            assert len(timestamp) == num_q

        ret = {}
        for key in list(subadj.keys()):
            if key.endswith(TIMESTAMPADJNAME):
                continue

            adj = subadj[key]
            if len(adj) == 0:
                continue

            if not isinstance(adj, torch.Tensor):

                assert isinstance(adj, list)
                adj = torch.nn.utils.rnn.pad_sequence(adj, batch_first=True, padding_value=-1)

            if len(adj.flatten()) == 0:
                continue

            assert adj.ndim == 2, f"{adj.shape}"
            assert adj.shape[0] == num_q

            rootnode = torch.arange(num_q).reshape(-1, 1).repeat(1, adj.shape[1])

            if hastimestamp:
                ts_key = key + TIMESTAMPADJNAME
                if ts_key in subadj:
                    adjtimestamp = subadj[ts_key]
                    if not isinstance(adjtimestamp, torch.Tensor):
                        adjtimestamp = torch.nn.utils.rnn.pad_sequence(adjtimestamp, batch_first=True, padding_value=-1)
                    assert adjtimestamp.ndim == 2
                    assert adjtimestamp.shape[0] == num_q
                    adj = adj.clone()
                    ts = timestamp
                    if ts is None:
                        pass
                    else:
                        if isinstance(ts, torch.Tensor):
                            ts_t = ts.to(device=adjtimestamp.device)
                        else:

                            ts_t = torch.as_tensor(ts, device=adjtimestamp.device)
                    

                        if ts_t.ndim == 1:
                            ts_t = ts_t.view(-1, 1)
                        elif ts_t.ndim == 2 and ts_t.shape[1] != 1:
                            ts_t = ts_t[:, :1]
                        else:
                            ts_t = ts_t.view(-1, 1)
                    

                        if ts_t.dtype != adjtimestamp.dtype:
                            ts_t = ts_t.to(adjtimestamp.dtype)
                    
                        adj.masked_fill_(adjtimestamp >= ts_t, -1)


            if adj.shape[1] > fanout:
                rank_val = torch.rand_like(adj, dtype=torch.float)
                rank_val.masked_fill_(adj < 0, -1000.0)
                topk_ind = torch.topk(rank_val, fanout, dim=-1)[1]
                adj = torch.gather(adj, 1, topk_ind)
                rootnode = rootnode[:, :fanout]

            mask = adj >= 0
            rootnode, adj = rootnode[mask], adj[mask]
            if len(rootnode) == 0:
                continue

            ret[key] = torch.stack((rootnode, adj), dim=0)
        return ret


def edgename2tail(edgename: str):
    if edgename.startswith("head of "):
        return edgename.split(":")[-1]
    elif edgename.startswith("tail of "):
        return edgename.split(":")[0].removeprefix("tail of ")
    else:
        raise NotImplementedError(f"cannot parse edgename: {edgename}")


def edgename2head(edgename: str):
    if edgename.startswith("head of "):
        return edgename.split(":")[0].removeprefix("head of ")
    elif edgename.startswith("tail of "):
        return edgename.split(":")[-1]
    else:
        raise NotImplementedError(f"cannot parse edgename: {edgename}")


class Graph:
    def __init__(self, path) -> None:
        with open(osp.join(path, "metanode.yaml")) as f:
            metanode = yaml.safe_load(f)
        with open(osp.join(path, "metaadj.yaml")) as f:
            metaadj = yaml.safe_load(f)

        for nodetype in metanode:
            metanode[nodetype].update(metaadj.get(nodetype, {}))
            metanode[nodetype]["name"] = nodetype

        self.metanode = metanode


        def _safe_load_pt(fp: str):
            try:
                return torch.load(fp, map_location="cpu", weights_only=True)
            except TypeError:
                return torch.load(fp, map_location="cpu")

        self.edgenameemb = _safe_load_pt(osp.join(path, "edgenameemb.pt"))
        self.featnameemb = _safe_load_pt(osp.join(path, "featnameemb.pt"))


        if "fewshot" not in self.edgenameemb:

            any_key = next(iter(self.edgenameemb.keys()))
            dim = int(self.edgenameemb[any_key].numel())
            self.edgenameemb["fewshot"] = torch.zeros(dim, dtype=self.edgenameemb[any_key].dtype)
            warnings.warn("[WARN] edgenameemb missing 'fewshot' -> injected zeros embedding")

        self.nodes = {}


        for nodetype in self.metanode:
            feat_path = osp.join(path, "node", nodetype, "feat")
            text_path = osp.join(path, "node", nodetype, "textemb")
            adj_path = osp.join(path, "edge", nodetype, "adj")


            feat_ds = load_hf_folder(feat_path).with_format("torch")


            text_ds = None
            if osp.exists(text_path):
                text_ds = load_hf_folder(text_path).with_format("torch")


            adj_ds = None
            if len(self.metanode[nodetype].get("in", []) + self.metanode[nodetype].get("out", [])) > 0:
                if osp.exists(adj_path):
                    adj_ds = load_hf_folder(adj_path).with_format("torch")
                else:
                    self.metanode[nodetype]["in"] = []
                    self.metanode[nodetype]["out"] = []
                    adj_ds = None

            self.nodes[nodetype] = Node(self.metanode[nodetype], feat_ds, text_ds, adj_ds)


        self.ensure_number_node(num_needed=2)

    def ensure_number_node(self, num_needed: int = 2):
        num_needed = int(num_needed)
        if num_needed < 2:
            num_needed = 2


        if "number" in self.nodes:
            meta = self.metanode.get("number", {})
            featlist = meta.get("feat", [])
            if len(featlist) == 1 and len(self.nodes["number"].feat) >= num_needed:
                return


        ds = hds.Dataset.from_dict(
            {"Griffin_float_0": list(range(num_needed))}
        ).with_format("torch")

        meta = {
            "name": "number",
            "num": len(ds),
            "is_target": False,
            "feat": ["Griffin_float_0"],
            "in": [],
            "out": [],
        }
        self.metanode["number"] = meta
        self.nodes["number"] = Node(meta, ds, None, None)


        if "Griffin_float_0" not in self.featnameemb:
            any_key = next(iter(self.featnameemb.keys()))
            dim = int(self.featnameemb[any_key].numel())
            self.featnameemb["Griffin_float_0"] = torch.zeros(dim, dtype=self.featnameemb[any_key].dtype)

    def subgraph(
        self,
        root_nodetype: str,
        root_nodeidx: Iterable[int],
        hop: int,
        floatemb,
        fanout: Union[int, list[int]] = INF,
        timestamp: Union[list[int], None] = None,
    ):
        hastimestamp: bool = timestamp is not None
        adj = {}
        node = {}
        root = {root_nodetype: root_nodeidx}
        roottimestamp = {root_nodetype: timestamp}


        if hop == 0:
            fanout_list: list[int] = []
        elif isinstance(fanout, int):
            fanout_list = [fanout] * hop
        else:
            fanout_list = list(fanout)
            if len(fanout_list) == 1 and hop > 1:
                fanout_list = fanout_list * hop
            if len(fanout_list) != hop:
                raise ValueError(f"fanout list length must equal hop. hop={hop}, fanout={fanout_list}")

        def dictgetlen(d: dict[str, torch.Tensor], key: str, dim: int = 0):
            if key not in d:
                return 0
            return d[key].shape[dim]

        def dictupdate(
            d: dict[str, torch.Tensor],
            key: str,
            value: torch.Tensor,
            concatdim: int = 0,
        ):
            if key not in d:
                d[key] = value
            else:
                d[key] = torch.concat((d[key], value), dim=concatdim)
            return d

        for h in range(hop):
            cur_fanout = fanout_list[h] if h < len(fanout_list) else INF
            nroot, nroottimestamp = {}, {}
            for nodetype in root:
                found_src_num = dictgetlen(node, nodetype)
                ttadj = self.nodes[nodetype].getedge(
                    root[nodetype], cur_fanout, roottimestamp[nodetype]
                )
                for edgetype in ttadj:
                    srcidx, taridx = ttadj[edgetype][0], ttadj[edgetype][1]

                    tartype = edgename2tail(edgetype)
                    found_tar_num = (
                        dictgetlen(node, tartype)
                        + dictgetlen(root, tartype)
                        + dictgetlen(nroot, tartype)
                    )

                    if hastimestamp:
                        base_ts = roottimestamp[nodetype]
                    

                        if not torch.is_tensor(base_ts):
                            base_ts = torch.as_tensor(base_ts)
                    

                        if base_ts.ndim == 0:
                            base_ts = base_ts.view(1)
                    

                        tartimestamp = base_ts[srcidx]
                    

                        if tartimestamp.ndim == 0:
                            tartimestamp = tartimestamp.view(1)
                    
                        nroottimestamp = dictupdate(nroottimestamp, tartype, tartimestamp)
                    else:
                        nroottimestamp[tartype] = None

                    nroot = dictupdate(nroot, tartype, taridx)

                    srcidx = srcidx + found_src_num
                    taridx = torch.arange(taridx.shape[0], device=taridx.device) + found_tar_num

                    adj = dictupdate(
                        adj, edgetype, torch.stack((srcidx, taridx), dim=0), concatdim=1
                    )

            for nodetype in root:
                node = dictupdate(node, nodetype, root[nodetype])

            root = nroot
            roottimestamp = nroottimestamp

        for nodetype in root:
            node = dictupdate(node, nodetype, root[nodetype])

        root_len = len(_to_hf_index(root_nodeidx)) if not torch.is_tensor(root_nodeidx) else int(root_nodeidx.numel())
        mapping = torch.arange(root_len, device=node[root_nodetype].device if torch.is_tensor(node.get(root_nodetype, None)) else torch.device("cpu"))


        for nodetype in node:
            node[nodetype] = self.nodes[nodetype].getfeat(node[nodetype], floatemb)

        edgenameemb = {edgetype: self.edgenameemb[edgetype] for edgetype in adj}
        nodenameemb = {
            nodetype: torch.stack([self.featnameemb[_] for _ in self.nodes[nodetype].featlist], dim=0)
            for nodetype in node
        }

        return node, adj, nodenameemb, edgenameemb, mapping

    def fewshot(
        self,
        root_nodetype: str,
        root_nodeidx: torch.LongTensor,
        task_mask: torch.BoolTensor,
        floatemb,
        fanout: int = INF,
        timestamp: Union[list[int], None] = None,
        prefetch_factor: int = 10,
    ):
        assert fanout < INF, "always sample past"
        idx = torch.randint(0, MAXINT64, (root_nodeidx.shape[0], prefetch_factor * fanout))
        idx = idx % (root_nodeidx.clamp_min(1)).reshape(-1, 1)

        if prefetch_factor > 1:
            rootfeat = self.nodes[root_nodetype].getfeat(root_nodeidx, floatemb)
            rootfeat[task_mask.unsqueeze(-1).expand(-1, -1, rootfeat.shape[-1])] = 0.0
            feat = self.nodes[root_nodetype].getfeat(idx.flatten(), floatemb).unflatten(
                0, (-1, prefetch_factor * fanout)
            )
            score = feat.flatten(-2, -1) @ rootfeat.flatten(-2, -1).unsqueeze(-1)
            score = score.squeeze(-1)
            idx = torch.gather(idx, 1, torch.topk(score, k=fanout, dim=-1)[1]).flatten()

        idx = idx.flatten()
        rootnode = torch.arange(root_nodeidx.shape[0]).repeat_interleave(fanout)
        mask = root_nodeidx[rootnode] > 0
        idx, rootnode = idx[mask], rootnode[mask]
        return idx, rootnode


class Task:
    def __init__(self, path) -> None:
        with open(osp.join(path, "metatask.yaml")) as f:
            self.metatask = yaml.safe_load(f)

        def _safe_load_pt(fp: str):
            try:
                return torch.load(fp, map_location="cpu", weights_only=True)
            except TypeError:
                return torch.load(fp, map_location="cpu")

        self.tasknameemb = _safe_load_pt(osp.join(path, "tasknameemb.pt"))

        self.tasks = {
            taskname: load_hf_folder(osp.join(path, "task", taskname)).with_format("torch")
            for taskname in self.metatask
        }

    def _slice_split(self, ds: hds.Dataset, meta: dict, split: str, idx_hf):
        if split == "train":
            return ds[idx_hf]
        if split == "valid":
            return ds[_to_hf_index((np.asarray(meta["split"][0]) + np.asarray(idx_hf)))]
        if split == "test":
            base = meta["split"][0] + meta["split"][1]
            return ds[_to_hf_index((np.asarray(base) + np.asarray(idx_hf)))]
        raise NotImplementedError

    def get_retrieval(
        self,
        graph: Graph,
        taskname: str,
        split: Literal["train", "valid", "test"],
        idx: Union[int, torch.Tensor, list[int], range],
    ):
        meta = self.metatask[taskname]
        assert meta["task_type"] == "retrieval"
        target_type = meta["target_type"]

        idx_hf = _to_hf_index(idx)
        taskinfo = self._slice_split(self.tasks[taskname], meta, split, idx_hf)

        nodeidx = taskinfo["nodeidx"]
        label = taskinfo["label"]

        if torch.is_tensor(label) and label.ndim > 1:
            label = label.view(-1)

        timestamp = taskinfo["timestamp"] if meta.get("hastimestamp", False) else None

        target_feat_mask = torch.tensor(
            [_ not in meta["masked_feat"] for _ in graph.metanode[target_type]["feat"]],
            dtype=torch.bool,
        )


        num_class = int(meta["num_class"])
        graph.ensure_number_node(num_needed=num_class)
        seed_type = "number"

        return (
            target_type,
            target_feat_mask,
            nodeidx,
            label,
            timestamp,
            self.tasknameemb[taskname],
            (seed_type, num_class),
        )

    def get_regression(
        self,
        graph: Graph,
        taskname: str,
        split: Literal["train", "valid", "test"],
        idx: Union[int, torch.Tensor, list[int], range],
    ):
        meta = self.metatask[taskname]
        assert meta["task_type"] == "regression"
        target_type = meta["target_type"]

        idx_hf = _to_hf_index(idx)
        taskinfo = self._slice_split(self.tasks[taskname], meta, split, idx_hf)

        nodeidx = taskinfo["nodeidx"]
        label = taskinfo["label"]
        if torch.is_tensor(label) and label.ndim > 1:
            label = label.view(-1)

        timestamp = taskinfo["timestamp"] if meta.get("hastimestamp", False) else None

        target_feat_mask = torch.tensor(
            [_ not in meta["masked_feat"] for _ in graph.metanode[target_type]["feat"]],
            dtype=torch.bool,
        )
        return (
            target_type,
            target_feat_mask,
            nodeidx,
            label,
            timestamp,
            self.tasknameemb[taskname],
            (None, None),
        )
