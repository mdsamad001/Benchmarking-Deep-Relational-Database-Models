from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch


@dataclass
class HopConfig:
    max_hop:            int
    fanout_per_hop:     Optional[List[int]]
    neighbor_pad_value: int = -1


def _as_list_int(x: Any) -> List[int]:
    return [int(v) for v in np.asarray(x).tolist()]


def _unique_preserve_order(xs: List[int]) -> List[int]:
    seen: Set[int] = set()
    out:  List[int] = []
    for v in xs:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


@torch.no_grad()
def apply_hop_control(batch: dict, hop_cfg: HopConfig) -> None:
    max_hop = int(hop_cfg.max_hop)
    fanout  = hop_cfg.fanout_per_hop
    pad_val = int(hop_cfg.neighbor_pad_value)

    if max_hop < 0 and fanout is None:
        return

    node_ids = batch["node_idxs"]
    is_task  = batch["is_task_nodes"]
    f2p_ids  = batch["f2p_nbr_idxs"]

    if node_ids.dim() != 2 or is_task.dim() != 2 or f2p_ids.dim() != 3:
        raise ValueError("Expected node_idxs/is_task_nodes [B,S], f2p_nbr_idxs [B,S,K].")

    if max_hop == 0:
        f2p_ids.fill_(pad_val)
        batch["f2p_nbr_idxs"] = f2p_ids
        return

    device = f2p_ids.device
    B, S   = node_ids.shape
    _, _, K = f2p_ids.shape

    if (f2p_ids != pad_val).sum().item() == 0:
        return

    sorted_ids, perm = torch.sort(node_ids, dim=1)
    valid_id  = (f2p_ids != pad_val)
    dummy     = sorted_ids[:, :1].view(B, 1, 1).expand(B, S, K)
    nbr_safe  = torch.where(valid_id, f2p_ids, dummy)
    idx       = torch.searchsorted(sorted_ids, nbr_safe).clamp(0, S - 1)
    gathered  = sorted_ids.gather(1, idx.view(B, -1)).view(B, S, K)
    match     = valid_id & (gathered == nbr_safe)
    vpos      = perm.gather(1, idx.view(B, -1)).view(B, S, K)
    vpos      = torch.where(match, vpos, torch.full_like(vpos, pad_val))
    valid     = (vpos != pad_val) & (vpos >= 0) & (vpos < S)
    vpos_safe = vpos.clamp(0, S - 1).long()

    seed      = is_task.bool()
    has_seed  = seed.any(dim=1, keepdim=True)
    if not has_seed.all():
        f2p_ids[~has_seed.view(B, 1, 1).expand(B, S, K)] = pad_val

    INF  = 10 ** 9
    dist = torch.full((B, S), INF, device=device, dtype=torch.int32)
    dist[seed] = 0

    batch_offsets = (torch.arange(B, device=device).view(B, 1, 1) * S).long()
    flat_dst      = (vpos_safe + batch_offsets).reshape(-1)

    for h in range(1, max_hop + 1):
        frontier = dist.eq(h - 1)
        if not frontier.any():
            break
        src_frontier = frontier.view(B, S, 1).expand(B, S, K) & valid
        m = src_frontier.reshape(-1)
        if not m.any():
            break
        idx_flat  = flat_dst[m]
        nxt_flat  = torch.zeros((B * S,), device=device, dtype=torch.int8)
        nxt_flat.scatter_reduce_(0, idx_flat, torch.ones_like(idx_flat, dtype=torch.int8), reduce="amax", include_self=False)
        new = nxt_flat.view(B, S).bool() & dist.eq(INF)
        if new.any():
            dist[new] = h

    dv   = dist.gather(1, vpos_safe.view(B, -1)).view(B, S, K)
    keep = valid & dv.le(max_hop) & dv.ne(INF)

    if fanout is not None:
        fanout_t = torch.tensor([int(x) for x in fanout], device=device, dtype=torch.int64)
        L        = int(fanout_t.numel())
        du       = dist.to(torch.int64).clamp(0, max(L - 1, 0))
        budget   = fanout_t[du]
        if budget.min().item() < K:
            rand      = torch.rand((B, S, K), device=device)
            rand      = torch.where(keep, rand, torch.full_like(rand, -1.0))
            order     = torch.argsort(rand, dim=2, descending=True)
            inv_order = torch.empty_like(order)
            inv_order.scatter_(2, order, torch.arange(K, device=device).view(1, 1, K).expand(B, S, K))
            keep = keep & inv_order.lt(budget.view(B, S, 1))

    out = torch.full_like(f2p_ids, pad_val)
    out[keep] = f2p_ids[keep]
    batch["f2p_nbr_idxs"] = out


@torch.no_grad()
def apply_hop_control_from_p2f(
    batch: dict,
    hop_cfg: HopConfig,
    *,
    sampler: Any,
    dataset_idx: int = 0,
    max_k: int = 64,
) -> None:
    max_hop = int(hop_cfg.max_hop)
    if max_hop < 0:
        return

    pad_val = int(hop_cfg.neighbor_pad_value)
    fanout  = hop_cfg.fanout_per_hop

    if max_hop == 0:
        non_seed = ~batch["is_task_nodes"].bool()
        batch["is_padding"][non_seed]  = True
        batch["masks"][non_seed]       = False
        batch["is_targets"][non_seed]  = False
        if "f2p_nbr_idxs" in batch:
            batch["f2p_nbr_idxs"].fill_(pad_val)
        return

    node_ids = batch["node_idxs"]
    is_seed  = batch["is_task_nodes"].bool()
    f2p_ids  = batch.get("f2p_nbr_idxs", None)

    B, S   = node_ids.shape
    device = node_ids.device

    keep_token    = torch.zeros((B, S), dtype=torch.bool, device=device)
    node_ids_cpu  = node_ids.detach().cpu().numpy()
    is_seed_cpu   = is_seed.detach().cpu().numpy()
    f2p_cpu       = f2p_ids.detach().cpu().numpy() if f2p_ids is not None else None

    for b in range(B):
        nodes_b     = node_ids_cpu[b]
        present     = [int(x) for x in nodes_b.tolist() if x != -1]
        if not present:
            continue
        present_set = set(present)

        nid_to_pos: Dict[int, List[int]] = defaultdict(list)
        for i in range(S):
            nid = int(nodes_b[i])
            if nid != -1:
                nid_to_pos[nid].append(i)

        seed_nodes = {int(nodes_b[i]) for i in range(S) if is_seed_cpu[b, i] and int(nodes_b[i]) != -1}
        if not seed_nodes:
            continue

        dist: Dict[int, int] = {s: 0 for s in seed_nodes}
        dq = deque(seed_nodes)

        while dq:
            u  = dq.popleft()
            du = dist[u]
            if du >= max_hop:
                continue

            neighs: List[int] = []

            if f2p_cpu is not None:
                for i in nid_to_pos.get(int(u), []):
                    for v in f2p_cpu[b, i].tolist():
                        v = int(v)
                        if v != pad_val:
                            neighs.append(v)

            nbrs = sampler.p2f_neighbors_py(int(dataset_idx), int(u), int(max_k))
            neighs.extend(_as_list_int(nbrs))
            neighs = _unique_preserve_order(neighs)

            candidates = [v for v in neighs if v != -1 and v in present_set and v not in dist]

            if fanout is not None and du < len(fanout):
                budget = int(fanout[du])
                if len(candidates) > budget:
                    rng        = np.random.RandomState(hash((b, int(u))) & 0xFFFFFFFF)
                    chosen_idx = rng.choice(len(candidates), size=budget, replace=False)
                    candidates = [candidates[i] for i in chosen_idx]

            for v in candidates:
                dist[v] = du + 1
                dq.append(v)

        keep_nodes   = set(dist.keys())
        keep_mask_b  = np.array([int(nodes_b[i]) in keep_nodes for i in range(S)], dtype=np.bool_)
        keep_token[b] = torch.from_numpy(keep_mask_b).to(device)

    drop = ~keep_token
    batch["is_padding"][drop]  = True
    batch["masks"][drop]       = False
    batch["is_targets"][drop]  = False

    if "f2p_nbr_idxs" in batch:
        f2p = batch["f2p_nbr_idxs"]
        if f2p.dim() == 3:
            f2p[drop.unsqueeze(-1).expand_as(f2p)] = pad_val
            batch["f2p_nbr_idxs"] = f2p
