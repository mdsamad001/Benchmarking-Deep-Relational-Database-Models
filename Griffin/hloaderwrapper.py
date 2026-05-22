from hdataset import Graph, INF, edgename2head, edgename2tail, Task
from hFloatEmb import SimpleRepeater
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed
from typing import Literal
import accelerate
import copy

def unifyheterograph(rootnodetype, node, nodenameemb, edgenameemb, adj):
    assert rootnodetype in node
    nodetypes = [rootnodetype] + list([_ for _ in node.keys() if _ != rootnodetype])
    nodeptr = [0] + [node[nodetype].shape[0] for nodetype in nodetypes]
    nodeptr = np.cumsum(nodeptr)
    node = [(nodenameemb[nodetype], node[nodetype])  for nodetype in nodetypes]
    
    edgetypes = list(adj.keys())

    if len(edgetypes) > 0:
        edge_attr = torch.stack([edgenameemb[edgetype] for edgetype in edgetypes], dim=0)
        edge_index = []
        edge_attr_type = []
        for i, edgetype in enumerate(edgetypes):
            head, tail = edgename2head(edgetype), edgename2tail(edgetype)
            row = nodeptr[nodetypes.index(head)] + adj[edgetype][0]
            col = nodeptr[nodetypes.index(tail)] + adj[edgetype][1]
            attr = torch.zeros_like(row).fill_(i)
            edge_index.append(torch.stack((row, col), dim=0))
            edge_attr_type.append(attr)
        edge_index = torch.concat(edge_index, dim=1)
        edge_attr_type = torch.concat(edge_attr_type, dim=0)
    else:
        edge_attr, edge_index, edge_attr_type = None, None, None
    return node, edge_index, edge_attr_type, edge_attr

def mergefewshotgraph(node, edge_index, edge_attr_type, edge_attr, fewshotnode, fewshotedge_index, fewshotedge_attr_type, fewshotedge_attr, tarnode, fewshotrelemb):

    lennode = sum([_[1].shape[0] for _ in node])
    if fewshotedge_index is not None:
        fewshotedge_index += lennode
        fewshotedge_attr_type += edge_attr.shape[0]    

    if edge_index is None:
        assert fewshotedge_index is None
        edge_index = torch.stack((tarnode, lennode+torch.arange(tarnode.shape[0])), dim=0)
        edge_attr_type = torch.zeros_like(tarnode)
        edge_attr = fewshotrelemb.reshape(1, -1)
    else:
        if fewshotedge_index is not None:
            edge_index = torch.concat((edge_index, fewshotedge_index, torch.stack((tarnode, lennode+torch.arange(tarnode.shape[0])), dim=0)), dim=1)
            edge_attr_type = torch.concat((edge_attr_type, fewshotedge_attr_type, torch.zeros_like(tarnode) + (edge_attr.shape[0]+fewshotedge_attr.shape[0])), dim=0)
            edge_attr = torch.concat((edge_attr, fewshotedge_attr, fewshotrelemb.reshape(1, -1)), dim=0)
        else:
            edge_index = torch.concat((edge_index, torch.stack((tarnode, lennode+torch.arange(tarnode.shape[0])), dim=0)), dim=1)
            edge_attr_type = torch.concat((edge_attr_type, torch.zeros_like(tarnode) + edge_attr.shape[0]), dim=0)
            edge_attr = torch.concat((edge_attr, fewshotrelemb.reshape(1, -1)), dim=0)    
    node = node + fewshotnode
    return node, edge_index, edge_attr_type, edge_attr

def scalefeat(node, taskfeat, edge_attr, y=None):
    dim = node[0][0].shape[-1]
    shape = (dim,)
    for i in range(len(node)):
        node[i] = (F.layer_norm(node[i][0], shape), F.layer_norm(node[i][1], shape))
        taskfeat[i] = None if taskfeat[i] is None else F.layer_norm(taskfeat[i], shape)
    if edge_attr is not None:
        edge_attr = F.layer_norm(edge_attr, shape)
    if y is not None:
        y = F.layer_norm(y, shape)
    return node, taskfeat, edge_attr, y

def buildindice(shuffle, lens, batch_size):
    ind = []
    if not shuffle:
        for nodetypeidx, (_, num) in enumerate(lens):
            leftnum = num % batch_size
            padnum = 0 if leftnum == 0 else batch_size - leftnum
            idx = torch.arange(num+padnum)
            idx[num:] = -1
            idx = idx.reshape(-1, batch_size)
            idx = torch.concat((idx[:, [0]].clone().fill_(nodetypeidx), idx), dim=-1)
            ind.append(idx)
    else:
        for nodetypeidx, (_, num) in enumerate(lens):
            leftnum = num % batch_size
            padnum = 0 if leftnum == 0 else batch_size - leftnum
            idx = torch.arange(num+padnum)
            idx[num:] = -1
            idx = idx[torch.randperm(idx.shape[0])]
            idx = idx.reshape(-1, batch_size)
            idx = torch.concat((idx[:, [0]].clone().fill_(nodetypeidx), idx), dim=-1)
            ind.append(idx)
    return torch.cat(ind, dim=0)

def buildindice_downsample(shuffle, lens, batch_size, downsample_ratio, downsample_seed=42):
    ind = []
    if not shuffle:
        for nodetypeidx, (_, num) in enumerate(lens):
            leftnum = num % batch_size
            padnum = 0 if leftnum == 0 else batch_size - leftnum
            idx = torch.arange(num+padnum)
            idx[num:] = -1
            idx = idx.reshape(-1, batch_size)
            idx = torch.concat((idx[:, [0]].clone().fill_(nodetypeidx), idx), dim=-1)
            ind.append(idx)
    else:
        for nodetypeidx, (_, num) in enumerate(lens):
            sample_num = int(num * downsample_ratio)
            leftnum = sample_num % batch_size
            padnum = 0 if leftnum == 0 else batch_size - leftnum


            rng = np.random.RandomState(downsample_seed)
            sampled_indices = torch.from_numpy(rng.permutation(num)[:sample_num])
            idx = torch.cat([sampled_indices, torch.full((padnum,), -1)])
            idx = idx[torch.randperm(idx.shape[0])]
            idx = idx.reshape(-1, batch_size)
            idx = torch.concat((idx[:, [0]].clone().fill_(nodetypeidx), idx), dim=-1)
            ind.append(idx)
    return torch.cat(ind, dim=0)

def buildindice_downsample_absolute(shuffle, lens, batch_size, downsample_num, downsample_seed=42):
    ind = []
    if not shuffle:
        for nodetypeidx, (_, num) in enumerate(lens):
            leftnum = num % batch_size
            padnum = 0 if leftnum == 0 else batch_size - leftnum
            idx = torch.arange(num+padnum)
            idx[num:] = -1
            idx = idx.reshape(-1, batch_size)
            idx = torch.concat((idx[:, [0]].clone().fill_(nodetypeidx), idx), dim=-1)
            ind.append(idx)
    else:
        for nodetypeidx, (_, num) in enumerate(lens):


            sample_num = min(downsample_num, num)
            leftnum = sample_num % batch_size
            padnum = 0 if leftnum == 0 else batch_size - leftnum


            rng = np.random.RandomState(downsample_seed)
            sampled_indices = torch.from_numpy(rng.permutation(num)[:sample_num])
            idx = torch.cat([sampled_indices, torch.full((padnum,), -1)])
            idx = idx[torch.randperm(idx.shape[0])]
            idx = idx.reshape(-1, batch_size)
            idx = torch.concat((idx[:, [0]].clone().fill_(nodetypeidx), idx), dim=-1)
            ind.append(idx)
    return torch.cat(ind, dim=0)

def buildindice_sample(shuffle, lens, batch_size, sample_config, sample_seed=42):
    ind = []
    if not shuffle:
        for nodetypeidx, (_, num) in enumerate(lens):
            leftnum = num % batch_size
            padnum = 0 if leftnum == 0 else batch_size - leftnum
            idx = torch.arange(num+padnum)
            idx[num:] = -1
            idx = idx.reshape(-1, batch_size)
            idx = torch.concat((idx[:, [0]].clone().fill_(nodetypeidx), idx), dim=-1)
            ind.append(idx)
    else:
        for nodetypeidx, (taskname, num) in enumerate(lens):
            task_config = sample_config.get(taskname, sample_config["default"])
            if task_config is None:
                raise ValueError(f"task {taskname} not found in sample_config")
            sample_num = int(num * task_config)
            leftnum = sample_num % batch_size
            padnum = 0 if leftnum == 0 else batch_size - leftnum


            rng = np.random.RandomState(sample_seed)


            sampled_indices = torch.from_numpy(rng.randint(0, num, sample_num))
            idx = torch.cat([sampled_indices, torch.full((padnum,), -1)])
            idx = idx[torch.randperm(idx.shape[0])]
            idx = idx.reshape(-1, batch_size)
            idx = torch.concat((idx[:, [0]].clone().fill_(nodetypeidx), idx), dim=-1)
            ind.append(idx)
    return torch.cat(ind, dim=0)


class LoaderWrapper:

    def __init__(self, graph: Graph, batch_size: int, shuffle: bool, subgraphargs: dict[str], fewshotfanout: int) -> None:
        self.prefetch_factor = 1
        self.fewshotfanout = fewshotfanout
        self.graph = graph
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subgraphargs = subgraphargs
        self.lens = [(nodetype, self.graph.metanode[nodetype]["num"]) for nodetype in self.graph.metanode]
        self.ind = None
    
    def rebuild_indice(self, accelerator: accelerate.Accelerator):
        if accelerator.is_main_process:
            self.ind = buildindice(self.shuffle, self.lens, self.batch_size)
        self.ind = accelerate.utils.broadcast_object_list([self.ind], from_process=0)[0]
    
    def rebuild_indice_downsample(self, accelerator: accelerate.Accelerator, downsample_ratio: float, downsample_seed: int):
        if accelerator.is_main_process:
            self.ind = buildindice_downsample(self.shuffle, self.lens, self.batch_size, downsample_ratio, downsample_seed)
        self.ind = accelerate.utils.broadcast_object_list([self.ind], from_process=0)[0]
    
    def rebuild_indice_downsample_absolute(self, accelerator: accelerate.Accelerator, downsample_num: int, downsample_seed: int):
        if accelerator.is_main_process:
            self.ind = buildindice_downsample_absolute(self.shuffle, self.lens, self.batch_size, downsample_num, downsample_seed)
        self.ind = accelerate.utils.broadcast_object_list([self.ind], from_process=0)[0]
    
    def rebuild_indice_sample(self, accelerator: accelerate.Accelerator, sample_config, sample_seed: int):
        if accelerator.is_main_process:
            self.ind = buildindice_sample(self.shuffle, self.lens, self.batch_size, sample_config, sample_seed)
        self.ind = accelerate.utils.broadcast_object_list([self.ind], from_process=0)[0]
    
    def __len__(self):
        assert self.ind is not None, "please rebuild_indice first"
        return self.ind.shape[0]

    def subgraph(self, nodetype, ind, timestamp=None):
        return self.graph.subgraph(nodetype, ind, **self.subgraphargs, timestamp=timestamp)

    def fewshotsubgraph(self, nodetype, ind, timestamp=None):
        tmpargs = copy.copy(self.subgraphargs)
        tmpargs["hop"] = 0
        return self.graph.subgraph(nodetype, ind, **tmpargs, timestamp=timestamp)

    def fewshotroot(self, rootnodetype, tind, taskmask, roottimestamp=None):
        return self.graph.fewshot(rootnodetype, tind, taskmask, self.subgraphargs["floatemb"], self.fewshotfanout, roottimestamp, prefetch_factor=self.prefetch_factor)

    def __getitem__(self, idx:int):        
        tind = self.ind[idx]
        nodetype = self.lens[tind[0].item()][0]
        tind = tind[1:]
        tind = tind[tind>=0]

        node, adj, nodenameemb, edgenameemb, mapping = self.subgraph(nodetype, tind)
        node, edge_index, edge_attr_type, edge_attr = unifyheterograph(nodetype, node, nodenameemb, edgenameemb, adj)

        yidx = [torch.randint(0, node[i][1].shape[1], (node[i][1].shape[0],)) for i in range(len(node))]
        taskfeat = [node[i][0][yidx[i]] for i in range(len(node))]
        y = [node[i][1][torch.arange(node[i][1].shape[0]), yidx[i]] for i in range(len(node))]
        mask = [F.one_hot(yidx[i], num_classes=node[i][1].shape[1]).to(torch.bool) for i in range(len(node))]
        
        y = torch.concat(y, dim=0)

        node, taskfeat, edge_attr, y = scalefeat(node, taskfeat, edge_attr, y)
        
        return  node, mask, taskfeat, edge_index, edge_attr_type, edge_attr, y


class LoaderWrapperCompletion(LoaderWrapper):

    def __init__(self, graph: Graph, batch_size: int, shuffle: bool, subgraphargs: dict[str], fewshotfanout: int) -> None:
        super().__init__(graph, batch_size, shuffle, subgraphargs, fewshotfanout)
        self.lens = [(nodetype, self.graph.metanode[nodetype]["num"]) for nodetype in self.graph.metanode if len(self.graph.metanode[nodetype]["feat"])>1]

    def __getitem__(self, idx:int):
        tind = self.ind[idx]
        rootnodetype = self.lens[tind[0].item()][0]
        tind = tind[1:]
        tind = tind[tind>=0]

        roottimestamp = self.graph.nodes[rootnodetype].feat[tind]["timestamp"]

        node, adj, nodenameemb, edgenameemb, mapping = self.subgraph(rootnodetype, tind, roottimestamp)

        node, edge_index, edge_attr_type, edge_attr = unifyheterograph(rootnodetype, node, nodenameemb, edgenameemb, adj)

        yidx = torch.randint(0, node[0][1].shape[1], (node[0][1].shape[0],))
        taskfeat = [node[0][0][yidx]]+ [None for i in range(len(node)-1)]
        y = node[0][1][torch.arange(node[0][1].shape[0]), yidx]
        mask = [F.one_hot(yidx, num_classes=node[0][1].shape[1]).to(torch.bool)] + [None for i in range(len(node)-1)]

        if self.fewshotfanout > 0:
            fewshot_center, tarnode = self.fewshotroot(rootnodetype, tind, mask[0], roottimestamp)

            fewshotnode, fewshotadj, fewshotnodenameemb, fewshotedgenameemb, fewshotmapping = self.fewshotsubgraph(rootnodetype, fewshot_center, None if roottimestamp is None else roottimestamp[tarnode])
            fewshotnode, fewshotedge_index, fewshotedge_attr_type, fewshotedge_attr = unifyheterograph(rootnodetype, fewshotnode, fewshotnodenameemb, fewshotedgenameemb, fewshotadj)

            node, edge_index, edge_attr_type, edge_attr = mergefewshotgraph(node, edge_index, edge_attr_type, edge_attr, fewshotnode, fewshotedge_index, fewshotedge_attr_type, fewshotedge_attr, tarnode, self.graph.edgenameemb["fewshot"])
            
            taskfeat = taskfeat + [taskfeat[0][tarnode]] + [None for i in range(len(fewshotnode)-1)]

            mask = mask + [None] * len(fewshotnode)

        node, taskfeat, edge_attr, y = scalefeat(node, taskfeat, edge_attr, y)
        
        return  node, mask, taskfeat, edge_index, edge_attr_type, edge_attr, y


class LoaderWrapperRetrieval(LoaderWrapper):

    def __init__(self, graph: Graph, batch_size: int, shuffle: bool, subgraphargs: dict[str], task: Task, tasknames: list[str], split: Literal["train", "valid", "test"], fewshotfanout: int) -> None:
        super().__init__(graph, batch_size, shuffle, subgraphargs, fewshotfanout)
        assert len(tasknames) > 0, "please provide task name"
        self.task = task
        self.lens = [(taskname, self.task.metatask[taskname]["split"][{"train":0, "valid": 1, "test": 2}[split]]) for i, taskname in enumerate(tasknames)]
        self.split = split

    def __getitem__(self, idx:int):
        tind = self.ind[idx]
        taskname = self.lens[tind[0].item()][0]
        tind = tind[1:]
        tind = tind[tind>=0]

        nodetype, target_feat_mask, tind, label, tasktimestamp, tasknameemb, (seed_type, num_class) = self.task.get_retrieval(self.graph, taskname, self.split, tind)

        y = self.graph.nodes[seed_type].getfeat(range(num_class), self.subgraphargs["floatemb"])
        assert y.shape[1] == 1
        y = y.squeeze_(1)


        node, adj, nodenameemb, edgenameemb, mapping = self.subgraph(nodetype, tind, tasktimestamp)
        node[nodetype] = node[nodetype][:, target_feat_mask]
        nodenameemb[nodetype] = nodenameemb[nodetype][target_feat_mask]


        node, edge_index, edge_attr_type, edge_attr = unifyheterograph(nodetype, node, nodenameemb, edgenameemb, adj)


        taskfeat = [tasknameemb for i in range(len(node))]
        mask = [None for i in range(len(node))]
        
        node, taskfeat, edge_attr = scalefeat(node, taskfeat, edge_attr)
        
        return  node, mask, taskfeat, edge_index, edge_attr_type, edge_attr, label, y, mapping


class LoaderWrapperRegression(LoaderWrapper):

    def __init__(self, graph: Graph, batch_size: int, shuffle: bool, subgraphargs: dict[str], task: Task, tasknames: list[str], split: Literal["train", "valid", "test"], fewshotfanout: int) -> None:
        super().__init__(graph, batch_size, shuffle, subgraphargs, fewshotfanout)
        assert len(tasknames) > 0, "please provide task name"
        self.task = task
        self.lens = [(taskname, self.task.metatask[taskname]["split"][{"train":0, "valid": 1, "test": 2}[split]]) for i, taskname in enumerate(tasknames)]
        self.split = split

    def __getitem__(self, idx:int):
        tind = self.ind[idx]
        taskname = self.lens[tind[0].item()][0]
        tind = tind[1:]
        tind = tind[tind>=0]

        nodetype, target_feat_mask, tind, label, tasktimestamp, tasknameemb, _ = self.task.get_regression(self.graph, taskname, self.split, tind)


        node, adj, nodenameemb, edgenameemb, mapping = self.subgraph(nodetype, tind, tasktimestamp)
        node[nodetype] = node[nodetype][:, target_feat_mask]
        nodenameemb[nodetype] = nodenameemb[nodetype][target_feat_mask]


        node, edge_index, edge_attr_type, edge_attr = unifyheterograph(nodetype, node, nodenameemb, edgenameemb, adj)


        taskfeat = [tasknameemb for i in range(len(node))]
        mask = [None for i in range(len(node))]
        
        node, taskfeat, edge_attr = scalefeat(node, taskfeat, edge_attr)
        
        return  node, mask, taskfeat, edge_index, edge_attr_type, edge_attr, label, mapping


class LoaderWrapperTask(LoaderWrapper):

    def __init__(self, graph: Graph, batch_size: int, shuffle: bool, subgraphargs: dict[str], task: Task, tasknames: list[str], split: Literal["train", "valid", "test"], fewshotfanout: int) -> None:
        super().__init__(graph, batch_size, shuffle, subgraphargs, fewshotfanout)
        assert len(tasknames) > 0, "please provide task name"
        self.task = task
        self.lens = [(taskname, self.task.metatask[taskname]["split"][{"train":0, "valid": 1, "test": 2}[split]]) for i, taskname in enumerate(tasknames)]
        self.split = split

    def __getitem__(self, idx:int):
        tind = self.ind[idx]
        taskname = self.lens[tind[0].item()][0]
        tind = tind[1:]
        tind = tind[tind>=0]

        is_regression = self.task.metatask[taskname]["task_type"] == "regression"

        if is_regression:
            rootnodetype, target_feat_mask, tind, label, tasktimestamp, tasknameemb, _ = self.task.get_regression(self.graph, taskname, self.split, tind)
            y = None
        else:
            rootnodetype, target_feat_mask, tind, label, tasktimestamp, tasknameemb, (seed_type, num_class) = self.task.get_retrieval(self.graph, taskname, self.split, tind)
            y = self.graph.nodes[seed_type].getfeat(range(num_class), self.subgraphargs["floatemb"])
            assert y.shape[1] == 1
            y = y.squeeze_(1)
        
        node, adj, nodenameemb, edgenameemb, mapping = self.subgraph(rootnodetype, tind, tasktimestamp)


        node, edge_index, edge_attr_type, edge_attr = unifyheterograph(rootnodetype, node, nodenameemb, edgenameemb, adj)

        taskfeat = [tasknameemb for i in range(len(node))]
        mask = [None for i in range(len(node))]
        mask[0] = torch.zeros(node[0][1].shape[:2], dtype=torch.bool)
        mask[0][mapping] = torch.logical_not(target_feat_mask)

        
        if self.fewshotfanout > 0:
            fewshot_center, tarnode = self.fewshotroot(rootnodetype, tind, mask[0][mapping], tasktimestamp)

            fewshotnode, fewshotadj, fewshotnodenameemb, fewshotedgenameemb, fewshotmapping = self.fewshotsubgraph(rootnodetype, fewshot_center, None if tasktimestamp is None else tasktimestamp[tarnode])
            fewshotnode, fewshotedge_index, fewshotedge_attr_type, fewshotedge_attr = unifyheterograph(rootnodetype, fewshotnode, fewshotnodenameemb, fewshotedgenameemb, fewshotadj)

            node, edge_index, edge_attr_type, edge_attr = mergefewshotgraph(node, edge_index, edge_attr_type, edge_attr, fewshotnode, fewshotedge_index, fewshotedge_attr_type, fewshotedge_attr, tarnode, self.graph.edgenameemb["fewshot"])
            
            taskfeat = taskfeat + [tasknameemb for i in range(len(fewshotnode))]
            mask = mask + [None] * len(fewshotnode)

        node, taskfeat, edge_attr, y = scalefeat(node, taskfeat, edge_attr, y)
        
        return  node, mask, taskfeat, edge_index, edge_attr_type, edge_attr, label, y, mapping
