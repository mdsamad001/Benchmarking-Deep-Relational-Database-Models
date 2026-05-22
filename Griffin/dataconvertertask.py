import yaml
import numpy as np
import os.path as osp
import torch
import yaml
from datasets import Dataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("srcpath", type=str)
parser.add_argument("dstpath", type=str)
parser.add_argument("--ncpu", type=int, default=16)
args = parser.parse_args()

srcpath = args.srcpath
dstpath = args.dstpath

with open(osp.join(srcpath, "metadata.yaml"), "r") as f:
    config: dict = yaml.safe_load(f)

def findtimestamp(datalist):
    for data in datalist:
        if data["name"] == "timestamp":
            return torch.from_numpy(np.load(osp.join(srcpath, data["path"])))
    return None

def findnodepairs(datalist):
    for data in datalist:
        if data["name"] == "node_pairs":
            return torch.from_numpy(np.load(osp.join(srcpath, data["path"])))
    print(data)
    raise NotImplementedError

def findseednodes(datalist):
    for data in datalist:
        if data["name"] == "seed_nodes":
            return torch.from_numpy(np.load(osp.join(srcpath, data["path"])))
    print(data)
    raise NotImplementedError

def findlabels(datalist):
    for data in datalist:
        if data["name"] == "labels":
            return torch.from_numpy(np.load(osp.join(srcpath, data["path"])))
    print(data)
    raise NotImplementedError

def findfeatname(datalist, featname):
    for data in datalist:
        if data["name"] == featname:
            return osp.join(srcpath, data["path"])
    print(data)
    raise NotImplementedError


def parsetask(task: dict):
    extrafield = task["extra_fields"]
    name, target_type, task_type, metric, task_emb, seed_type = extrafield["name"], extrafield["target_type"], extrafield["task_type"], extrafield["evaluation_metric"], torch.tensor(extrafield["task_emb"]), extrafield["seed_type"].split(":")[-1]
    assert task_type in ["retrieval", "regression"]


    if task_type == "regression":
        target_type, seed_type = seed_type, target_type

    extra_feat_path = [{}, {}, {}]
    if task_type == "retrieval":


        assert seed_type.startswith(target_type)
        target_column = seed_type[len(target_type) + 1:]
        target_column_feat = f"{target_type}___Griffin_text_{target_column}"
        if target_column_feat in nodemeta[target_type]["feat"]:

            allowed_feat = [
                feat for feat in nodemeta[target_type]["feat"] if feat != target_column_feat
            ]
            extra_feat = []
        else:

            allowed_feat = nodemeta[target_type]["feat"]
            extra_feat = []
    elif task_type == "regression":
        if target_type == seed_type:
            assert seed_type.startswith(target_type)


            target_column = name.split("-")[-1]
            target_column_feat = f"{target_type}___{target_column}"
            if target_column_feat in nodemeta[target_type]["feat"]:
                allowed_feat = [
                    feat for feat in nodemeta[target_type]["feat"] if feat != target_column_feat
                ]
                extra_feat = []
            else:
                allowed_feat = nodemeta[target_type]["feat"]
                extra_feat = []
        else:
            return None, None, None
            allowed_feat = None
            extra_feat = [_["name"] for _ in extrafield["seed_feature_schema"]]

    if task_type == "retrieval":
        nodepair_trn = findnodepairs(task["train_set"][0]["data"])
        nodeidx_trn, label_trn = nodepair_trn[:, 0], nodepair_trn[:, 1]
        timestamp_trn = findtimestamp(task["train_set"][0]["data"])
    elif task_type == "regression":
        label_trn = findlabels(task["train_set"][0]["data"])
        nodeidx_trn = findseednodes(task["train_set"][0]["data"])
        timestamp_trn = findtimestamp(task["train_set"][0]["data"])
        if len(extra_feat) > 0:
            extra_feat_path[0] = {featname: findfeatname(task["train_set"][0]["data"], featname) for featname in extra_feat}

    
    if task_type == "retrieval":
        nodepair_val = findnodepairs(task["validation_set"][0]["data"])
        label_val = findlabels(task["validation_set"][0]["data"])
        timestamp_val = findtimestamp(task["validation_set"][0]["data"])

        if timestamp_val is not None:
            timestamp_val = timestamp_val[label_val==1]
        nodepair_val = nodepair_val[label_val==1]
        nodeidx_val, label_val = nodepair_val[:, 0], nodepair_val[:, 1]

    elif task_type == "regression":
        label_val = findlabels(task["validation_set"][0]["data"])
        nodeidx_val = findseednodes(task["validation_set"][0]["data"])
        timestamp_val = findtimestamp(task["validation_set"][0]["data"])
        if len(extra_feat) > 0:
            extra_feat_path[1] = {featname: findfeatname(task["validation_set"][0]["data"], featname) for featname in extra_feat}


    if task_type == "retrieval":
        nodepair_tst = findnodepairs(task["test_set"][0]["data"])
        label_tst = findlabels(task["test_set"][0]["data"])
        timestamp_tst = findtimestamp(task["test_set"][0]["data"])
        
        if timestamp_tst is not None:
            timestamp_tst = timestamp_tst[label_tst==1]
        nodepair_tst = nodepair_tst[label_tst==1]
        nodeidx_tst, label_tst = nodepair_tst[:, 0], nodepair_tst[:, 1]

    elif task_type == "regression":
        label_tst = findlabels(task["test_set"][0]["data"])
        nodeidx_tst = findseednodes(task["test_set"][0]["data"])
        timestamp_tst = findtimestamp(task["test_set"][0]["data"])
        if len(extra_feat) > 0:
            extra_feat_path[2] = {featname: findfeatname(task["test_set"][0]["data"], featname) for featname in extra_feat}

    
    num_trn, num_val, num_tst = nodeidx_trn.shape[0], nodeidx_val.shape[0], nodeidx_tst.shape[0]
    assert nodeidx_trn.ndim == 1
    assert nodeidx_val.ndim == 1
    assert nodeidx_tst.ndim == 1

    assert label_trn.shape[0] == num_trn
    assert label_val.shape[0] == num_val
    assert label_tst.shape[0] == num_tst
    if timestamp_trn is not None:
        assert timestamp_trn.shape[0] == num_trn
        assert timestamp_val.shape[0] == num_val
        assert timestamp_tst.shape[0] == num_tst
    
    if timestamp_trn is not None:
        ds = Dataset.from_dict({
            "nodeidx": torch.concat((nodeidx_trn, nodeidx_val, nodeidx_tst), dim=0),
            "label": torch.concat((label_trn, label_val, label_tst), dim=0),
            "timestamp": torch.concat((timestamp_trn, timestamp_val, timestamp_tst), dim=0)})
    
    else:
        ds = Dataset.from_dict({
            "nodeidx": torch.concat((nodeidx_trn, nodeidx_val, nodeidx_tst), dim=0),
            "label": torch.concat((label_trn, label_val, label_tst), dim=0)})
    
    ds.save_to_disk(osp.join(dstpath, f"task/{name}"))
    

    if task_type == "retrieval":
        num_class = extrafield["num_classes"]
    else:
        num_class = 1

    meta = {
        "name": name, 
        "task_type": task_type, 
        "num_class": num_class,
        "seed_type": seed_type,
        "target_type": target_type,
        "split": [num_trn, num_val, num_tst],
        "metric": metric,
        "hastimestamp": timestamp_trn is not None,
        "allowed_feat": allowed_feat,
        "extra_feat": extra_feat
        }
    return name, meta, task_emb


task_embs = {}
metas = {}
with open(osp.join(dstpath, "metanode.yaml"), "r") as f:
    nodemeta = yaml.safe_load(f)

for i, task in enumerate(config["tasks"]):
    print(i)
    name, meta, task_emb = parsetask(task)
    print(name)
    if name is None:
        continue
    
    task_type = meta["task_type"]
    seed_type = meta["seed_type"]
    target_type = meta["target_type"]
    if task_type == "retrieval":
        assert nodemeta[seed_type]["is_target"], "for retrieval, we suppose seed is manual target node"
        assert len(nodemeta[seed_type]["feat"]) == 1
    
    allowed_feat = meta.pop("allowed_feat")
    masked_feat = [_ for _ in nodemeta[target_type]["feat"] if _ not in allowed_feat]
    meta["masked_feat"] = masked_feat
        
    metas[name] = meta
    task_embs[name] = task_emb

torch.save(task_embs, osp.join(dstpath, "tasknameemb.pt"))
with open(osp.join(dstpath, "metatask.yaml"), "w") as f:
    yaml.dump(metas, f)

exit()
print(config["tasks"][1]["extra_fields"])
{'evaluation_metric': 'hr@1', 
 'key_prediction_label_column': 'label', 
 'key_prediction_query_idx_column': 'query_idx', 
 'name': 'outbrain-small-Event-platform', 
 'num_classes': 3, 
 'num_seeds': 2, 
 'seed_feature_schema': [
     {'dtype': 'float', 'in_size': 1, 'is_target_column': False, 'name': 'TIMESTAMP(timestamp)', 'name_emb': floatlist}, {'category_mapping': dict[str, str], 'dtype': 'category', 'is_target_column': False, 'name': 'geo_location', 'name_emb': floatlist, 'num_categories': 974}, 
     {'dtype': 'float', 'is_target_column': False, 'name': 'Griffin_text_platform', 'name_emb': floatlist}, 
     {'dtype': 'float', 'is_target_column': False, 'name': 'Griffin_text_geo_location', 'name_emb': floatlist}], 
  'seed_type': 'outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform', 'target_seed_idx': 1, 'target_type': 'outbrain-small-Event', 'task_emb': floatlist, 'task_type': 'retrieval'}

print(config["tasks"][1]["train_set"])
[{'data': [
    {'format': 'numpy', 'in_memory': True, 'name': 'node_pairs', 'path': 'outbrain-small-Event-platform/train_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_node_pairs.npy'}, 
    {'format': 'numpy', 'in_memory': True, 'name': 'TIMESTAMP(timestamp)', 'path': 'outbrain-small-Event-platform/train_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_TIMESTAMP(timestamp).npy'}, 
    {'format': 'numpy', 'in_memory': True, 'name': 'geo_location', 'path': 'outbrain-small-Event-platform/train_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_geo_location.npy'}, 
    {'format': 'numpy', 'in_memory': True, 'name': 'Griffin_text_platform', 'path': 'outbrain-small-Event-platform/train_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_Griffin_text_platform.npy'}, 
    {'format': 'numpy', 'in_memory': True, 'name': 'Griffin_text_geo_location', 'path': 'outbrain-small-Event-platform/train_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_Griffin_text_geo_location.npy'}], 
    'type': 'outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform'}]

print(config["tasks"][1]["validation_set"])
[{'data': [
    {'format': 'numpy', 'in_memory': True, 'name': 'node_pairs', 'path': 'outbrain-small-Event-platform/validation_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_node_pairs.npy'}, 
    {'format': 'numpy', 'in_memory': True, 'name': 'labels', 'path': 'outbrain-small-Event-platform/validation_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_labels.npy'}, 
    {'format': 'numpy', 'in_memory': True, 'name': 'query_idx', 'path': 'outbrain-small-Event-platform/validation_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_query_idx.npy'}, 
    {'format': 'numpy', 'in_memory': True, 'name': 'TIMESTAMP(timestamp)', 'path': 'outbrain-small-Event-platform/validation_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_TIMESTAMP(timestamp).npy'}, 
    {'format': 'numpy', 'in_memory': True, 'name': 'geo_location', 'path': 'outbrain-small-Event-platform/validation_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_geo_location.npy'}, 
    {'format': 'numpy', 'in_memory': True, 'name': 'Griffin_text_platform', 'path': 'outbrain-small-Event-platform/validation_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_Griffin_text_platform.npy'}, 
    {'format': 'numpy', 'in_memory': True, 'name': 'Griffin_text_geo_location', 'path': 'outbrain-small-Event-platform/validation_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_Griffin_text_geo_location.npy'}], 
    'type': 'outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform'}]

print(config["tasks"][1]["test_set"])
[{'data': [
    {'format': 'numpy', 'in_memory': True, 'name': 'node_pairs', 'path': 'outbrain-small-Event-platform/test_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_node_pairs.npy'}, 
    {'format': 'numpy', 'in_memory': True, 'name': 'labels', 'path': 'outbrain-small-Event-platform/test_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_labels.npy'}, 
    {'format': 'numpy', 'in_memory': True, 'name': 'query_idx', 'path': 'outbrain-small-Event-platform/test_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_query_idx.npy'}, 
    {'format': 'numpy', 'in_memory': True, 'name': 'TIMESTAMP(timestamp)', 'path': 'outbrain-small-Event-platform/test_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_TIMESTAMP(timestamp).npy'}, 
    {'format': 'numpy', 'in_memory': True, 'name': 'geo_location', 'path': 'outbrain-small-Event-platform/test_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_geo_location.npy'}, 
    {'format': 'numpy', 'in_memory': True, 'name': 'Griffin_text_platform', 'path': 'outbrain-small-Event-platform/test_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_Griffin_text_platform.npy'}, 
    {'format': 'numpy', 'in_memory': True, 'name': 'Griffin_text_geo_location', 'path': 'outbrain-small-Event-platform/test_set/outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform_Griffin_text_geo_location.npy'}], 
    'type': 'outbrain-small-Event:outbrain-small-Event-outbrain-small-Event_platform:outbrain-small-Event_platform'}]