import yaml
import numpy as np
import os.path as osp
import torch
import copy
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("srcpath", type=str)
parser.add_argument("dstpath", type=str)
parser.add_argument("--ncpu", type=int, default=16)
args = parser.parse_args()

srcpath = args.srcpath
dstpath = args.dstpath

os.makedirs(dstpath, exist_ok=True)

with open(osp.join(srcpath, "metadata.yaml"), "r") as f:
    config: dict = yaml.safe_load(f)

def buildnodemeta(config):
    nodelist = {}
    featlist = {}
    for _ in config["graph"]["nodes"]:
        nodenum, nodetype = _["num"], _["type"]
        assert nodetype not in nodelist
        nodelist[nodetype] = {"num": nodenum, "feat": []}
    for _ in config["graph"]["feature_data"]:
        assert _["domain"] == "node"
        assert _['name'] == '__timestamp__'
        nodetype, path  = _["type"], _["path"]
        assert nodetype in nodelist
        nodelist[nodetype]["timestamp"] = path
    for nodetype in nodelist:
        assert "timestamp" in nodelist[nodetype]
    for _ in config["feature_data"]:
        assert _["domain"] == "node"
        nodetype, path, extrafield, name = _["type"], _["path"], _["extra_fields"], _["name"]
        assert nodetype in nodelist
        nameemb = np.array(extrafield.pop("name_emb"))

        is_target_column = extrafield.get("is_target_column", False)
        dtype = extrafield.get("dtype")
        num_categories = extrafield.get("num_categories", 0)
        name = f"{nodetype}___{name}"
        assert name not in featlist
        featlist[name] = (path, (is_target_column, dtype, num_categories), nameemb)
        nodelist[nodetype]['feat'].append(name)
    for nodetype in nodelist:
        featnamelist = copy.deepcopy(nodelist[nodetype]['feat'])
        for feat in featnamelist:
            if "Griffin_text_" in feat:
                origin_name = feat.replace("Griffin_text_", "")
                if origin_name in nodelist[nodetype]['feat']:
                    nodelist[nodetype]['feat'].remove(origin_name)
                    featlist.pop(origin_name)
    for nodetype in nodelist:
        featnamelist = copy.deepcopy(nodelist[nodetype]['feat'])
        nodelist[nodetype]["is_target"] = False
        for feat in featnamelist:
            if featlist[feat][1][0]:
                assert len(featnamelist) == 1
                nodelist[nodetype]["is_target"] = True
                print("target node", nodetype)
    return nodelist, featlist

nodelist, featlist = buildnodemeta(config)
with open(osp.join(dstpath, "metanode.yaml"), "w") as f:
    tnodelist = copy.deepcopy(nodelist)
    for nodetype in tnodelist:
        tnodelist[nodetype].pop("timestamp")

    yaml.dump(tnodelist, f)


torch.save({featname: torch.from_numpy(featlist[featname][-1]).to(torch.float32) for featname in featlist}, osp.join(dstpath, "featnameemb.pt"))

from datasets import Dataset

def process(nodetype):
    featembs = {}
    timestamp = np.load(osp.join(srcpath, nodelist[nodetype]["timestamp"]))
    if nodelist[nodetype]["is_target"]:
        assert np.all(timestamp==0)
        timestamp[:] = -9223372036854775808
    elif np.all(timestamp==0):
        timestamp[:] = -9223372036854775808
    nodefeatdict = {"timestamp": timestamp}

    textfeat_compressed = None
    for featname in nodelist[nodetype]["feat"]:
        path, extra, featemb = featlist[featname]
        is_target_column, dtype, num_categories = extra
        
        featembs[featname] = torch.from_numpy(featemb)

        feattensor = np.load(osp.join(srcpath, path))
        if "Griffin_text_" in featname:
            ufeattensor, feattensor = torch.unique(torch.from_numpy(feattensor), dim=0, return_inverse=True)
            if textfeat_compressed is None:
                textfeat_compressed = ufeattensor
            else:
                feattensor += textfeat_compressed.shape[0]
                textfeat_compressed = torch.concat((textfeat_compressed, ufeattensor), dim=0)
            print(featname, " compress to ", ufeattensor.shape[0], feattensor.shape[0])

        
        nodefeatdict[featname] = feattensor

    ds = Dataset.from_dict(nodefeatdict)
    ds.save_to_disk(osp.join(dstpath, "node/", nodetype, "feat"))
    ds = Dataset.from_dict({"emb": textfeat_compressed})
    ds.save_to_disk(osp.join(dstpath, "node/", nodetype, "textemb"))

from pqdm.processes import pqdm
pqdm(list(nodelist.keys()), process, n_jobs=args.ncpu)

exit()


    


