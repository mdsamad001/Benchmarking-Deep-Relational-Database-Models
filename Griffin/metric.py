import torch
import torch.nn.functional as F
from torchmetrics.regression import (
    MeanSquaredError,
    MeanSquaredLogError,
    MeanAbsoluteError,
)
from torchmetrics.classification import AUROC, BinaryAUROC


LOWER_IS_BETTER = {"rmse", "mae", "mse", "retrieval_logloss"}


def metric_higher_is_better(metric_name: str) -> bool:
    return metric_name not in LOWER_IS_BETTER


def compute_metric(outputs, labels, metric):
    if metric == "rmse":
        score = MeanSquaredError(squared=False)(outputs.flatten(), labels).item()
    elif metric == "mae":
        score = MeanAbsoluteError()(outputs.flatten(), labels).item()
    elif metric == "mse":
        score = MeanSquaredError(squared=True)(outputs.flatten(), labels).item()
    elif metric == "hr@1":
        score = torch.mean((outputs.argmax(dim=1) == labels).to(torch.float)).item()
    elif metric in ("retrieval_auroc", "auc"):
        score = AUROC(task="multiclass", num_classes=outputs.shape[1])(
            outputs, labels
        ).item()
    elif metric == "retrieval_logloss":

        score = F.cross_entropy(outputs, labels).item()
    else:
        raise NotImplementedError(metric)
    return score