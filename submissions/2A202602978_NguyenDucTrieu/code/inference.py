"""Deterministic inference; temperature must be fitted on validation only."""
from __future__ import annotations
import copy
import numpy as np
import torch
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize_scalar


def predict_logits(model, loader, device, view=None):
    model.eval()
    names, labels, logits = [], [], []
    with torch.inference_mode():
        for x, y, filenames in loader:
            x = x.to(device)
            z = model(view(x) if view else x)
            names.extend(filenames)
            labels.append(y.numpy())
            logits.append(z.float().cpu().numpy())
    return names, np.concatenate(labels), np.concatenate(logits)


def view_identity(x):
    return x


def view_hflip(x):
    return x.flip(-1)


def views_multicrop(x, crop: int):
    h, w = x.shape[-2:]
    if crop <= 0 or crop > min(h, w):
        raise ValueError("Crop must fit inside image")
    return [x[..., a:a+crop, b:b+crop] for a,b in
            [(0,0),(0,w-crop),(h-crop,0),(h-crop,w-crop),((h-crop)//2,(w-crop)//2)]]


def views_multiscale(x, sizes):
    return [torch.nn.functional.interpolate(x, size=(s,s), mode="bilinear",
            align_corners=False, antialias=True) for s in sizes]


def aggregate_views(logits_per_view, space="prob"):
    z = np.stack(logits_per_view)
    if space == "prob":
        return softmax(z, axis=-1).mean(0)
    if space == "logit":
        return softmax(z.mean(0), axis=-1)
    raise ValueError("space must be prob or logit")


def ensemble_probs(list_of_probs):
    p = np.stack(list_of_probs)
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1):
        raise ValueError("Expected normalized probabilities with matching rows/classes")
    return p.mean(0)


def fit_temperature(val_logits, val_labels):
    z, y = np.asarray(val_logits, dtype=np.float64), np.asarray(val_labels, dtype=int)
    def nll(log_t):
        scaled = z / np.exp(log_t)
        return (logsumexp(scaled, axis=1)-scaled[np.arange(len(y)), y]).mean()
    result = minimize_scalar(nll, bounds=(-5,5), method="bounded")
    return float(np.exp(result.x)) if result.success and result.fun < nll(0.) else 1.0


def apply_temperature(logits, T):
    if not np.isfinite(T) or T <= 0:
        raise ValueError("Temperature must be finite and positive")
    return softmax(np.asarray(logits, dtype=np.float64)/T, axis=-1)


def fuse_conv_bn(model):
    """FX traces actual graph edges, never guesses execution order from child names."""
    from torch.fx import symbolic_trace
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    result = symbolic_trace(copy.deepcopy(model).eval())
    modules = dict(result.named_modules())
    for node in list(result.graph.nodes):
        if node.op != 'call_module' or not isinstance(modules[node.target], torch.nn.BatchNorm2d):
            continue
        previous = node.args[0]
        if previous.op != 'call_module' or len(previous.users) != 1:
            continue
        conv = modules[previous.target]
        if not isinstance(conv, torch.nn.Conv2d):
            continue
        result.set_submodule(previous.target, fuse_conv_bn_eval(conv, modules[node.target]))
        node.replace_all_uses_with(previous)
        result.graph.erase_node(node)
    result.graph.lint()
    result.recompile()
    return result
