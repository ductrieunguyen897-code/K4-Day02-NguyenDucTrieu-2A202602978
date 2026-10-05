from __future__ import annotations

import torch
import torch.nn as nn
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}

def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError(f"Unknown initialization: {init}")
    if init == "scratch":
        pretrained = False
        
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )
    
    if init == "frozen":
        freeze_backbone(model)
        
    return model

def freeze_backbone(model) -> None:
    classifier_params = set(model.get_classifier().parameters())
    
    for param in model.parameters():
        if param not in classifier_params:
            param.requires_grad = False
            
    model.frozen_backbone = True

def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    head = {id(p) for p in model.get_classifier().parameters()}
    exempt = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = lr_head if id(param) in head else lr_backbone
        wd = 0.0 if param.ndim <= 1 or name.endswith(".bias") or name in exempt else weight_decay
        groups.setdefault((lr, wd), []).append(param)
    return [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def count_gmacs(model, img_size: int = 224) -> float:
    # THOP is a MAC estimate; unsupported operations are not counted.
    import copy
    from thop import profile
    m = copy.deepcopy(model).eval()
    x = torch.zeros(1, 3, img_size, img_size, device=next(m.parameters()).device)
    with torch.inference_mode():
        macs, _ = profile(m, inputs=(x,), verbose=False)
    return macs / 1e9
