from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def build_criterion(kind: str = "ce", **kw):
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        smoothing = kw.get("smoothing", 0.1)
        return LabelSmoothingCE(smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", 2.0)
        alpha = kw.get("alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Unknown criterion kind: {kind}")

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits, target):
        return self.ce(logits, target)

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
            
        return loss.mean()

def class_weights(counts, beta: float = 0.0):
    counts = np.array(counts)
    if (counts <= 0).any() or not 0 <= beta < 1:
        raise ValueError("Positive counts and 0 <= beta < 1 required")
    if beta == 0.0:
        w = 1.0 / counts
        w = w / w.mean()
    else:
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
        w = w / w.sum() * len(counts)
    return torch.tensor(w, dtype=torch.float32)

def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)
    y_a, y_b = y, y[perm]
    
    if mode == "mixup":
        x_mix = lam * x + (1 - lam) * x[perm]
    elif mode == "cutmix":
        W, H = x.size(3), x.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        x_mix = x.clone()
        x_mix[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    return x_mix, (y_a, y_b, lam)

def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
