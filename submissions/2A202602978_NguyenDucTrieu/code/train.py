from __future__ import annotations

import os
import random
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pandas as pd
import time
import math

from dataset import load_split, check_split, build_transforms, make_loader
from model import build_model, param_groups, count_params, count_gmacs
from losses import build_criterion, class_weights, mix_batch, mixed_loss

import sys
import copy
import platform
import importlib.metadata
from inference import apply_temperature

for parent in Path(__file__).resolve().parents:
    if (parent / "eval.py").exists():
        sys.path.insert(0, str(parent))
        break
from eval import save_predictions, compute_metrics

@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False
    curves_dir: str = "curves"
    resume: bool = True
    pretrained: bool = True


def run_dir(cfg):
    return Path(cfg.out_dir)/cfg.exp_id/f'seed{cfg.seed}'


def pred_path(cfg, split):
    return Path(cfg.pred_dir)/f'{cfg.exp_id}_seed{cfg.seed}_{split}.csv'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg):
    return torch.optim.AdamW(param_groups(model,cfg.lr_backbone,cfg.lr_head,cfg.weight_decay))


def build_scheduler(optimizer, cfg, steps_per_epoch):
    total = cfg.epochs * steps_per_epoch
    warmup = int(cfg.warmup_epochs * steps_per_epoch)
    def multiplier(step):
        if step < warmup:
            return (step+1)/max(1,warmup)
        progress = min(1., (step-warmup)/max(1,total-warmup))
        return .5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class EMA:
    """Average floating parameters; copy BN statistics and integer counters."""
    def __init__(self, model, decay):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0,1)')
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for target, source in zip(self.model.parameters(), model.parameters()):
            target.lerp_(source, 1-self.decay)
        for target, source in zip(self.model.buffers(), model.buffers()):
            target.copy_(source)

    def copy_to(self, model):
        model.load_state_dict(self.model.state_dict())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=None):
    model.train()
    if getattr(model, 'frozen_backbone', False):
        model.eval()
        model.get_classifier().train()
    total, count = 0., 0
    for x,y,_ in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if cfg.mix:
            x, targets = mix_batch(x,y,cfg.mix_alpha,cfg.mix)
        with torch.autocast(device.type, enabled=cfg.amp and device.type=='cuda'):
            logits = model(x)
            loss = mixed_loss(criterion,logits,targets) if cfg.mix else criterion(logits,y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        scaler.scale(loss).backward()
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
            if ema:
                ema.update(model)
        total += loss.item()*len(y)
        count += len(y)
    if not count:
        raise ValueError('Empty training loader: reduce batch size')
    return {'train_loss':total/count, 'lr':optimizer.param_groups[0]['lr']}


def evaluate(model, loader, criterion, device):
    model.eval()
    names, labels, zs, total = [], [], [], 0.
    with torch.inference_mode():
        for x,y,filenames in loader:
            x,y = x.to(device),y.to(device)
            z = model(x).float()
            total += criterion(z,y).item()*len(y)
            names.extend(filenames)
            labels.append(y.cpu().numpy())
            zs.append(z.cpu().numpy())
    y,z = np.concatenate(labels),np.concatenate(zs)
    return names,y,z,total/len(y)


def plot_curves(history,path,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for field in ['train_loss','val_loss']:
        axes[0].plot(df.epoch,df[field],label=field)
    axes[1].plot(df.epoch,df.val_macro_f1,label='val macro-F1 (9 classes)')
    axes[2].plot(df.epoch,df.lr,label='LR, first parameter group')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.2)
    axes[0].set_ylabel('Loss'); axes[1].set_ylabel('F1'); axes[2].set_ylabel('LR')
    fig.suptitle(title); fig.tight_layout()
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=150); plt.close(fig)


def atomic_save(obj,path):
    temporary = path.with_suffix('.tmp')
    torch.save(obj,temporary)
    temporary.replace(path)


def run(cfg):
    if cfg.save_test_predictions:
        raise ValueError('Use experiments.py final after locking validation choices; train.run only uses val')
    if cfg.epochs <= 0 or cfg.batch_size <= 0:
        raise ValueError('Positive epochs and batch size required')
    set_seed(cfg.seed)
    rd = run_dir(cfg)
    rd.mkdir(parents=True,exist_ok=True)
    config = asdict(cfg)
    if (rd/'config.json').exists():
        previous = json.loads((rd/'config.json').read_text())
        if previous != config:
            raise ValueError(f'Config changed for {rd}; use a new exp_id')
        if not cfg.resume:
            raise FileExistsError(rd)
    (rd/'config.json').write_text(json.dumps(config,indent=2))
    train_df,val_df,test_df = load_split(cfg.labels_dir,cfg.fold)
    split = check_split(train_df,val_df,test_df,cfg.images_dir)
    (rd/'split.json').write_text(json.dumps(split,indent=2))
    train_loader = make_loader(train_df,cfg.images_dir,build_transforms(True,cfg.img_size,cfg.aug),
                               cfg.batch_size,True,cfg.sampler,cfg.num_workers)
    val_loader = make_loader(val_df,cfg.images_dir,build_transforms(False,cfg.img_size),
                             cfg.batch_size,False,None,cfg.num_workers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg.backbone,cfg.pretrained,9,cfg.drop_rate,cfg.init).to(device)
    versions = {x:importlib.metadata.version(x) for x in ['torch','torchvision','timm','numpy','pandas']}
    metadata = dict(versions=versions, python=platform.python_version(),
                    device=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',
                    pretrained_cfg=getattr(model,'pretrained_cfg',{}),
                    normalization='ImageNet mean/std; resize round(size*256/224), center crop',
                    ema_buffers='copy current model BN statistics; no post-training recalibration')
    (rd/'environment.json').write_text(json.dumps(metadata,indent=2,default=str))
    kwargs = dict(smoothing=cfg.label_smoothing,gamma=cfg.focal_gamma)
    if cfg.loss=='ce_weighted':
        kwargs['weight'] = class_weights(train_df.Label.value_counts().reindex(range(9),fill_value=0),
                                         cfg.class_weight_beta or 0.).to(device)
    criterion = build_criterion(cfg.loss,**kwargs).to(device)
    val_criterion = build_criterion('ce')
    optimizer = build_optimizer(model,cfg)
    scheduler = build_scheduler(optimizer,cfg,len(train_loader))
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and device.type=='cuda')
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type=='cuda')
    ema = EMA(model,cfg.ema_decay) if cfg.ema_decay is not None else None
    history,best,best_epoch,start = [],-1.,0,1
    if cfg.resume and (rd/'last.pt').exists():
        state = torch.load(rd/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        if ema: ema.model.load_state_dict(state['ema'])
        history,best,best_epoch,start = state['history'],state['best'],state['best_epoch'],state['epoch']+1
        random.setstate(state['random']); np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    for epoch in range(start,cfg.epochs+1):
        started = time.perf_counter()
        row = train_one_epoch(model,train_loader,criterion,optimizer,scheduler,scaler,cfg,device,ema)
        eval_model = ema.model if ema else model
        names,y,z,loss = evaluate(eval_model,val_loader,val_criterion,device)
        probs = apply_temperature(z,1.)
        metrics = compute_metrics(y,probs.argmax(1),probs)
        row.update(epoch=epoch,val_loss=loss,val_macro_f1=metrics['macro_f1'],
                   val_top1=metrics['top1'],seconds=time.perf_counter()-started)
        history.append(row)
        if metrics['macro_f1'] > best:
            best,best_epoch = metrics['macro_f1'],epoch
            atomic_save(eval_model.state_dict(),rd/'best.pt')
            np.save(rd/'val_logits.npy',z)
            save_predictions(pred_path(cfg,'val'),names,y,probs)
        atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                    ema=ema.model.state_dict() if ema else None, history=history,
                    best=best,best_epoch=best_epoch,epoch=epoch,random=random.getstate(),
                    numpy=np.random.get_state(),torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else []),rd/'last.pt')
        pd.DataFrame(history).to_csv(rd/'history.csv',index=False)
        plot_curves(history,Path(cfg.curves_dir)/f'{cfg.exp_id}_{cfg.backbone}_seed{cfg.seed}.png',
                    f'{cfg.exp_id} / {cfg.backbone} / seed {cfg.seed}')
        print(json.dumps(row),flush=True)
    from benchmark import latency_report
    summary = dict(best_epoch=best_epoch,best_val_f1=best,params_M=count_params(model),
                   gmacs=count_gmacs(model,cfg.img_size),gmac_note='THOP estimate; unsupported ops excluded',
                   train_seconds_per_epoch=float(np.mean([h['seconds'] for h in history])))
    if (rd/'summary.json').exists():
        return json.loads((rd/'summary.json').read_text())
    summary['latency'] = latency_report(model,1,cfg.img_size,device=str(device))
    (rd/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def parse_overrides(items):
    result = {}
    for item in items:
        key,value = item.split('=',1)
        if key not in Config.__dataclass_fields__:
            raise ValueError(f'Unknown config field: {key}')
        try: value = json.loads(value)
        except json.JSONDecodeError: pass
        result[key] = value
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--set',nargs='*',default=[])
    args = parser.parse_args()
    run(Config(**parse_overrides(args.set)))


if __name__=='__main__': main()
