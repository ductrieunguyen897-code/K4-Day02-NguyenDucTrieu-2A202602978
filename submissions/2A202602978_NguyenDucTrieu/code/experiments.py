"""Staged validation selection. Final configuration is locked before reading test predictions."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from train import Config, run, run_dir, pred_path, save_predictions, compute_metrics
from dataset import load_split, build_transforms, make_loader
from model import build_model
from inference import apply_temperature, fit_temperature, predict_logits
from benchmark import latency_report

BACKBONES = ['resnet50','resnext50_32x4d','convnext_tiny','deit_small_patch16_224','efficientnet_b0']
# Each T01..T08 differs from T00 in exactly one experimental factor.
ABLATIONS = [('T01','A',{'init':'scratch'}),('T02','A',{'init':'frozen'}),
             ('T03','B',{'aug':'color'}),('T04','B',{'mix':'mixup'}),('T05','B',{'mix':'cutmix'}),
             ('T06','C',{'loss':'ls','label_smoothing':.1}),('T07','C',{'loss':'focal'}),
             ('T08','C',{'loss':'ce_weighted'})]
METHODS = {'I00':('single',1), 'I01':('hflip_prob',2), 'I02':('hflip_logit',2),
           'I03':('fivecrop',5), 'I04':('temperature',1)}


def dump(path,obj):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))


def base_config(root,data,epochs=12,batch=64):
    root,data = Path(root).resolve(),Path(data).resolve()
    return Config(epochs=epochs,batch_size=batch,images_dir=str(data/'images'),labels_dir=str(data/'labels'),
                  out_dir=str(root/'runs'),pred_dir=str(root/'predictions'),curves_dir=str(root/'curves'))


def read_cfg(path):
    return Config(**json.loads(Path(path).read_text()))


def result(cfg):
    return json.loads((run_dir(cfg)/'summary.json').read_text())['best_val_f1']


def load_model(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg.backbone,False,9,cfg.drop_rate,cfg.init).to(device).eval()
    model.load_state_dict(torch.load(run_dir(cfg)/'best.pt',map_location=device,weights_only=True))
    return model,device


class Predictor(torch.nn.Module):
    def __init__(self,model,method,size,temperature=1.):
        super().__init__(); self.model=model; self.method=method; self.size=size; self.temperature=temperature
    def forward(self,x):
        if self.method=='fivecrop':
            from inference import views_multicrop
            views = views_multicrop(x,self.size)
        elif self.method.startswith('hflip'):
            views = [x,x.flip(-1)]
        else: views = [x]
        z = torch.stack([self.model(v) for v in views])
        if self.method=='hflip_logit': return z.mean(0).softmax(-1)
        if self.method=='temperature': return (z[0]/self.temperature).softmax(-1)
        return z.softmax(-1).mean(0)


def method_loader(cfg,df,method):
    size = cfg.img_size
    if method=='fivecrop':
        from torchvision.transforms import v2
        from dataset import IMAGENET_MEAN,IMAGENET_STD
        transform = v2.Compose([v2.Resize(round(size*256/224)),v2.CenterCrop(round(size*256/224)),
                    v2.ToImage(),v2.ToDtype(torch.float32,scale=True),v2.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    else: transform=build_transforms(False,size)
    return make_loader(df,cfg.images_dir,transform,cfg.batch_size,False,num_workers=cfg.num_workers)


def predict_method(cfg,model,device,df,method,temperature=1.):
    wrapper = Predictor(model,method,cfg.img_size,temperature).eval()
    names,ys,ps = [],[],[]
    with torch.inference_mode():
        for x,y,n in method_loader(cfg,df,method):
            ps.append(wrapper(x.to(device)).float().cpu().numpy()); ys.append(y.numpy()); names.extend(n)
    return names,np.concatenate(ys),np.concatenate(ps)


def backbone_stage(cfg,root):
    candidates = [replace(cfg,exp_id=f'B{i:02}',backbone=b) for i,b in enumerate(BACKBONES,1)]
    for c in candidates: run(c)
    best = max(candidates,key=result)
    dump(root/'selected_backbone.json',asdict(replace(best,exp_id='T00')))


def training_stage(root):
    baseline = read_cfg(root/'selected_backbone.json'); run(baseline)
    candidates = [baseline]
    for exp,axis,changes in ABLATIONS:
        c = replace(baseline,exp_id=exp,**changes); run(c); candidates.append(c)
    # Best per axis by val, include baseline; combined configuration is explicitly non-causal.
    changes = {}
    for axis in ['A','B','C']:
        group = [baseline]+[c for c in candidates if any(c.exp_id==e and a==axis for e,a,_ in ABLATIONS)]
        winner = max(group,key=result)
        if winner.exp_id!='T00':
            changes.update(next(delta for exp,a,delta in ABLATIONS if exp==winner.exp_id))
    combined = replace(baseline,exp_id='T09',**changes); run(combined); candidates.append(combined)
    best = max(candidates,key=result)
    dump(root/'selected_training.json',asdict(best))


def inference_stage(root):
    cfg = read_cfg(root/'selected_training.json')
    _,val,_ = load_split(cfg.labels_dir,cfg.fold)
    model,device = load_model(cfg)
    z = np.load(run_dir(cfg)/'val_logits.npy')
    temperature = fit_temperature(z,val.Label.to_numpy())
    rows,latencies = [],[]
    for exp,(method,k) in METHODS.items():
        names,y,p = predict_method(cfg,model,device,val,method,temperature)
        metrics = compute_metrics(y,p.argmax(1),p)
        save_predictions(root/'predictions'/f'{exp}_seed{cfg.seed}_val.csv',names,y,p)
        wrapper=Predictor(model,method,cfg.img_size,temperature)
        input_size=round(cfg.img_size*256/224) if method=='fivecrop' else cfg.img_size
        timings=[]
        for batch in [1,32]:
            latency=latency_report(wrapper,batch,input_size,device=str(device))
            latency.update(exp_id=exp,method=method)
            timings.append(latency); latencies.append(latency)
        rows.append(dict(exp_id=exp,method=method,K=k,checkpoint=str(run_dir(cfg)/'best.pt'),
                         macro_f1=metrics['macro_f1'],top1=metrics['top1'],ece=metrics['ece'],
                         p50=timings[0]['p50'],p95=timings[0]['p95'],p99=timings[0]['p99'],
                         images_per_s=timings[1]['images_per_s'],temperature=temperature if method=='temperature' else 1.))
        dump(root/'inference.json',rows); dump(root/'latency.json',latencies)
    for row in rows: row['relative_cost']=row['p50']/rows[0]['p50']
    dump(root/'inference.json',rows)
    # Macro-F1 first, ECE breaks ties, then latency. Declared before final testing.
    best = sorted(rows,key=lambda r:(-r['macro_f1'],r['ece'],r['p50']))[0]
    dump(root/'selected_inference.json',best)


def final_stage(root):
    selected=read_cfg(root/'selected_training.json'); baseline=read_cfg(root/'selected_backbone.json')
    method=json.loads((root/'selected_inference.json').read_text())['method']
    lock=dict(selected=asdict(selected),baseline=asdict(baseline),method=method,seeds=[0,1,2])
    lock_path=root/'final_lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text())!=lock:
        raise ValueError('Final choices already locked. Do not tune after test evaluation.')
    dump(lock_path,lock)
    # Finish all seed training and val calibration before evaluating any test image.
    configs=[]
    for exp,source in [('T00',baseline),('F01',selected)]:
        for seed in lock['seeds']:
            cfg=replace(source,exp_id=exp,seed=seed); run(cfg); configs.append(cfg)
    for cfg in configs:
        method_used=method if cfg.exp_id=='F01' else 'single'
        _,val,test=load_split(cfg.labels_dir,cfg.fold)
        z=np.load(run_dir(cfg)/'val_logits.npy')
        temperature=fit_temperature(z,val.Label.to_numpy()) if method_used=='temperature' else 1.
        target=pred_path(cfg,'test')
        receipt=run_dir(cfg)/'test_receipt.json'
        if target.exists():
            if not receipt.exists(): raise RuntimeError('Test output exists without receipt; investigate before proceeding')
            continue
        marker=run_dir(cfg)/'test_started.json'
        if marker.exists():
            raise RuntimeError(f'Interrupted test in {marker}; document incident before any manual recovery')
        model,device=load_model(cfg)
        names,y,p=predict_method(cfg,model,device,val,method_used,temperature)
        save_predictions(pred_path(cfg,'val'),names,y,p)
        dump(marker,dict(method=method_used,temperature=temperature))
        if method_used=='temperature':
            # A single test forward pass produces both calibrated and uncalibrated predictions.
            names,y,z=predict_logits(model,method_loader(cfg,test,'single'),device)
            np.save(run_dir(cfg)/'test_logits.npy',z)
            save_predictions(target.with_name(f'{cfg.exp_id}_uncal_seed{cfg.seed}_test.csv'),names,y,apply_temperature(z,1.))
            p=apply_temperature(z,temperature)
        else:
            names,y,p=predict_method(cfg,model,device,test,method_used)
            # Log-probabilities are equivalent logits for aggregated inference; label this explicitly.
            np.save(run_dir(cfg)/'test_log_probs.npy',np.log(np.clip(p,1e-30,1.)))
        save_predictions(target,names,y,p)
        dump(receipt,dict(method=method_used,temperature=temperature,
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    eval_candidates = [p/'eval.py' for p in [Path.cwd(), *Path(__file__).resolve().parents]] + [Path(p)/'eval.py' for p in sys.path]
    eval_script = next(p for p in eval_candidates if p.exists())
    common=['--test-csv',str(Path(selected.labels_dir)/'test_subset0.csv'),'--labels',str(Path(selected.labels_dir)/'labels.csv')]
    for exp in ['T00','F01']:
        subprocess.run([sys.executable,str(eval_script),'score','--pred',str(root/'predictions'/f'{exp}_seed*_test.csv'),
                        *common,'--tag',exp,'--out',str(root/'eval_out')],check=True)
    command=[sys.executable,str(eval_script),'grade','--final',str(root/'predictions/F01_seed*_test.csv'),
             '--baseline',str(root/'predictions/T00_seed*_test.csv'),'--final-val',str(root/'predictions/F01_seed*_val.csv'),
             '--val-csv',str(Path(selected.labels_dir)/'val_subset0.csv'),*common]
    if method=='temperature': command+=['--uncal',str(root/'predictions/F01_uncal_seed*_test.csv')]
    grade=subprocess.run(command,check=True,capture_output=True,text=True)
    (root/'grade.txt').write_text(grade.stdout); print(grade.stdout)


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['backbones','training','inference','final','all'])
    p.add_argument('--root',default='.'); p.add_argument('--data',default='data')
    p.add_argument('--epochs',type=int,default=12); p.add_argument('--batch',type=int,default=64)
    args=p.parse_args(); root=Path(args.root).resolve(); root.mkdir(parents=True,exist_ok=True)
    # No full experiments before the prescribed sanity check has passed.
    sanity=json.loads((root/'eda/sanity.json').read_text())
    if not sanity['passed'] or not sanity['initial_ce_near_uniform']:
        raise RuntimeError('Run prepare.py --sanity first')
    stages=['backbones','training','inference','final'] if args.stage=='all' else [args.stage]
    for stage in stages:
        if stage!='final' and (root/'final_lock.json').exists():
            # An all-stage resume after final locking must not retune on validation.
            if args.stage=='all': continue
            raise RuntimeError('Choices locked; screening is closed')
        if stage=='backbones': backbone_stage(base_config(root,args.data,args.epochs,args.batch),root)
        elif stage=='training': training_stage(root)
        elif stage=='inference': inference_stage(root)
        else: final_stage(root)
    from report import generate
    generate(root)


if __name__=='__main__': main()
