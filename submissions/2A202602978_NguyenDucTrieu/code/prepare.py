"""Download official fold 0, verify every file, produce EDA and pipeline checks."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import torch
from PIL import Image
from dataset import load_split, check_split, build_transforms, make_loader, IMAGENET_MEAN, IMAGENET_STD


def download(data='data'):
    root = Path(data)
    (root/'labels').mkdir(parents=True,exist_ok=True)
    for name in ['labels','train_subset0','val_subset0','test_subset0']:
        target = root/'labels'/f'{name}.csv'
        if not target.exists():
            urllib.request.urlretrieve(f'https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/{name}.csv',target)
    archive = root/'images.zip'
    if not archive.exists():
        urllib.request.urlretrieve('https://zenodo.org/records/7939060/files/images.zip?download=1',archive)
    with archive.open('rb') as f:
        checksum = hashlib.file_digest(f,'md5').hexdigest()
    if checksum != 'b7b30f96d466fba86016aa5a26606e0f':
        raise ValueError('images.zip MD5 mismatch; remove file and download again')
    images = root/'images'
    images.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir() or Path(item.filename).suffix.lower() not in {'.jpg','.jpeg','.png'}:
                continue
            target = images/Path(item.filename).name
            if not target.exists() or target.stat().st_size != item.file_size:
                target.write_bytes(z.read(item))
    return check_split(*load_split(root/'labels'),images)


def eda(data='data', output='eda'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,out = Path(data),Path(output)
    out.mkdir(parents=True,exist_ok=True)
    frames = load_split(root/'labels')
    stats = check_split(*frames,root/'images')
    labels = pd.read_csv(root/'labels/labels.csv')
    names = labels.groupby('Label').Species.first().reindex(range(9)).tolist()
    counts = pd.DataFrame(stats['per_class']).sort_index()
    counts.index = names
    counts.to_csv(out/'class_counts.csv')
    counts.plot.bar(figsize=(12,5)); plt.ylabel('Images'); plt.tight_layout()
    plt.savefig(out/'class_counts.png',dpi=150)
    plt.savefig(out/'class_distribution.png',dpi=150)
    plt.close()
    stats['imbalance_ratio'] = float(counts.sum(axis=1).max()/counts.sum(axis=1).min())
    # Visual examples and image statistics use training data only.
    shapes = {}
    for name in frames[0].Filename:
        with Image.open(root/'images'/name) as img:
            key = f'{img.size}/{img.mode}'
            shapes[key] = shapes.get(key,0)+1
    stats['train_image_shapes_modes'] = shapes
    stats['train_img_size'] = '(256, 256)'
    stats['train_channels'] = 3
    if shapes:
        first_key = list(shapes.keys())[0]
        if '/' in first_key:
            stats['train_img_size'] = first_key.split('/')[0]
            mode = first_key.split('/')[1]
            stats['train_channels'] = 3 if mode == 'RGB' else 1
    fig,axes = plt.subplots(9,3,figsize=(9,24))
    for label in range(9):
        sample = frames[0][frames[0].Label==label].sample(3,random_state=0)
        for ax,filename in zip(axes[label],sample.Filename):
            with Image.open(root/'images'/filename) as img: ax.imshow(img)
            ax.set_title(names[label]); ax.axis('off')
    fig.tight_layout(); fig.savefig(out/'examples.png',dpi=100); plt.close(fig)
    loader = make_loader(frames[0],root/'images',build_transforms(True),8,True,num_workers=0)
    x,y,_ = next(iter(loader))
    fig,axes = plt.subplots(2,4,figsize=(12,6))
    for ax,img,label in zip(axes.flat,x,y):
        restored = img.numpy().transpose(1,2,0)*np.array(IMAGENET_STD)+np.array(IMAGENET_MEAN)
        ax.imshow(restored.clip(0,1)); ax.set_title(names[int(label)]); ax.axis('off')
    fig.tight_layout(); fig.savefig(out/'augmentation.png',dpi=120); plt.close(fig)
    (out/'stats.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2))
    return stats


def sanity(data='data',output='eda',backbone='resnet50'):
    from train import set_seed
    from model import build_model
    set_seed(0)
    df,_,_ = load_split(Path(data)/'labels')
    tiny = df.groupby('Label').head(1)
    loader = make_loader(tiny,Path(data)/'images',build_transforms(False),9,False,num_workers=0)
    x,y,_ = next(iter(loader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(backbone,True).to(device)
    x,y = x.to(device),y.to(device)
    model.eval()
    with torch.no_grad(): initial = torch.nn.functional.cross_entropy(model(x),y).item()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.)
    losses = []
    for step in range(200):
        model.train(); optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x),y)
        loss.backward(); optimizer.step(); losses.append(loss.item())
        if loss.item()<.02: break
    result = dict(initial_ce=float(initial),uniform_ce=float(np.log(9)),final_loss=float(losses[-1]),
                  steps=int(len(losses)),passed=bool(losses[-1]<.02),initial_ce_near_uniform=bool(abs(initial-np.log(9))<1.))
    out = Path(output); out.mkdir(parents=True,exist_ok=True)
    (out/'sanity.json').write_text(json.dumps(result,indent=2))
    if not result['passed'] or not result['initial_ce_near_uniform']:
        raise RuntimeError(f'Pipeline check needs investigation: {result}')
    return result


if __name__=='__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--data',default='data'); p.add_argument('--output',default='eda')
    p.add_argument('--sanity',action='store_true'); args = p.parse_args()
    download(args.data); eda(args.data,args.output)
    if args.sanity: print(sanity(args.data,args.output))
