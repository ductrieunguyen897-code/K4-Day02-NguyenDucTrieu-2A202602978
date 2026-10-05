import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import pandas as pd
from PIL import Image
from losses import FocalLoss, mix_batch, mixed_loss
from model import param_groups
from inference import fit_temperature, apply_temperature, aggregate_views, fuse_conv_bn, views_multicrop
from train import EMA, Config, run, train_one_epoch, build_scheduler
from eval import read_pred, compute_metrics
from dataset import check_split

torch.set_num_threads(2)

class Tiny(nn.Module):
    def __init__(self):
        super().__init__(); self.features=nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.BatchNorm2d(8),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.head=nn.Linear(8,9)
    def forward(self,x): return self.head(self.features(x).flatten(1))
    def get_classifier(self): return self.head

class SolutionTests(unittest.TestCase):
    def test_focal_gamma_zero(self):
        x=torch.randn(13,9,requires_grad=True); y=torch.randint(9,(13,))
        a=FocalLoss(0)(x,y); b=nn.CrossEntropyLoss()(x,y)
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(torch.autograd.grad(a,x,retain_graph=True)[0],torch.autograd.grad(b,x)[0])

    def test_mix_labels(self):
        x=torch.randn(10,3,12,12); y=torch.arange(10)%9
        for mode in ['mixup','cutmix']:
            out,labels=mix_batch(x,y,mode=mode)
            self.assertEqual(out.shape,x.shape); self.assertTrue(0<=labels[2]<=1)
            z=torch.randn(10,9); loss=mixed_loss(nn.CrossEntropyLoss(),z,labels)
            self.assertTrue(torch.isfinite(loss))

    def test_no_decay_bias_norm(self):
        m=Tiny(); groups=param_groups(m,1e-4,1e-3,.05)
        assigned=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(assigned),len(set(assigned)))
        for name,p in m.named_parameters():
            g=next(g for g in groups if any(q is p for q in g['params']))
            if p.ndim<=1: self.assertEqual(g['weight_decay'],0)
            self.assertEqual(g['lr'],1e-3 if name.startswith('head') else 1e-4)

    def test_ema_frozen_and_integer_buffers(self):
        m=Tiny(); m.features.requires_grad_(False); ema=EMA(m,.9)
        m.train(); m(torch.randn(3,3,12,12)); ema.update(m)
        target=Tiny(); ema.copy_to(target)
        self.assertEqual(target.features[1].num_batches_tracked.dtype,torch.int64)
        torch.testing.assert_close(target.features[0].weight,m.features[0].weight)

    def test_calibration_and_aggregation(self):
        rng=np.random.default_rng(12); z=rng.normal(size=(300,9))*5; y=rng.integers(9,size=300)
        t=fit_temperature(z,y); p=apply_temperature(z,t); base=apply_temperature(z,1)
        np.testing.assert_array_equal(p.argmax(1),base.argmax(1))
        self.assertLessEqual(compute_metrics(y,p.argmax(1),p)['nll'],compute_metrics(y,base.argmax(1),base)['nll'])
        np.testing.assert_allclose(aggregate_views([z,z]),base)
        with self.assertRaises(ValueError): apply_temperature(z,0)

    def test_bn_fusion_graph(self):
        m=Tiny().eval(); x=torch.randn(3,3,12,12); f=fuse_conv_bn(m)
        torch.testing.assert_close(m(x),f(x),atol=1e-5,rtol=1e-5)
        self.assertIsInstance(m.features[1],nn.BatchNorm2d)

    def test_crops(self):
        x=torch.arange(36).reshape(1,1,6,6)
        crops=views_multicrop(x,4)
        self.assertEqual(len(crops),5); torch.testing.assert_close(crops[-1],x[:,:,1:5,1:5])

    def test_split_rejects_missing_and_duplicates(self):
        df=pd.DataFrame({'Filename':['a.jpg'],'Label':[0]})
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError): check_split(df,df,df,d)
            Image.new('RGB',(8,8)).save(Path(d)/'a.jpg')
            with self.assertRaises(ValueError): check_split(df,df,df,d)

    def test_tiny_run_predictions_resume(self):
        # Synthetic data checks mechanics only; never reported as DeepWeeds results.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); images=root/'images'; images.mkdir(); labels=root/'labels'; labels.mkdir()
            for split in ['train','val','test']:
                rows=[]
                for i in range(9):
                    name=f'{split}{i}.jpg'; Image.new('RGB',(32,32),(i*25,50,150)).save(images/name); rows.append({'Filename':name,'Label':i})
                pd.DataFrame(rows).to_csv(labels/f'{split}_subset0.csv',index=False)
            cfg=Config(exp_id='B01',epochs=1,batch_size=3,img_size=32,num_workers=0,amp=False,pretrained=False,
                images_dir=str(images),labels_dir=str(labels),out_dir=str(root/'runs'),pred_dir=str(root/'predictions'),curves_dir=str(root/'curves'))
            with patch('train.build_model',side_effect=lambda *a: Tiny()),patch('train.check_split',return_value={'synthetic':True}):
                summary=run(cfg); resumed=run(cfg)
            self.assertEqual(summary['best_val_f1'],resumed['best_val_f1'])
            p=read_pred(str(root/'predictions/B01_seed0_val.csv'))
            self.assertEqual(len(p.y_true),9)
            self.assertFalse((root/'predictions/B01_seed0_test.csv').exists())
            self.assertTrue(list((root/'curves').glob('*.png')))
            from report import generate
            generate(root)
            self.assertTrue((root/'results.xlsx').exists())
            # Exercise validation methods, locked 3-seed finals, official score/grade,
            # and report generation end-to-end on synthetic data in a temporary folder.
            import json
            from dataclasses import asdict, replace
            from experiments import inference_stage, final_stage
            selected=replace(cfg,exp_id='T00')
            pd.DataFrame({'Filename':[f'train{i}.jpg' for i in range(9)],'Label':range(9),
                          'Species':['Chinee apple','Lantana','Parkinsonia','Parthenium','Prickly acacia',
                                     'Rubber vine','Siam weed','Snake weed','Negative']}).to_csv(labels/'labels.csv',index=False)
            for name in ['selected_backbone.json','selected_training.json']:
                (root/name).write_text(json.dumps(asdict(selected)))
            with patch('train.build_model',side_effect=lambda *a: Tiny()), patch('experiments.build_model',side_effect=lambda *a: Tiny()), patch('train.check_split',return_value={'synthetic':True}):
                run(selected)
                inference_stage(root)
                final_stage(root)
                final_stage(root)  # Completed test predictions must be reused.
            self.assertEqual(len(list((root/'predictions').glob('F01_seed*_test.csv'))),3)
            self.assertTrue((root/'final_lock.json').exists())
            self.assertTrue((root/'grade.txt').exists())
            generate(root)
            final=pd.read_excel(root/'results.xlsx',sheet_name='Final')
            self.assertEqual(len(final),8)


if __name__=='__main__': unittest.main()
