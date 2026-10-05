"""Forward-only timings: no decoding/CPU preprocessing or host-device transfer."""
import copy
import time
import numpy as np
import torch


def bench(fn, warmup=10, iters=100, sync=None):
    if warmup < 10 or iters < 50:
        raise ValueError("Require >=10 warmup and >=50 timed iterations")
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(iters):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        timings.append((time.perf_counter()-start)*1000)
    return dict(zip(['p50','p95','p99'], map(float,np.percentile(timings,[50,95,99]))),
                mean=float(np.mean(timings)), n=iters)


def latency_report(model, batch_size, img_size, dtype='fp32', device='cuda', warmup=10, iters=100):
    if dtype not in {'fp32','amp','fp16'}:
        raise ValueError(dtype)
    device = torch.device(device)
    model = copy.deepcopy(model).to(device).eval().float()
    x = torch.randn(batch_size,3,img_size,img_size,device=device)
    if dtype == 'fp16':
        model, x = model.half(), x.half()
    with torch.inference_mode(), torch.autocast(device.type, enabled=dtype=='amp'):
        result = bench(lambda: model(x), warmup, iters,
                       (lambda: torch.cuda.synchronize(device)) if device.type=='cuda' else None)
    return dict(result, gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                dtype=dtype, batch=batch_size, img_size=img_size,
                images_per_s=batch_size/(result['p50']/1000), torch=torch.__version__,
                preprocessing=False, bn_fused=False)


def tta_latency(model, k_views, **kw):
    if k_views < 1:
        raise ValueError('k_views must be positive')
    class Repeated(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            return torch.stack([self.base(x if i%2==0 else x.flip(-1)).softmax(-1)
                                for i in range(k_views)]).mean(0)
    single = latency_report(model, **kw)
    result = latency_report(Repeated(model), **kw)
    return dict(result, k_views=k_views, relative_cost=result['p50']/single['p50'])
