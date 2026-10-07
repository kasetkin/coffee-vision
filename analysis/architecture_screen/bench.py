import time
import torch
import torch.nn as nn
import torchvision.models as models
torch.set_num_threads(6)

def timeit(fn, n=8, warmup=2):
    for _ in range(warmup):
        fn()
    t=time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter()-t)/n

def params(m): return sum(p.numel() for p in m.parameters())/1e6

class UNetR18(nn.Module):
    """ResNet18 encoder + U-Net decoder, classification head on bottleneck +
    segmentation head on decoder output (the multi-task shape)."""
    def __init__(self, nc=10):
        super().__init__()
        r = models.resnet18(weights=None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)   # /2, 64
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        def up(ci, cs, co):
            return nn.Sequential(nn.Conv2d(ci+cs, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                                 nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True))
        self.d4, self.d3, self.d2, self.d1 = up(512,256,256), up(256,128,128), up(128,64,64), up(64,64,32)
        self.seg = nn.Conv2d(32, 1, 1)
        self.fc = nn.Linear(512, nc)
    def forward(self, x):
        s0 = self.stem(x)
        x1 = self.l1(self.pool(s0))
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        u = nn.functional.interpolate(x4, scale_factor=2, mode="nearest")
        u = self.d4(torch.cat([u, x3], 1))
        u = nn.functional.interpolate(u, scale_factor=2, mode="nearest")
        u = self.d3(torch.cat([u, x2], 1))
        u = nn.functional.interpolate(u, scale_factor=2, mode="nearest")
        u = self.d2(torch.cat([u, x1], 1))
        u = nn.functional.interpolate(u, scale_factor=2, mode="nearest")
        u = self.d1(torch.cat([u, s0], 1))
        logits = self.fc(x4.mean((2,3)))
        return logits, self.seg(u)

B = 32
x = torch.randn(B,3,224,224)
rows=[]
for name, model, fwd in [
    ("resnet18 (current)", models.resnet18(weights=None), lambda m,x: m(x)),
    ("resnet34", models.resnet34(weights=None), lambda m,x: m(x)),
    ("resnet50", models.resnet50(weights=None), lambda m,x: m(x)),
    ("convnext_tiny", models.convnext_tiny(weights=None), lambda m,x: m(x)),
    ("efficientnet_b0", models.efficientnet_b0(weights=None), lambda m,x: m(x)),
    ("UNet-R18 (enc+dec, multitask)", UNetR18(), lambda m,x: m(x)[0]),
]:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    def step():
        opt.zero_grad(set_to_none=True)
        out = fwd(model, x)
        out.sum().backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        t_inf = timeit(lambda: fwd(model, x), n=6)
    model.train()
    t_tr = timeit(step, n=4, warmup=1)
    rows.append((name, params(model), t_inf/B*1000, t_tr))
    print(f"{name:34s} {params(model):7.1f}M  inf {t_inf/B*1000:7.2f} ms/img  train-step(B=32) {t_tr:6.2f} s")
