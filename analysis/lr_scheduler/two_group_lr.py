import torch
import torch.nn as nn
def run(eta_min, epochs=100):
    head = nn.Linear(4, 2)
    back = nn.Linear(4, 4)
    opt = torch.optim.AdamW([{"params": head.parameters(), "lr": 1e-3},
                             {"params": back.parameters(), "lr": 1e-5}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=eta_min)
    rows = []
    for ep in range(1, epochs + 1):
        rows.append((ep, opt.param_groups[0]["lr"], opt.param_groups[1]["lr"]))
        opt.step()
        sch.step()
    return rows
for em in (0.0, 1e-5):
    rows = run(em)
    print(f"--- eta_min={em}")
    for ep in (1, 10, 25, 50, 75, 80, 90, 100):
        _, h, b = rows[ep - 1]
        print(f"epoch {ep:3d}  head_lr={h:.3e}  backbone_lr={b:.3e}  ratio={h/b:8.1f}")
