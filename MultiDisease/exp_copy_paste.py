"""
Experimento: o copy-paste de lesões ajuda? Treina com p_paste em {0, 0.25, 0.5} e 3 sementes.
Uso (dentro de Steps/MultiDisease):  python exp_copy_paste.py
"""
import json, os, sys, time
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import average_precision_score

import multi_utils as mu

BRACOL = os.getenv("BRACOL_PATH", "/media/kaua-matheus/HD3201/Data/Agroscope/Coffee/BRACOL/repo")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, BATCH, LR = 20, 32, 5e-4

leaves, boxes, crops = mu.Load_Bracol(BRACOL)
split = mu.Split(leaves, boxes)
bank = mu.Lesion_Bank(crops, split["train"])
val_loader = torch.utils.data.DataLoader(mu.Leaf_Dataset(split["val"], leaves, mu.Eval_Transform()), batch_size=64, num_workers=6)
test_loader = torch.utils.data.DataLoader(mu.Leaf_Dataset(split["test"], leaves, mu.Eval_Transform()), batch_size=64, num_workers=6)
tl = np.stack([leaves[i]["labels"] for i in split["train"]])
pos_weight = torch.tensor(np.clip((len(tl) - tl.sum(0)) / tl.sum(0), 1, 4), dtype=torch.float32, device=DEVICE)


def run(p_paste, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    import random; random.seed(seed)
    ds = mu.Leaf_Dataset(split["train"], leaves, mu.Train_Transform(), bank=bank, p_paste=p_paste)
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=8, persistent_workers=True, drop_last=True, generator=g)
    net = mu.Build_Multi_Model().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(loader), pct_start=0.1)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best, state = -1, None
    for ep in range(EPOCHS):
        net.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                loss = crit(net(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        P, Y = mu.Predict_Probs(net, val_loader, DEVICE)
        m = average_precision_score(Y, P, average="macro")
        if m > best:
            best, state = m, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    Pv, Yv = mu.Predict_Probs(net, val_loader, DEVICE)
    thr = mu.Best_Thresholds(Pv, Yv)
    P, Y = mu.Predict_Probs(net, test_loader, DEVICE)
    pred = (P >= thr).astype(int)
    multi = Y.sum(1) >= 2
    return {
        "p_paste": p_paste, "seed": seed, "val_mAP": best,
        "test_mAP": average_precision_score(Y, P, average="macro"),
        "test_AP": [average_precision_score(Y[:, k], P[:, k]) for k in range(4)],
        "exact": float((pred == Y).all(1).mean()),
        "multi_all_found": float((pred[multi] >= Y[multi]).all(1).mean()),
        "multi_exact": float((pred[multi] == Y[multi]).all(1).mean()),
    }


if __name__ == "__main__":
    out = []
    for seed in (42, 7, 123):
        for p in (0.0, 0.25, 0.5):
            t = time.time()
            r = run(p, seed)
            out.append(r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "test_AP"}), f"{time.time() - t:.0f}s", flush=True)
    json.dump(out, open("exp_copy_paste_results.json", "w"), indent=1)
    print("\nMédia (3 sementes):")
    for p in (0.0, 0.25, 0.5):
        rs = [r for r in out if r["p_paste"] == p]
        f = lambda k: f"{np.mean([r[k] for r in rs]) * 100:.1f} ± {np.std([r[k] for r in rs]) * 100:.1f}"
        ap = np.mean([r["test_AP"] for r in rs], 0)
        print(f"p_paste={p}: mAP teste {f('test_mAP')} | exato {f('exact')} | 2+ doenças todas {f('multi_all_found')} | "
              f"AP por doença " + " ".join(f"{d} {a:.3f}" for d, a in zip(mu.DISEASES, ap)))
