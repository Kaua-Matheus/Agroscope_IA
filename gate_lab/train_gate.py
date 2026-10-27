"""
Modelo de triagem: leaf_ok / leaf_poor / not_leaf.
- leaf_poor é gerado aplicando degradações SEVERAS (níveis que derrubam o especialista) em folhas.
- O tipo 'noise' fica fora do treino para testar se o modelo generaliza para um defeito nunca visto.
"""
import os, sys, json, random, argparse, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from degrade import LEVELS, apply

p = argparse.ArgumentParser()
p.add_argument('--arch', default='mobilenet_v3_large')
p.add_argument('--epochs', type=int, default=6)
p.add_argument('--bad', default='{"gauss":[3],"motion":[3],"dark":[3],"bright":[2,3],"lowres":[3],"jpeg":[3]}')
p.add_argument('--ok', default='{"gauss":[1],"motion":[1],"dark":[1],"bright":[1],"lowres":[1],"jpeg":[1,2]}')
p.add_argument('--p_bad', type=float, default=0.35)
p.add_argument('--out', default='gate.pth')
p.add_argument('--skip', default='{}')  # níveis ambíguos: não entram nem como ok nem como ruim
p.add_argument('--p_syn', type=float, default=0.12)
p.add_argument('--wheat', action='store_true')
a = p.parse_known_args()[0]
BAD, OK = json.loads(a.bad), json.loads(a.ok)
from synth import synth_negative
CLASSES = ['leaf_ok', 'leaf_poor', 'not_leaf']
random.seed(0); np.random.seed(0); torch.manual_seed(0)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

idx = json.load(open(f'{HERE}/cache/index.json'))
train = [x for x in idx if x[3] == 'train']
if a.wheat:
    W1 = '/media/kaua-matheus/HD3201/Data/Agroscope/Wheat/Wheat_1/Train'
    wf = sorted(os.path.join(r, f) for r, _, fs in os.walk(W1) for f in fs if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    random.Random(1).shuffle(wf)
    train += [(f, 'leaf', 'wheat_train', 'train') for f in wf[:600]]
random.shuffle(train)
nv = int(0.15 * len(train)); val, train = train[:nv], train[nv:]

post_train = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.6, 1.6)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                        T.ColorJitter(0.15, 0.15, 0.15, 0.02), T.ToTensor(), T.Normalize(MEAN, STD)])
post_eval = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])

class GateDS(Dataset):
    def __init__(s, items, train): s.items, s.train = items, train
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        path, label, group, _ = s.items[i]
        rng = random.Random(None if s.train else i)
        np.random.seed(rng.randrange(2**31))
        if label == 'not_leaf' and rng.random() < a.p_syn:
            im = synth_negative(rng)
        else:
            im = Image.open(path).convert('RGB')
            if max(im.size) > 768: im.thumbnail((768, 768))
        r = rng.random()
        if r < a.p_bad:
            kind = rng.choice(list(BAD)); im = apply(im, kind, rng.choice(BAD[kind])); deg = 'bad'
        elif r < a.p_bad + 0.3:
            kind = rng.choice(list(OK)); im = apply(im, kind, rng.choice(OK[kind])); deg = 'ok'
        else:
            deg = 'clean'
        if label == 'not_leaf': y = 2
        else: y = 1 if deg == 'bad' else 0
        return (post_train if s.train else post_eval)(im), y

def build(arch, n=3):
    if arch == 'mobilenet_v3_large':
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT); m.classifier[3] = nn.Linear(m.classifier[3].in_features, n)
    elif arch == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT); m.classifier[1] = nn.Linear(1280, n)
    return m

if __name__ == '__main__':
    # balanceia folha x não-folha por amostragem
    w = torch.tensor([1.0 / sum(1 for y in train if y[1] == x[1]) for x in train], dtype=torch.double)
    trl = DataLoader(GateDS(train, True), batch_size=48, sampler=torch.utils.data.WeightedRandomSampler(w, len(train)), num_workers=10, persistent_workers=True)
    val_l = DataLoader(GateDS(val, False), batch_size=64, num_workers=10)
    net = build(a.arch).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.05)
    steps = a.epochs * len(trl)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=steps, pct_start=0.1)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, state = -1, None; t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        for x, y in trl:
            with torch.autocast('cuda', dtype=torch.bfloat16): loss = crit(net(x.cuda()), y.cuda())
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        net.eval(); ok = n = 0
        with torch.no_grad():
            for x, y in val_l:
                ok += (net(x.cuda()).argmax(1).cpu() == y).sum().item(); n += len(y)
        print(f'ep{ep+1} val_acc={ok/n:.4f} t={time.time()-t0:.0f}s', flush=True)
        if ok / n >= best: best, state = ok / n, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    torch.save({'model_state_dict': state, 'class_names': CLASSES, 'model_info': {'architecture': a.arch, 'img_size': 224, 'version': 'gate-1.0',
                'val_accuracy': best, 'bad_levels': BAD}}, f'{HERE}/{a.out}')
    print('saved', a.out, best)
