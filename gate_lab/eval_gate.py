"""Avalia a triagem: folha/não-folha e qualidade. Compara modelo dedicado x heurísticas x certeza do especialista."""
import os, sys, json, time, numpy as np, torch, torch.nn.functional as F, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, '/home/kaua-matheus/Documents/Agroscope_IA/Steps')
from PIL import Image
from torchvision import transforms as T
from sklearn.metrics import roc_auc_score
from degrade import LEVELS, apply
from train_gate import build, CLASSES
from utils.leaf import Leaf_BBox
from utils.inference import LoadedModel

name = sys.argv[1] if len(sys.argv) > 1 else 'gate_mnv3.pth'
ck = torch.load(f'{HERE}/{name}', weights_only=False)
net = build(ck['model_info']['architecture']); net.load_state_dict(ck['model_state_dict']); net = net.cuda().eval()
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
expert = LoadedModel('/home/kaua-matheus/Documents/Agroscope_IA/Steps/Models/Saved/Expert/Coffee/coffee.pth', torch.device('cuda'))

def gate_probs(ims):
    with torch.no_grad():
        x = torch.stack([tf(i) for i in ims]).cuda()
        return F.softmax(net(x), 1).cpu().numpy()

def heur(im):
    small = im.copy(); small.thumbnail((512, 512)); a = np.asarray(small)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    plant = ((hsv[..., 0] >= 15) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 40) & (hsv[..., 2] >= 40)).mean()
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    return dict(plant_frac=float(plant), sharp=float(cv2.Laplacian(gray, cv2.CV_64F).var()), bright=float(gray.mean()))

idx = json.load(open(f'{HERE}/cache/index.json'))
test = [x for x in idx if x[3] == 'test']
main_leaf = ['coffee_test', 'soy_test', 'tomato_test', 'coffee_field', 'wheat_unseen']
main_neg = ['objects_unseen', 'background']
ambig = ['plants_not_leaf', 'wheat_root_grain']

# ---------- 1) folha x não-folha (imagens limpas) ----------
rows = {}
for g in main_leaf + main_neg + ambig:
    items = [x for x in test if x[2] == g]
    ims = [Image.open(x[0]).convert('RGB') for x in items]
    P = np.concatenate([gate_probs(ims[i:i + 64]) for i in range(0, len(ims), 64)])
    H = [heur(im) for im in ims]
    E = np.array([expert.probs(Leaf_BBox(im) and im.crop(Leaf_BBox(im)) or im, tta=False).max() for im in ims])
    rows[g] = dict(n=len(ims), p_not=P[:, 2], p_poor=P[:, 1], plant=np.array([h['plant_frac'] for h in H]),
                   bbox=np.array([Leaf_BBox(im) is not None for im in ims]), exp=E)

def rate(groups, f):
    v = np.concatenate([f(rows[g]) for g in groups]); return v.mean()

print('== Folha x não-folha (imagens limpas) ==')
print(f'{"grupo":18s} {"n":>5s} | {"modelo":>8s} | {"heur. cor":>9s} | {"heur. bbox":>10s} | {"especialista":>12s}')
print('   (para folhas: % aceitas como folha; para não-folhas: % rejeitadas)')
for g in main_leaf + main_neg + ambig:
    r = rows[g]; leaf = g in main_leaf
    m = (r['p_not'] < 0.5) if leaf else (r['p_not'] >= 0.5)
    hc = (r['plant'] >= 0.15) if leaf else (r['plant'] < 0.15)
    hb = r['bbox'] if leaf else ~r['bbox']
    ex = (r['exp'] >= 0.6) if leaf else (r['exp'] < 0.6)
    print(f'{g:18s} {r["n"]:5d} | {m.mean()*100:7.1f}% | {hc.mean()*100:8.1f}% | {hb.mean()*100:9.1f}% | {ex.mean()*100:11.1f}%')
y = np.concatenate([np.zeros(rows[g]['n']) for g in main_leaf] + [np.ones(rows[g]['n']) for g in main_neg])
for label, key, sign in [('modelo (p_not_leaf)', 'p_not', 1), ('heurística de cor', 'plant', -1), ('certeza do especialista', 'exp', -1)]:
    s = np.concatenate([sign * rows[g][key] for g in main_leaf + main_neg])
    print(f'AUROC {label}: {roc_auc_score(y, s):.4f}')

# ---------- 2) qualidade ----------
print('\n== Qualidade: % de folhas marcadas como "imagem ruim" ==')
print('   (níveis 0-2 deveriam passar; nível 3 deveria ser barrado; "noise" nunca foi visto no treino)')
qitems = [x for x in test if x[2] in main_leaf]
rng = np.random.default_rng(0); qitems = [qitems[i] for i in rng.choice(len(qitems), 400, replace=False)]
base = [Image.open(x[0]).convert('RGB') for x in qitems]
EXPECT_BAR = {'gauss_3', 'motion_2', 'motion_3', 'bright_1', 'bright_2', 'bright_3', 'lowres_3', 'jpeg_2', 'jpeg_3', 'noise_2', 'noise_3'}
EXPECT_SKIP = {'dark_3'}  # ambíguo (-4 pts na soja)
print(f'{"degradação":12s} {"esperado":>9s} | {"modelo: recusa total":>20s} | {"heurística":>10s}')
res = {}
for kind in LEVELS:
    for lvl in range(4):
        if lvl == 0 and kind != 'gauss': continue
        np.random.seed(2)
        ims = [apply(im, kind, lvl) for im in base]
        P = np.concatenate([gate_probs(ims[i:i + 64]) for i in range(0, len(ims), 64)])
        poor = P[:, 1] / (P[:, 0] + P[:, 1] + 1e-9) >= 0.5
        H = [heur(im) for im in ims]
        hbad = np.array([h['sharp'] < 60 or h['bright'] < 50 or h['bright'] > 220 for h in H])
        key = 'limpa' if lvl == 0 else f'{kind}_{lvl}'
        rej = poor | (P[:, 2] >= .5)
        exp = 'ambíguo' if key in EXPECT_SKIP else ('barrar' if key in EXPECT_BAR else 'passar')
        res[key] = (exp, rej.mean(), hbad.mean())
        print(f'{key:12s} {exp:>9s} | {rej.mean()*100:19.1f}% | {hbad.mean()*100:9.1f}%')
for who, i in [('modelo', 1), ('heurística', 2)]:
    seen_pass = [v[i] for k, v in res.items() if v[0] == 'passar' and not k.startswith('noise')]
    seen_bar = [v[i] for k, v in res.items() if v[0] == 'barrar' and not k.startswith('noise')]
    noise_bar = [v[i] for k, v in res.items() if v[0] == 'barrar' and k.startswith('noise')]
    noise_pass = [v[i] for k, v in res.items() if v[0] == 'passar' and k.startswith('noise')]
    print(f'RESUMO {who}: recusa indevida (deveria passar) {np.mean(seen_pass)*100:.1f}% | recusa correta (deveria barrar) {np.mean(seen_bar)*100:.1f}% '
          f'| ruído (nunca visto): barrar {np.mean(noise_bar)*100:.1f}%, passar indevidamente recusado {np.mean(noise_pass)*100:.1f}%')

# negativos sintéticos com outra semente (céu, papel, parede...)
import random
from synth import synth_negative
rr = random.Random(12345); np.random.seed(5)
syn = [synth_negative(rr) for _ in range(300)]
P = np.concatenate([gate_probs(syn[i:i + 64]) for i in range(0, 300, 64)])
print(f'\nNegativos sintéticos (300): {(P[:, 2] >= .5).mean()*100:.1f}% como não-folha | {(P[:, 2] < .5).mean()*100:.1f}% recusados como qualidade ou aceitos: aceitos={((P[:, 2] < .5) & (P[:, 1] / (P[:, 0] + P[:, 1]) < .5)).mean()*100:.1f}%')

# ---------- 3) custo ----------
x = torch.randn(1, 3, 224, 224)
for dev in ['cuda', 'cpu']:
    m = net.to(dev); xx = x.to(dev)
    with torch.no_grad():
        for _ in range(5): m(xx)
        if dev == 'cuda': torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(30): m(xx)
        if dev == 'cuda': torch.cuda.synchronize()
    print(f'latência {dev}: {(time.perf_counter() - t) / 30 * 1000:.1f} ms')
print('parâmetros:', round(sum(p.numel() for p in net.parameters()) / 1e6, 2), 'M | arquivo (fp32):', round(sum(p.numel() for p in net.parameters()) * 4 / 2**20, 1), 'MB')
