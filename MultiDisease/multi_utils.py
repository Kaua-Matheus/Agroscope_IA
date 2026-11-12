"""
Funções de apoio do laboratório multi-doença (MultiDisease/lab.ipynb).

- Leitura do BRACOL (rótulos múltiplos por folha + caixas das lesões por doença)
- Copy-paste de lesões reais para criar folhas com combinações de doenças
- Modelo leve (MobileNetV3) com uma saída por doença (multi-rótulo)
- Mapa de calor por doença (Grad-CAM) e métricas de localização
"""
import collections
import csv
import os
import random
import re

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import models, transforms

DISEASES = ["miner", "rust", "phoma", "cercospora"]
DISEASE_PT = {"miner": "Bicho-mineiro", "rust": "Ferrugem", "phoma": "Phoma", "cercospora": "Cercosporiose"}
DISEASE_COLOR = {"miner": (0, 200, 255), "rust": (255, 140, 0), "phoma": (220, 0, 220), "cercospora": (255, 230, 0)}
SYMPTOM_DIRS = {"miner": "2_miner", "rust": "3_rust", "phoma": "4_phoma", "cercospora": "5_cercospora"}

IMG_W, IMG_H = 512, 256          # resolução das folhas do BRACOL (e entrada do modelo)
BOX_SCALE = 512 / 2048           # caixas dos recortes estão na resolução original (2048 x 1024)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ======================================================================
# Dados
# ======================================================================
def Load_Bracol(root :str):
    """
        Retorna:
          leaves: {id: {"path", "labels": np.array(4), "severity"}}
          boxes:  {id: [(doença, x0, y0, x1, y1), ...]} em coordenadas 512x256, só as coerentes com o CSV
          crops:  {id: [(doença, caminho_do_recorte), ...]}
    """
    ds = os.path.join(root, "classification", "dataset")
    leaves = {}
    with open(os.path.join(ds, "dataset-full.csv")) as f:
        for row in csv.DictReader(f):
            leaves[row["id"]] = {
                "path": os.path.join(ds, "leaf", f"{row['id']}.jpg"),
                "labels": np.array([int(row[d]) for d in DISEASES], dtype=np.float32),
                "severity": int(row["severity"]),
            }

    boxes, crops = collections.defaultdict(list), collections.defaultdict(list)
    for split in ("train", "val", "test"):
        for d in DISEASES:
            folder = os.path.join(ds, "symptom", split, SYMPTOM_DIRS[d])
            for name in os.listdir(folder):
                m = re.match(r"(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.", name)
                if not m or m.group(1) not in leaves:
                    continue  # recortes de outro banco (ex.: "ferr001") ou de folhas fora do CSV
                lid = m.group(1)
                if leaves[lid]["labels"][DISEASES.index(d)] == 0:
                    continue  # caixa de uma doença que o CSV não marca: ruído de anotação
                x, y, w, h = (int(v) * BOX_SCALE for v in m.groups()[1:])
                boxes[lid].append((d, x, y, x + w, y + h))
                crops[lid].append((d, os.path.join(folder, name)))
    return leaves, dict(boxes), dict(crops)


def Combo(labels) -> str:
    names = [d for d, v in zip(DISEASES, labels) if v]
    return "+".join(names) if names else "healthy"


def Split(leaves :dict, boxes :dict, fractions=(0.6, 0.15, 0.25), seed :int=42):
    """
        Divisão estratificada pela combinação de doenças (e por ter caixas para todas as doenças),
        para treino, validação e teste terem as mesmas proporções. Combinações raras (< 4 folhas)
        vão inteiras para o treino.
    """
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for lid, leaf in leaves.items():
        present = {DISEASES[i] for i, v in enumerate(leaf["labels"]) if v}
        boxed = {b[0] for b in boxes.get(lid, [])}
        groups[(Combo(leaf["labels"]), present <= boxed)].append(lid)
    split = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        ids = sorted(groups[key], key=int)
        rng.shuffle(ids)
        if len(ids) < 4:
            split["train"] += ids
            continue
        n_val, n_test = round(len(ids) * fractions[1]), round(len(ids) * fractions[2])
        split["val"] += ids[:n_val]
        split["test"] += ids[n_val:n_val + n_test]
        split["train"] += ids[n_val + n_test:]
    return split


def Leaf_Mask(img :Image.Image) -> np.ndarray:
    """Máscara da folha (fundo claro tem saturação baixa)."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(np.asarray(img.convert("RGB")), (5, 5), 0), cv2.COLOR_RGB2HSV)
    mask = (hsv[..., 1] > 50).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask.astype(bool)
    return lab == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def Leaf_BBox_Background(img :Image.Image, work :int=512):
    """
        Caixa da folha em fotos com fundo claro, com limiar FIXO de saturação.
        O Leaf_BBox do laboratório principal usa limiar automático (Otsu), que sobe demais quando metade da
        folha é muito saturada e corta a metade verde-escura (8,7% das fotos de teste do café perdem mais de 25% da folha).
    """
    a = np.asarray(img.convert("RGB"))
    h, w = a.shape[:2]
    s = work / max(h, w)
    small = cv2.GaussianBlur(cv2.resize(a, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA), (5, 5), 0)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    # limiar FIXO de saturação: fundo e sombra são pouco saturados; a folha (inclusive a metade escura) não
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # ... e matiz de folha/lesão (vermelho-marrom-amarelo-verde, < 95 no OpenCV): exclui fundos azulados
    fg = (((sat > 40) & (hue <= 95)) | ((val < 60) & (sat > 20))).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        return None
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, bw, bh = stats[k, :4]
    if bw * bh > 0.9 * fg.size:
        return None  # "folha" ocupando a foto toda: já é um close-up
    m = int(0.03 * max(bw, bh))
    return (max(0, int((x - m) / s)), max(0, int((y - m) / s)), min(w, int((x + bw + m) / s)), min(h, int((y + bh + m) / s)))


def Valid_Boxes(bxs, leaf_mask):
    """Descarta caixas com o centro fora da folha (8,5% das caixas do BRACOL estão deslocadas)."""
    H, W = leaf_mask.shape
    out = []
    for b in bxs:
        cx, cy = int((b[1] + b[3]) / 2), int((b[2] + b[4]) / 2)
        if 0 <= cx < W and 0 <= cy < H and leaf_mask[cy, cx]:
            out.append(b)
    return out


def To_Leaf_Frame(img :Image.Image, leaf_bbox_fn=Leaf_BBox_Background) -> Image.Image:
    """
        Prepara uma foto de outra fonte no formato das folhas do BRACOL: folha deitada e
        ocupando a imagem 2:1. Usado no teste com as fotos de café do próprio projeto.
    """
    img = img.convert("RGB")
    if leaf_bbox_fn is not None:
        bbox = leaf_bbox_fn(img)
        if bbox is not None:
            img = img.crop(bbox)
    if img.height > img.width:
        img = img.rotate(90, expand=True)
    return img.resize((IMG_W, IMG_H), Image.BICUBIC)


# ======================================================================
# Copy-paste de lesões
# ======================================================================
class Lesion_Bank:
    """Recortes de lesões reais por doença, já na escala das folhas 512x256."""

    def __init__(self, crops :dict, leaf_ids):
        self.items = collections.defaultdict(list)
        for lid in leaf_ids:
            for d, path in crops.get(lid, []):
                im = Image.open(path).convert("RGB")
                # canal alfa = "não é fundo": recortes na borda da folha trazem pedaços do fundo claro
                hsv = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2HSV)
                background = (hsv[..., 1] < 45) & (hsv[..., 2] > 140)
                if background.mean() > 0.5:
                    continue  # recorte quase todo de fundo: não serve para colar
                alpha = Image.fromarray(np.where(background, 0, 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3))
                im.putalpha(alpha)
                w, h = max(6, round(im.width * BOX_SCALE)), max(6, round(im.height * BOX_SCALE))
                self.items[d].append(im.resize((w, h), Image.LANCZOS))

    def counts(self):
        return {d: len(self.items[d]) for d in DISEASES}

    def sample(self, disease :str, rng :random.Random):
        return rng.choice(self.items[disease])


def _feather_alpha(w :int, h :int, rng :random.Random) -> Image.Image:
    """Máscara elíptica com borda suave: cola a lesão sem deixar a "caixa" do recorte aparecer."""
    a = Image.new("L", (w, h), 0)
    from PIL import ImageDraw
    m = 0.1 + 0.08 * rng.random()
    ImageDraw.Draw(a).ellipse([w * m, h * m, w * (1 - m), h * (1 - m)], fill=255)
    return a.filter(ImageFilter.GaussianBlur(max(1, min(w, h) * 0.12)))


def Paste_Lesions(img :Image.Image, labels :np.ndarray, bank :Lesion_Bank, rng :random.Random,
                  disease :str=None, n_range=(1, 4)):
    """
        Cola de 1 a 4 lesões reais de uma doença em pontos aleatórios DENTRO da folha.
        Retorna (imagem, novos rótulos, caixas coladas).
    """
    if disease is None:
        absent = [d for i, d in enumerate(DISEASES) if labels[i] == 0 and bank.items[d]]
        if not absent:
            return img, labels, []
        # favorece as doenças com menos exemplos reais em combinação
        weights = [1.0 / max(1, len(bank.items[d])) ** 0.5 for d in absent]
        disease = rng.choices(absent, weights)[0]

    img = img.copy()
    mask = Leaf_Mask(img)
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        return img, labels, []
    pasted = []
    for _ in range(rng.randint(*n_range)):
        lesion = bank.sample(disease, rng)
        s = rng.uniform(0.8, 1.25)
        lesion = lesion.resize((max(6, int(lesion.width * s)), max(6, int(lesion.height * s))), Image.LANCZOS)
        if rng.random() < 0.5:
            lesion = lesion.transpose(Image.FLIP_LEFT_RIGHT)
        lesion = lesion.rotate(rng.choice([0, 90, 180, 270]), expand=True)
        w, h = lesion.size
        for _try in range(20):  # procura um ponto em que a lesão caiba inteira na folha
            i = rng.randrange(len(xs))
            x0, y0 = xs[i] - w // 2, ys[i] - h // 2
            if x0 < 0 or y0 < 0 or x0 + w > img.width or y0 + h > img.height:
                continue
            if mask[y0:y0 + h, x0:x0 + w].mean() < 0.95:
                continue
            # ajusta só o BRILHO da lesão ao da folha no ponto (sombra/luz diferentes entre fotos);
            # ajustar cada canal separadamente distorce a cor (lesões ficavam arroxeadas)
            target = np.asarray(img.crop((x0, y0, x0 + w, y0 + h))).astype(np.float32).mean()
            rgba = np.asarray(lesion)
            src, fg = rgba[..., :3].astype(np.float32), rgba[..., 3]
            ring = np.concatenate([src[0], src[-1], src[:, 0], src[:, -1]])
            ring_fg = np.concatenate([fg[0], fg[-1], fg[:, 0], fg[:, -1]]) > 127
            border = ring[ring_fg].mean() if ring_fg.any() else src[fg > 127].mean()
            gain = float(np.clip(target / max(border, 1.0), 0.7, 1.4))
            adj = np.clip(src * gain, 0, 255).astype(np.uint8)
            alpha = np.asarray(_feather_alpha(w, h, rng)).astype(np.float32) * (fg.astype(np.float32) / 255)
            img.paste(Image.fromarray(adj), (x0, y0), Image.fromarray(alpha.astype(np.uint8)))
            pasted.append((disease, x0, y0, x0 + w, y0 + h))
            break
    labels = labels.copy()
    if pasted:
        labels[DISEASES.index(disease)] = 1
    return img, labels, pasted


# ======================================================================
# Dataset
# ======================================================================
def Train_Transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.9, 1.1)),
        transforms.ColorJitter(0.25, 0.25, 0.2, 0.02),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


def Eval_Transform():
    return transforms.Compose([transforms.Resize((IMG_H, IMG_W)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class Leaf_Dataset(torch.utils.data.Dataset):
    """Folhas do BRACOL; no treino, com probabilidade `p_paste`, cola lesões de uma doença ausente."""

    def __init__(self, ids, leaves, transform, bank :Lesion_Bank=None, p_paste :float=0.0, seed :int=0):
        self.ids, self.leaves, self.transform = list(ids), leaves, transform
        self.bank, self.p_paste, self.seed = bank, p_paste, seed
        self.cache = {}

    def __len__(self):
        return len(self.ids)

    def image(self, lid):
        if lid not in self.cache:
            self.cache[lid] = Image.open(self.leaves[lid]["path"]).convert("RGB").resize((IMG_W, IMG_H))
        return self.cache[lid]

    def __getitem__(self, i):
        lid = self.ids[i]
        img, labels = self.image(lid), self.leaves[lid]["labels"]
        if self.bank is not None and self.p_paste > 0:
            # derivado do `random` global, que o DataLoader semeia em cada worker a partir do gerador do loader:
            # mesma semente -> mesmas colagens (resultados reproduzíveis)
            rng = random.Random(random.getrandbits(64))
            if rng.random() < self.p_paste:
                img, labels, _ = Paste_Lesions(img, labels, self.bank, rng)
        return self.transform(img), torch.from_numpy(labels)


# ======================================================================
# Modelo
# ======================================================================
def Build_Multi_Model(num_labels :int=len(DISEASES), pretrained :bool=True) -> nn.Module:
    """MobileNetV3-Large (~4,2 M parâmetros) com uma saída independente por doença."""
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    net = models.mobilenet_v3_large(weights=weights)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_labels)
    return net


@torch.no_grad()
def Predict_Probs(net, loader, device):
    net.eval()
    P, Y = [], []
    for x, y in loader:
        P.append(torch.sigmoid(net(x.to(device))).cpu())
        Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def Best_Thresholds(P :np.ndarray, Y :np.ndarray):
    """Limiar por doença que maximiza o F1 na validação."""
    from sklearn.metrics import f1_score
    out = []
    for k in range(Y.shape[1]):
        grid = np.linspace(0.05, 0.95, 91)
        f1 = [f1_score(Y[:, k], P[:, k] >= t, zero_division=0) for t in grid]
        out.append(float(grid[int(np.argmax(f1))]))
    return np.array(out)


# ======================================================================
# Mapa de calor por doença (Grad-CAM)
# ======================================================================
class Multi_GradCAM:
    """Um mapa por doença, calculado em uma única passada (um backward por doença)."""

    def __init__(self, net :nn.Module, layer :nn.Module=None):
        self.net = net
        self.layer = layer if layer is not None else net.features[-1]

    def __call__(self, x :torch.Tensor, diseases_idx=None):
        store = {}

        def hook(module, inp, out):
            store["a"] = out
            out.retain_grad()

        h = self.layer.register_forward_hook(hook)
        try:
            self.net.eval()
            with torch.enable_grad():
                logits = self.net(x)
                cams = {}
                idxs = range(logits.shape[1]) if diseases_idx is None else diseases_idx
                for k in idxs:
                    self.net.zero_grad()
                    if store["a"].grad is not None:
                        store["a"].grad = None
                    logits[0, k].backward(retain_graph=True)
                    w = store["a"].grad.mean(dim=(2, 3), keepdim=True)
                    cam = F.relu((w * store["a"]).sum(1, keepdim=True))
                    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
                    cam = cam.detach().cpu().numpy()
                    cams[k] = cam / (cam.max() + 1e-8)
        finally:
            h.remove()
        return torch.sigmoid(logits[0]).detach().cpu().numpy(), cams


def Disease_Map(img :Image.Image, cams :dict, present_idx, leaf_mask=None, threshold :float=0.45) -> Image.Image:
    """
        Pinta cada região da folha com a cor da doença de maior ativação ali
        (só doenças detectadas e só onde o mapa passa do limiar).
    """
    base = np.asarray(img.convert("RGB")).astype(np.float32)
    if not present_idx:
        return img
    stack = np.stack([cams[k] for k in present_idx])
    winner = stack.argmax(0)
    strong = stack.max(0) >= threshold
    if leaf_mask is not None:
        strong &= leaf_mask
    color = np.zeros_like(base)
    for j, k in enumerate(present_idx):
        color[(winner == j) & strong] = DISEASE_COLOR[DISEASES[k]]
    alpha = (strong * 0.5)[..., None]
    out = base * (1 - alpha) + color * alpha
    return Image.fromarray(out.astype(np.uint8))


def Pointing_Hit(cam :np.ndarray, boxes, leaf_mask=None, margin :int=6) -> bool:
    """O ponto mais quente do mapa (dentro da folha) cai em alguma das caixas (com margem)?"""
    c = cam.copy()
    if leaf_mask is not None:
        c[~leaf_mask] = 0
    y, x = np.unravel_index(c.argmax(), c.shape)
    return any(x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin for _, x0, y0, x1, y1 in boxes)


def Box_Area_Fraction(boxes, leaf_mask, margin :int=6) -> float:
    """Chance de acertar 'chutando' um ponto qualquer da folha: fração da folha coberta pelas caixas."""
    m = np.zeros_like(leaf_mask, dtype=bool)
    H, W = m.shape
    for _, x0, y0, x1, y1 in boxes:
        m[max(0, int(y0) - margin):min(H, int(y1) + margin), max(0, int(x0) - margin):min(W, int(x1) + margin)] = True
    return float((m & leaf_mask).sum() / max(1, leaf_mask.sum()))
