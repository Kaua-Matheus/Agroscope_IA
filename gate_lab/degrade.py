"""Degradações sintéticas de qualidade (aplicadas em imagens de até 768px)."""
import io, numpy as np, cv2
from PIL import Image

def gauss(im, s):
    return im if s == 0 else Image.fromarray(cv2.GaussianBlur(np.asarray(im), (0, 0), s))

def motion(im, L):
    if L == 0: return im
    k = np.zeros((L, L), np.float32); k[L // 2, :] = 1 / L
    M = cv2.getRotationMatrix2D((L / 2 - .5, L / 2 - .5), np.random.uniform(0, 180), 1)
    k = cv2.warpAffine(k, M, (L, L)); k /= k.sum()
    return Image.fromarray(cv2.filter2D(np.asarray(im), -1, k))

def exposure(im, f):
    return im if f == 1 else Image.fromarray(np.clip(np.asarray(im).astype(np.float32) * f, 0, 255).astype(np.uint8))

def lowres(im, side):
    if side >= max(im.size): return im
    s = side / max(im.size)
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR).resize(im.size, Image.BICUBIC)

def jpeg(im, q):
    if q >= 95: return im
    b = io.BytesIO(); im.save(b, 'JPEG', quality=q); b.seek(0); return Image.open(b).convert('RGB')

def noise(im, s):
    if s == 0: return im
    a = np.asarray(im).astype(np.float32) + np.random.normal(0, s, (im.height, im.width, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

# nível 0 = limpo; 1 = leve; 2 = moderado; 3 = severo
LEVELS = {
    'gauss':  (gauss,    [0, 2, 4, 8]),
    'motion': (motion,   [0, 15, 31, 61]),
    'dark':   (exposure, [1, .35, .2, .1]),
    'bright': (exposure, [1, 1.8, 2.6, 3.5]),
    'lowres': (lowres,   [768, 160, 80, 40]),
    'jpeg':   (jpeg,     [95, 15, 7, 3]),
    'noise':  (noise,    [0, 20, 40, 70]),
}

def apply(im, kind, level):
    f, vals = LEVELS[kind]
    return f(im, vals[level])
