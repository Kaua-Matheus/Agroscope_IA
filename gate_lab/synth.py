"""Negativos sintéticos: superfícies lisas, gradientes (céu), papel com texto, paredes, ruído colorido."""
import random, numpy as np
from PIL import Image, ImageDraw, ImageFilter

def synth_negative(rng :random.Random) -> Image.Image:
    W, H = rng.choice([(768, 576), (576, 768), (768, 768)])
    kind = rng.choice(['flat', 'gradient', 'document', 'bricks', 'tiles', 'blobs', 'stripes'])
    c = lambda: tuple(rng.randint(0, 255) for _ in range(3))
    if kind == 'flat':
        im = Image.new('RGB', (W, H), c())
    elif kind == 'gradient':
        a, b = np.array(c(), np.float32), np.array(c(), np.float32)
        t = np.linspace(0, 1, H if rng.random() < .5 else W)
        g = a[None] * (1 - t[:, None]) + b[None] * t[:, None]
        arr = np.repeat(g[:, None], W, 1) if len(t) == H else np.repeat(g[None], H, 0)
        im = Image.fromarray(arr.astype(np.uint8))
    elif kind == 'document':
        bg = rng.randint(200, 255); im = Image.new('RGB', (W, H), (bg, bg, rng.randint(190, bg)))
        d = ImageDraw.Draw(im)
        for i in range(rng.randint(10, 45)):
            y = 30 + i * rng.randint(14, 30)
            d.rectangle([30, y, 30 + rng.randint(80, W - 60), y + rng.randint(4, 12)], fill=(rng.randint(0, 80),) * 3)
    elif kind in ('bricks', 'tiles'):
        im = Image.new('RGB', (W, H), c()); d = ImageDraw.Draw(im); line = c(); s = rng.randint(30, 120)
        for row in range(0, H, s // (2 if kind == 'bricks' else 1)):
            d.line([(0, row), (W, row)], fill=line, width=rng.randint(2, 8))
            off = s // 2 if kind == 'bricks' and (row // (s // 2)) % 2 else 0
            for x in range(off, W, s):
                d.line([(x, row), (x, row + s)], fill=line, width=rng.randint(2, 8))
    elif kind == 'blobs':
        im = Image.new('RGB', (W, H), c()); d = ImageDraw.Draw(im)
        for _ in range(rng.randint(3, 25)):
            x, y, r = rng.randint(0, W), rng.randint(0, H), rng.randint(10, 200)
            d.ellipse([x - r, y - r, x + r, y + r], fill=c())
    else:
        im = Image.new('RGB', (W, H), c()); d = ImageDraw.Draw(im); w = rng.randint(5, 60)
        for x in range(0, W + H, w * 2):
            d.line([(x, 0), (x - H, H)], fill=c(), width=w)
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 3)))
    if rng.random() < 0.5:
        a = np.asarray(im).astype(np.float32) + np.random.normal(0, rng.uniform(2, 15), (H, W, 3))
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return im
