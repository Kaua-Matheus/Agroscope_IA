"""Geradores de imagens usados pelos testes."""
import io
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def fixture(name :str) -> Image.Image:
    """Fotos reais (tirados dos conjuntos de teste externos, nunca usados no treino)."""
    return Image.open(os.path.join(FIXTURES, name)).convert("RGB")


def jpeg_bytes(img :Image.Image, quality :int=90, exif=None) -> bytes:
    buf = io.BytesIO()
    kwargs = {"quality": quality}
    if exif is not None:
        kwargs["exif"] = exif
    img.convert("RGB").save(buf, "JPEG", **kwargs)
    return buf.getvalue()


def png_bytes(img :Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def solid_image(size=(256, 256), color=(120, 180, 90)) -> Image.Image:
    return Image.new("RGB", size, color)


# ---------- coisas que não são folha ----------
def fake_document() -> Image.Image:
    """Foto de um papel com texto (usuário enviou a foto errada)."""
    img = Image.new("RGB", (900, 1200), (245, 245, 240))
    d = ImageDraw.Draw(img)
    for i in range(40):
        y = 60 + i * 27
        d.rectangle([60, y, 60 + (i * 97) % 700 + 100, y + 10], fill=(40, 40, 40))
    return img


def fake_sky() -> Image.Image:
    """Céu azul (foto tirada sem querer para cima)."""
    a = np.zeros((900, 1200, 3), np.uint8)
    a[..., 0] = np.linspace(90, 170, 900)[:, None]
    a[..., 1] = np.linspace(140, 200, 900)[:, None]
    a[..., 2] = 235
    return Image.fromarray(a)


def fake_wall() -> Image.Image:
    """Parede de tijolos."""
    img = Image.new("RGB", (1200, 900), (150, 60, 40))
    d = ImageDraw.Draw(img)
    for row in range(0, 900, 60):
        off = 60 if (row // 60) % 2 else 0
        d.line([(0, row), (1200, row)], fill=(200, 200, 190), width=6)
        for x in range(off, 1200, 120):
            d.line([(x, row), (x, row + 60)], fill=(200, 200, 190), width=6)
    return img.filter(ImageFilter.GaussianBlur(1))


def fake_noise(seed :int=0) -> Image.Image:
    """Ruído colorido (câmera tampada / sensor com defeito)."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (768, 768, 3), dtype=np.uint8))


# ---------- defeitos de imagem ----------
def blur(img, sigma :float):
    return img.filter(ImageFilter.GaussianBlur(sigma))


def scale_brightness(img, factor :float):
    a = np.asarray(img).astype(np.float32) * factor
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def recompress(img, quality :int):
    return Image.open(io.BytesIO(jpeg_bytes(img, quality))).convert("RGB")


def downscale(img, side :int):
    small = img.copy()
    small.thumbnail((side, side))
    return small.resize(img.size, Image.BICUBIC)
