"""
    Leitura da imagem enviada e métricas simples de qualidade.

    A foto é aberta uma única vez aqui e o mesmo objeto é usado pela triagem e pelo especialista.
"""
import io
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImage(Exception):
    pass


@dataclass
class UploadedImage:
    image: Image.Image                  # RGB, já rotacionada pelo EXIF e reduzida para no máx. `work_side`
    original_size: tuple                # (largura, altura) reais da foto enviada
    scale: float                        # original / reduzida (para converter coordenadas)
    taken_at: Optional[str] = None      # DateTime do EXIF
    format: Optional[str] = None
    extra: dict = field(default_factory=dict)


def Load_Image(data :bytes, work_side :int=1536) -> UploadedImage:
    try:
        image = Image.open(io.BytesIO(data))
        exif = image.getexif()
        fmt = image.format
        w0, h0 = image.size
        if image.format == "JPEG":
            image.draft("RGB", (work_side // 2, work_side // 2))  # decodifica JPEG já reduzido (bem mais rápido)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise InvalidImage(f"Não foi possível ler a imagem: {e}") from e

    original_size = (h0, w0) if exif.get(274) in (5, 6, 7, 8) else (w0, h0)  # EXIF com rotação de 90°
    image.thumbnail((work_side, work_side))
    return UploadedImage(
        image=image,
        original_size=original_size,
        scale=original_size[0] / image.width,
        taken_at=exif.get(306),
        format=fmt,
    )


def Quality_Metrics(img :Image.Image) -> dict:
    """Nitidez (variância do Laplaciano) e brilho médio, medidos em uma versão de 512px."""
    thumb = img.copy()
    thumb.thumbnail((512, 512))
    gray = cv2.cvtColor(np.asarray(thumb.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return {
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "brightness": round(float(gray.mean()), 1),
    }
