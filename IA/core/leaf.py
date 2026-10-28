import cv2
import numpy as np
from PIL import Image


def Leaf_BBox(pil :Image.Image, work :int=512, min_frac :float=0.01, max_frac :float=0.6):
    """
        Localiza a folha numa foto com fundo claro (branco/cinza/azulado) e retorna
        a bbox (x0, y0, x1, y1) em coordenadas da imagem original.
        Retorna None quando nada é encontrado ou quando a imagem já é um close-up
        (folha ocupando mais que `max_frac` da imagem) — nesses casos não há o que recortar.
    """
    rgb = np.asarray(pil.convert("RGB"))
    h, w = rgb.shape[:2]
    s = work / max(h, w)
    small = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)

    # Fundo claro tem saturação baixa; folha (verde/amarela/marrom) tem saturação alta
    sat = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[..., 1]
    t, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (sat > max(t, 40)).astype(np.uint8) * 255

    # Limpeza de máscara
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    frac = cv2.contourArea(largest) / (mask.shape[0] * mask.shape[1])
    if frac < min_frac or frac > max_frac:
        return None

    x, y, bw, bh = cv2.boundingRect(largest)
    m = int(0.05 * max(bw, bh))  # margem de 5%
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(mask.shape[1], x + bw + m), min(mask.shape[0], y + bh + m)
    return int(x0 / s), int(y0 / s), int(x1 / s), int(y1 / s)


class LeafCrop:
    """
        Transform (PIL -> PIL) que recorta a folha antes do Resize.
        Sem isso, em fotos de folha inteira sobre fundo branco a folha vira uma faixa
        estreita no centro da imagem de 224px e o modelo só "enxerga" o centro.
        Em close-ups (folha ocupando a imagem toda) não faz nada.
    """
    def __call__(self, img :Image.Image) -> Image.Image:
        bbox = Leaf_BBox(img)
        return img if bbox is None else img.crop(bbox)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
