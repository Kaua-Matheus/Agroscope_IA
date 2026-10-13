import os
import shutil

import cv2
import numpy as np


def Lesion_Boxes(img, delta :int=7, thr_min :int=131, thr_max :int=140, min_area :float=0.004):
    """
        Gera caixas (x, y, w, h) das regiões lesionadas de uma folha (BGR) sem anotação manual.

        Usa o canal a* do espaço Lab (eixo verde <-> vermelho): tecido sadio é verde (a* baixo),
        lesões de ferrugem/phoma são laranja, marrom ou roxas (a* alto). O limiar é adaptativo
        (mediana de a* da imagem + delta) para lidar com fotos desbotadas/de baixo contraste.
    """
    h, w = img.shape[:2]

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    a = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)[..., 1]

    thr = float(np.clip(np.median(a) + delta, thr_min, thr_max))
    mask = (a > thr).astype(np.uint8) * 255

    # Limpeza de Máscara
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, w // 40) | 1,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # Cada componente conexo relevante vira uma caixa
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return [tuple(int(v) for v in s[:4]) for s in stats[1:] if s[4] >= min_area * h * w]


def Yolo_Lines(boxes, cls :int, w :int, h :int):
    """Converte caixas (x, y, w, h) em pixels para linhas YOLO: 'cls xc yc bw bh' normalizado."""
    return [
        f"{cls} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
        for x, y, bw, bh in boxes
    ]


def Export_Split(samples, split :str, out_dir :str, det_class_to_idx :dict, healthy :str="Healthy"):
    """
        Escreve um split no formato do Ultralytics:
            out_dir/images/<split>/*.jpg
            out_dir/labels/<split>/*.txt

        samples: lista de (caminho_da_imagem, nome_da_classe).
        - Healthy -> label vazio (imagem de fundo, ensina o modelo a NÃO detectar nada).
        - Doença  -> caixas geradas por Lesion_Boxes. Imagens doentes sem nenhuma caixa são
          descartadas, pois virariam "fundo" e ensinariam o contrário do que queremos.
    """
    img_dir = os.path.join(out_dir, "images", split)
    lbl_dir = os.path.join(out_dir, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    stats = {"written": 0, "boxes": 0, "skipped": 0}

    for i, (path, class_name) in enumerate(samples):
        img = cv2.imread(path)
        if img is None:
            stats["skipped"] += 1
            continue

        h, w = img.shape[:2]
        lines = []

        if class_name != healthy:
            boxes = Lesion_Boxes(img)
            if not boxes:
                stats["skipped"] += 1
                continue
            lines = Yolo_Lines(boxes, det_class_to_idx[class_name], w, h)

        # Nome único (as pastas originais repetem nomes, ex. "1 (101).jpg" em Phoma e Rust)
        name = f"{class_name}_{i:05d}"
        shutil.copy(path, os.path.join(img_dir, name + os.path.splitext(path)[1].lower()))
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
            f.write("\n".join(lines))

        stats["written"] += 1
        stats["boxes"] += len(lines)

    return stats


def Write_Data_Yaml(out_dir :str, names :list):
    """Gera o data.yaml que o Ultralytics usa para encontrar os splits e nomes das classes."""
    path = os.path.join(out_dir, "data.yaml")
    with open(path, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")
    return path


def Leaf_Region(xc :float, yc :float):
    """Descreve em texto em qual segmento (grade 3x3) da imagem está o centro normalizado da caixa."""
    rows = ["superior", "central", "inferior"]
    cols = ["esquerda", "centro", "direita"]
    r, c = min(int(yc * 3), 2), min(int(xc * 3), 2)
    return "centro" if (r, c) == (1, 1) else f"{rows[r]}-{cols[c]}"
