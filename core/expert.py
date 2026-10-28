"""
    Modelo especialista de uma cultura (ex.: café: Healthy / Phoma / Rust).

    Suporta os dois formatos de checkpoint do projeto:
      - v2.x (pipeline novo, Steps/lab.ipynb): recorta a folha (LeafCrop) e redimensiona para img_size x img_size.
      - v1.x (modelos antigos): Resize(256) + CenterCrop(224), sem recorte — exatamente como foram treinados.
"""
import base64
import io
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from core.imaging import UploadedImage
from core.leaf import Leaf_BBox

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def Build_Model(arch :str, num_classes :int) -> nn.Module:
    if arch == "convnext_tiny":
        net = models.convnext_tiny(weights=None)
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, num_classes)
    elif arch == "efficientnet_b0":
        net = models.efficientnet_b0(weights=None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        net = models.mobilenet_v3_large(weights=None)
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Arquitetura não suportada: {arch}")
    return net


def _entropy(p :np.ndarray) -> float:
    """Entropia normalizada: 0 = certeza total, 1 = distribuição uniforme."""
    p = np.clip(p, 1e-12, 1)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


class ExpertModel:
    def __init__(self, culture :str, path :str, device :torch.device, review_threshold :float=0.6,
                 confident_threshold :float=0.8):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        info = ck.get("model_info", {}) or {}
        self.culture = culture
        self.path = path
        self.classes = list(ck["class_names"])
        self.info = info
        self.version = str(info.get("version", "1.0"))
        self.legacy = not self.version.startswith("2")
        self.arch = info.get("architecture", "convnext_tiny")
        self.img_size = int(info.get("img_size", 224))
        self.device = device
        self.review_threshold = review_threshold
        self.confident_threshold = confident_threshold
        self._lock = threading.Lock()  # hooks do Grad-CAM não podem rodar em paralelo no mesmo modelo

        self.net = Build_Model(self.arch, len(self.classes))
        self.net.load_state_dict(ck["model_state_dict"])
        self.net.to(device).eval()

        if self.legacy:
            self.transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(), transforms.Normalize(MEAN, STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(), transforms.Normalize(MEAN, STD),
            ])

        # Warm-up: a 1ª inferência (e o 1º backward) na GPU são lentos; paga isso na subida da API
        dummy = Image.new("RGB", (256, 256), (90, 140, 60))
        self._probs(dummy, tta=False)
        self._gradcam(dummy, 0)

    # ---------- informações ----------
    def describe(self) -> dict:
        return {
            "culture": self.culture,
            "classes": self.classes,
            "version": self.version,
            "architecture": self.arch,
            "preprocessing": "legacy (Resize 256 + CenterCrop 224)" if self.legacy else
                             f"LeafCrop + Resize({self.img_size}x{self.img_size})",
            "metrics": {k: v for k, v in self.info.items() if "accuracy" in k},
            "training_date": self.info.get("training_date"),
        }

    # ---------- inferência ----------
    def _input_region(self, up :UploadedImage):
        """Região que vai para o modelo e a bbox da folha (em coordenadas da imagem original)."""
        if self.legacy:
            return up.image, None
        bbox = Leaf_BBox(up.image)
        if bbox is None:
            return up.image, None
        return up.image.crop(bbox), [int(round(v * up.scale)) for v in bbox]

    def _probs(self, img :Image.Image, tta :bool=True) -> np.ndarray:
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            views = [x, torch.flip(x, dims=[3])] if tta else [x]
            p = sum(F.softmax(self.net(v), dim=1) for v in views) / len(views)
        return p[0].cpu().numpy()

    def _gradcam(self, img :Image.Image, class_idx :int) -> np.ndarray:
        x = self.transform(img).unsqueeze(0).to(self.device)
        store = {}

        def hook(module, inp, out):
            store["a"] = out
            out.register_hook(lambda g: store.__setitem__("g", g))

        handle = self.net.features[-1].register_forward_hook(hook)
        try:
            with torch.enable_grad():
                out = self.net(x)
                self.net.zero_grad()
                out[0, class_idx].backward()
        finally:
            handle.remove()

        w = store["g"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * store["a"]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def _heatmap_png(self, img :Image.Image, cam :np.ndarray, max_side :int=384) -> str:
        """Sobreposição do Grad-CAM na região analisada, como PNG em base64."""
        x = self.transform(img)  # mesma região/tamanho que o modelo viu
        rgb = (np.clip(x.numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1) * 255).astype(np.uint8)
        color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        out = Image.fromarray((0.55 * rgb + 0.45 * color).astype(np.uint8))
        out.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def predict(self, up :UploadedImage, explain :bool=False, tta :bool=True) -> dict:
        t0 = time.perf_counter()
        region, leaf_bbox = self._input_region(up)
        with self._lock:
            p = self._probs(region, tta)
            order = np.argsort(-p)
            idx = int(order[0])
            confidence = float(p[idx])
            explanation = None
            if explain:
                cam = self._gradcam(region, idx)
                y, x = np.unravel_index(cam.argmax(), cam.shape)
                explanation = {
                    "method": "Grad-CAM",
                    "target_class": self.classes[idx],
                    "hot_area_fraction": round(float((cam > 0.5).mean()), 4),
                    "peak_xy": [round(x / cam.shape[1], 3), round(y / cam.shape[0], 3)],  # relativo à região analisada
                    "heatmap_png_base64": self._heatmap_png(region, cam),
                }

        reasons = []
        if confidence < self.review_threshold:
            reasons.append("confiança baixa")
        status = "review" if reasons else ("confident" if confidence >= self.confident_threshold else "uncertain")

        return {
            "predicted_class": self.classes[idx],
            "confidence": round(confidence, 4),
            "probabilities": {c: round(float(v), 4) for c, v in zip(self.classes, p)},
            "ranking": [self.classes[i] for i in order],
            "margin": round(float(p[order[0]] - p[order[1]]), 4) if len(p) > 1 else 1.0,
            "entropy": round(_entropy(p), 4),
            "is_healthy": self.classes[idx].lower() in ("healthy", "saudavel", "saudável"),
            "status": status,                 # confident | uncertain | review
            "review_reasons": reasons,
            "leaf_bbox": leaf_bbox,           # (x0, y0, x1, y1) na foto original; None = imagem inteira
            "model": {"version": self.version, "architecture": self.arch, "legacy_preprocessing": self.legacy},
            "explanation": explanation,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
