"""
    Triagem da imagem antes do diagnóstico: é uma folha? A qualidade permite analisar?

    Com modelo (GATE no .env): MobileNetV3 com 3 saídas — leaf_ok / leaf_poor / not_leaf.
    Sem modelo: apenas as verificações simples (resolução, brilho e nitidez), sem checar se é folha.
"""
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from core.expert import Build_Model, MEAN, STD
from core.imaging import UploadedImage, Quality_Metrics

MIN_SIDE_PX = 128  # abaixo disso não há detalhe suficiente de lesão


class Gate:
    def __init__(self, path :str=None, device :torch.device=torch.device("cpu"),
                 not_leaf_threshold :float=0.5, poor_threshold :float=0.5):
        self.path = path
        self.device = device
        self.not_leaf_threshold = not_leaf_threshold
        self.poor_threshold = poor_threshold
        self.net = None
        self.info = {}
        self._lock = threading.Lock()
        if path:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.info = ck.get("model_info", {})
            self.classes = list(ck["class_names"])
            assert self.classes == ["leaf_ok", "leaf_poor", "not_leaf"], f"Classes inesperadas no GATE: {self.classes}"
            self.net = Build_Model(self.info.get("architecture", "mobilenet_v3_large"), 3)
            self.net.load_state_dict(ck["model_state_dict"])
            self.net.to(device).eval()
            size = int(self.info.get("img_size", 224))
            self.transform = transforms.Compose([
                transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
            ])
            with torch.no_grad():
                self.net(torch.zeros(1, 3, size, size, device=device))  # warm-up

    @property
    def enabled(self) -> bool:
        return self.net is not None

    def describe(self) -> dict:
        return {
            "enabled": self.enabled,
            "version": self.info.get("version"),
            "architecture": self.info.get("architecture"),
            "not_leaf_threshold": self.not_leaf_threshold,
            "poor_threshold": self.poor_threshold,
            "min_side_px": MIN_SIDE_PX,
        }

    def check(self, up :UploadedImage) -> dict:
        """
            verdict:
              ok          -> segue para o diagnóstico
              not_leaf    -> recusada (não parece uma folha)
              low_quality -> recusada (borrada, escura, estourada ou pequena demais)
        """
        t0 = time.perf_counter()
        metrics = Quality_Metrics(up.image)
        metrics["width"], metrics["height"] = up.original_size
        reasons, warnings = [], []

        if min(up.original_size) < MIN_SIDE_PX:
            reasons.append(f"resolução muito baixa (menor lado < {MIN_SIDE_PX}px)")

        scores = None
        if self.enabled:
            with self._lock, torch.no_grad():
                x = self.transform(up.image).unsqueeze(0).to(self.device)
                p = F.softmax(self.net(x), dim=1)[0].cpu().numpy()
            p_leaf = float(p[0] + p[1])
            p_poor = float(p[1] / (p[0] + p[1] + 1e-9))
            scores = {"leaf": round(p_leaf, 4), "not_leaf": round(float(p[2]), 4), "poor_quality": round(p_poor, 4)}
            if p[2] >= self.not_leaf_threshold:
                verdict = "not_leaf"
            elif p_poor >= self.poor_threshold or reasons:
                verdict = "low_quality"
                if p_poor >= self.poor_threshold:
                    reasons.append("imagem sem qualidade para análise (desfoque, exposição ou compressão)")
            else:
                verdict = "ok"
        else:
            verdict = "low_quality" if reasons else "ok"

        # Avisos leves (não bloqueiam): ajudam o usuário a tirar uma foto melhor
        if metrics["sharpness"] < 60:
            warnings.append("imagem possivelmente desfocada")
        if metrics["brightness"] < 50:
            warnings.append("imagem escura")
        elif metrics["brightness"] > 220:
            warnings.append("imagem muito clara")

        return {
            "verdict": verdict,
            "usable": verdict == "ok",
            "method": "model" if self.enabled else "rules",
            "scores": scores,
            "reasons": reasons,
            "warnings": warnings,
            "metrics": metrics,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
