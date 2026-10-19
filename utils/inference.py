"""
    Inferência para a aplicação final.

    Devolve um dicionário (pronto para JSON) com o diagnóstico final e as informações de cada modelo:
    probabilidades, confiança, margem, entropia, concordância entre modelos, qualidade da imagem,
    recorte da folha, Grad-CAM e latência.

    Uso:
        diag = Diagnoser(primary="Models/Saved/Expert/Coffee/coffee.pth")                 # 1 modelo
        diag = Diagnoser(primary="coffee.pth", secondary="coffee_vgg.pth", mode="cascade")  # 2 modelos
        result = diag.predict(Image.open("folha.jpg"), explain=True)
"""
import base64
import io
import time
import datetime as dt

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models, transforms

from utils.leaf import Leaf_BBox

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def Build_Model(arch :str, num_classes :int) -> nn.Module:
    if arch == "convnext_tiny":
        net = models.convnext_tiny(weights=None)
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, num_classes)
    elif arch == "vgg16_bn":
        net = models.vgg16_bn(weights=None)
        net.classifier[6] = nn.Linear(net.classifier[6].in_features, num_classes)
    elif arch == "efficientnet_b0":
        net = models.efficientnet_b0(weights=None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Arquitetura não suportada: {arch}")
    return net


def Cam_Layer(net :nn.Module, arch :str) -> nn.Module:
    """Última camada convolucional com dimensão espacial (alvo do Grad-CAM)."""
    if arch == "vgg16_bn":
        return net.features[42]  # último ReLU antes do MaxPool final
    return net.features[-1]


class LoadedModel:
    def __init__(self, path :str, device :torch.device):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        info = ck.get("model_info", {})
        self.name = path.rsplit("/", 1)[-1]
        self.arch = info.get("architecture", "convnext_tiny")
        self.img_size = info.get("img_size", 224)
        self.version = info.get("version")
        self.classes = ck["class_names"]
        self.net = Build_Model(self.arch, len(self.classes))
        self.net.load_state_dict(ck["model_state_dict"])
        self.net.to(device).eval()
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def probs(self, leaf :Image.Image, tta :bool=True) -> np.ndarray:
        x = self.transform(leaf).unsqueeze(0).to(self.device)
        with torch.no_grad():
            views = [x, torch.flip(x, dims=[3])] if tta else [x]
            p = sum(F.softmax(self.net(v), dim=1) for v in views) / len(views)
        return p[0].cpu().numpy()

    def gradcam(self, leaf :Image.Image, class_idx :int) -> np.ndarray:
        x = self.transform(leaf).unsqueeze(0).to(self.device)
        store = {}

        def hook(module, inp, out):
            store["a"] = out
            out.register_hook(lambda g: store.__setitem__("g", g))

        handle = Cam_Layer(self.net, self.arch).register_forward_hook(hook)
        try:
            with torch.enable_grad():
                out = self.net(x)
                self.net.zero_grad()
                out[0, class_idx].backward()
        finally:
            handle.remove()

        w = store["g"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * store["a"]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(leaf.height, leaf.width), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def _entropy(p :np.ndarray) -> float:
    """Entropia normalizada: 0 = certeza total, 1 = distribuição uniforme."""
    p = np.clip(p, 1e-12, 1)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def _js(p :np.ndarray, q :np.ndarray) -> float:
    """Divergência de Jensen-Shannon (base 2, entre 0 e 1) entre as saídas dos dois modelos."""
    p, q = np.clip(p, 1e-12, 1), np.clip(q, 1e-12, 1)
    m = (p + q) / 2
    return float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum())


def _model_report(m :LoadedModel, p :np.ndarray, ms :float) -> dict:
    order = np.argsort(-p)
    return {
        "model": m.name,
        "architecture": m.arch,
        "version": m.version,
        "predicted_class": m.classes[order[0]],
        "confidence": round(float(p[order[0]]), 4),
        "probabilities": {c: round(float(v), 4) for c, v in zip(m.classes, p)},
        "ranking": [m.classes[i] for i in order],
        "margin": round(float(p[order[0]] - p[order[1]]), 4),  # top1 - top2: quanto menor, mais "dividido"
        "entropy": round(_entropy(p), 4),
        "latency_ms": round(ms, 1),
    }


def _image_quality(img :Image.Image, bbox) -> dict:
    thumb = img.copy()
    thumb.thumbnail((512, 512))
    small = cv2.cvtColor(np.asarray(thumb.convert("RGB")), cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(small, cv2.CV_64F).var())
    brightness = float(small.mean())
    area = None
    if bbox is not None:
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (img.width * img.height)
    warnings = []
    if blur < 60:
        warnings.append("imagem possivelmente desfocada")
    if brightness < 50:
        warnings.append("imagem escura")
    elif brightness > 220:
        warnings.append("imagem estourada (muito clara)")
    return {
        "width": img.width,
        "height": img.height,
        "sharpness": round(blur, 1),       # variância do Laplaciano (512px); < 60 costuma ser borrado
        "brightness": round(brightness, 1),  # 0-255
        "leaf_detected": bbox is not None,
        "leaf_bbox": list(bbox) if bbox is not None else None,  # (x0, y0, x1, y1) na imagem original
        "leaf_area_fraction": round(area, 4) if area is not None else None,
        "warnings": warnings,
    }


def _heatmap_png(leaf :Image.Image, cam :np.ndarray, max_side :int=384) -> str:
    """Sobreposição do Grad-CAM na folha, como PNG em base64 (para exibir direto no front)."""
    rgb = np.asarray(leaf.convert("RGB"))
    color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    over = (0.55 * rgb + 0.45 * color).astype(np.uint8)
    out = Image.fromarray(over)
    out.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _cam_summary(cam :np.ndarray) -> dict:
    """Quanto da folha concentra a atenção do modelo e onde fica o ponto principal (coordenadas relativas 0-1)."""
    h, w = cam.shape
    y, x = np.unravel_index(cam.argmax(), cam.shape)
    return {
        "hot_area_fraction": round(float((cam > 0.5).mean()), 4),
        "peak_xy": [round(x / w, 3), round(y / h, 3)],
    }


class Diagnoser:
    """
        mode:
            "single"   -> só o primário.
            "average"  -> média das probabilidades dos dois modelos, sempre.
            "cascade"  -> roda o primário; só consulta o secundário quando a confiança < cascade_threshold.
    """
    def __init__(self, primary :str, secondary :str=None, mode :str="single", cascade_threshold :float=0.8,
                 review_threshold :float=0.6, device :str=None, work_side :int=1536):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.primary = LoadedModel(primary, self.device)
        self.secondary = LoadedModel(secondary, self.device) if secondary else None
        if self.secondary is not None:
            assert self.secondary.classes == self.primary.classes, "Os dois modelos precisam ter as mesmas classes"
        self.mode = mode if self.secondary is not None else "single"
        self.cascade_threshold = cascade_threshold
        self.review_threshold = review_threshold
        self.work_side = work_side

        # Warm-up: a 1ª inferência na GPU é lenta (~1s, inicialização do cuDNN); paga isso na subida da API
        dummy = Image.new("RGB", (256, 256), (90, 140, 60))
        for m in (self.primary, self.secondary):
            if m is not None:
                m.probs(dummy)
        self.primary.gradcam(dummy, 0)

    def predict(self, image :Image.Image, explain :bool=False, tta :bool=True) -> dict:
        t_start = time.perf_counter()
        exif = image.getexif()
        taken_at = exif.get(306)  # DateTime do EXIF, útil para cruzar com clima
        w0, h0 = image.size  # tamanho real, antes do draft
        original_size = (h0, w0) if exif.get(274) in (5, 6, 7, 8) else (w0, h0)  # EXIF com rotação de 90°
        if image.format == "JPEG":
            image.draft("RGB", (self.work_side // 2, self.work_side // 2))  # decodifica JPEG já reduzido (bem mais rápido)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((self.work_side, self.work_side))  # a folha recortada continua bem maior que a entrada do modelo
        scale = original_size[0] / image.width

        bbox = Leaf_BBox(image)
        leaf = image.crop(bbox) if bbox is not None else image

        t = time.perf_counter()
        p1 = self.primary.probs(leaf, tta)
        reports = [_model_report(self.primary, p1, (time.perf_counter() - t) * 1000)]

        final_p, source, agreement = p1, "primary", None
        use_secondary = self.mode == "average" or (self.mode == "cascade" and p1.max() < self.cascade_threshold)
        if use_secondary:
            t = time.perf_counter()
            p2 = self.secondary.probs(leaf, tta)
            reports.append(_model_report(self.secondary, p2, (time.perf_counter() - t) * 1000))
            final_p, source = (p1 + p2) / 2, "ensemble"
            agreement = {
                "agree": bool(p1.argmax() == p2.argmax()),
                "js_divergence": round(_js(p1, p2), 4),
            }

        classes = self.primary.classes
        idx = int(final_p.argmax())
        confidence = float(final_p[idx])
        quality = _image_quality(image, bbox)
        quality["width"], quality["height"] = original_size
        if bbox is not None:
            quality["leaf_bbox"] = [int(round(v * scale)) for v in bbox]  # coordenadas na imagem original

        # Status para a interface: o que mostrar ao usuário
        reasons = []
        if confidence < self.review_threshold:
            reasons.append("confiança baixa")
        if agreement is not None and not agreement["agree"]:
            reasons.append("modelos discordam")
        if quality["warnings"]:
            reasons.append("qualidade da imagem")
        status = "review" if reasons else ("confident" if confidence >= self.cascade_threshold else "uncertain")

        result = {
            "final": {
                "predicted_class": classes[idx],
                "confidence": round(confidence, 4),
                "probabilities": {c: round(float(v), 4) for c, v in zip(classes, final_p)},
                "is_healthy": classes[idx].lower() == "healthy",
                "decision_source": source,
                "status": status,           # confident | uncertain | review
                "review_reasons": reasons,
            },
            "models": reports,
            "agreement": agreement,
            "image": quality,
            "metadata": {
                "taken_at": taken_at,
                "processed_at": dt.datetime.now().isoformat(timespec="seconds"),
                "device": self.device.type,
                "mode": self.mode,
                "tta": tta,
                "total_latency_ms": None,
            },
        }

        if explain:
            cam = self.primary.gradcam(leaf, idx)
            result["explanation"] = {
                "method": "Grad-CAM",
                "model": self.primary.name,
                "target_class": classes[idx],
                **_cam_summary(cam),
                "heatmap_png_base64": _heatmap_png(leaf, cam),
            }

        result["metadata"]["total_latency_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        return result
