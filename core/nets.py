"""
    Construção das redes e leitura dos checkpoints (.pth).

    Tudo que cria ou carrega um modelo passa por aqui. Nos testes rápidos, `build_model` e
    `load_checkpoint` são trocados por versões falsas (ver tests/conftest.py), sem precisar dos pesos reais.
"""
import torch
import torch.nn as nn
from torchvision import models

# Normalização do ImageNet (usada no treino de todos os modelos do projeto)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(arch :str, num_classes :int) -> nn.Module:
    """Arquitetura vazia (sem pesos) com a última camada ajustada para `num_classes`."""
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


def load_checkpoint(path :str) -> dict:
    """
        Lê um checkpoint do projeto. Formato esperado:
            {"model_state_dict": ..., "class_names": [...], "model_info": {...}}
        `weights_only=False` porque os checkpoints guardam metadados Python; só carregue arquivos confiáveis.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model_state_dict" not in ck or "class_names" not in ck:
        raise ValueError(f"Checkpoint em formato inesperado: {path}")
    ck["model_info"] = ck.get("model_info") or {}
    return ck


def load_model(path :str, default_arch :str, device :torch.device):
    """Checkpoint -> (rede pronta em modo eval, checkpoint)."""
    ck = load_checkpoint(path)
    net = build_model(ck["model_info"].get("architecture", default_arch), len(ck["class_names"]))
    net.load_state_dict(ck["model_state_dict"])
    return net.to(device).eval(), ck
