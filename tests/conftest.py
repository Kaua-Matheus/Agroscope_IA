"""
Infraestrutura comum dos testes.

- `fake_api`: API com modelos FALSOS (redes minúsculas com saída fixa). Não precisa dos .pth reais;
  roda em segundos e testa as regras da API (validação, CORS, códigos HTTP, triagem, formato da resposta).
- `real_api`: API com os modelos REAIS de ./models (pulado se os arquivos não existirem).

Cada API é criada com `create_app()`, então as duas convivem sem interferir uma na outra.
"""
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Os testes definem as variáveis que precisam; o .env local não deve interferir.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from core import nets  # noqa: E402
import app as app_module  # noqa: E402
from core.settings import Settings  # noqa: E402

CULTURE_VARS = ("COFFEE", "SOYBEAN", "WHEAT", "TOMATO", "CORN")


# ======================================================================
# Modelos falsos
# ======================================================================
class TinyNet(nn.Module):
    """Mesma "forma" das redes reais (tem .features para o Grad-CAM), mas minúscula e com saída fixa."""

    def __init__(self, num_classes :int):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, kernel_size=8, stride=8), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


_FAKE_CHECKPOINTS = {}
_real_build_model = nets.build_model
_real_load_checkpoint = nets.load_checkpoint


def _build_model(arch, num_classes):
    return TinyNet(num_classes) if arch == "fake" else _real_build_model(arch, num_classes)


def _load_checkpoint(path):
    if str(path) in _FAKE_CHECKPOINTS:
        return _FAKE_CHECKPOINTS[str(path)]
    return _real_load_checkpoint(path)


# Os desvios só valem para arquivos/arquiteturas falsos; os modelos reais carregam normalmente.
nets.build_model = _build_model
nets.load_checkpoint = _load_checkpoint


def make_fake_checkpoint(path :Path, classes, logits, version :str="2.0") -> str:
    """Cria um arquivo vazio (para o registry achar) e registra o checkpoint falso correspondente."""
    torch.manual_seed(0)
    net = TinyNet(len(classes))
    with torch.no_grad():
        net.classifier.weight.mul_(0.001)              # saída praticamente constante...
        net.classifier.bias.copy_(torch.tensor(logits, dtype=torch.float32))  # ...definida pelos logits
    path.write_bytes(b"")
    _FAKE_CHECKPOINTS[str(path)] = {
        "model_state_dict": net.state_dict(),
        "class_names": list(classes),
        "model_info": {"architecture": "fake", "version": version, "img_size": 64, "external_accuracy": 0.99},
    }
    return str(path)


GATE_CLASSES = ["leaf_ok", "leaf_poor", "not_leaf"]
GATE_LOGITS = {"ok": [5.0, 0.0, 0.0], "poor": [0.0, 5.0, 0.0], "not_leaf": [0.0, 0.0, 5.0]}


@pytest.fixture
def make_fake_api(tmp_path, monkeypatch):
    """
        Fábrica de API falsa. Por padrão: café, soja e tomate (v2) + trigo (v1), classes Healthy/Diseased,
        sem modelo de triagem (só regras) e limite de upload de 64 KB.
    """
    clients = []

    def factory(gate :str=None, env :dict=None, logits=(0.1, 0.9)):
        for var in CULTURE_VARS + ("GATE", "CORS_ORIGINS", "MAX_UPLOAD_MB"):
            monkeypatch.delenv(var, raising=False)
        classes = ["Healthy", "Diseased"]
        for culture, version in (("COFFEE", "2.0"), ("SOYBEAN", "2.0"), ("TOMATO", "2.0"), ("WHEAT", "1.0")):
            monkeypatch.setenv(culture, make_fake_checkpoint(tmp_path / f"{culture.lower()}.pth", classes, logits, version))
        if gate:
            monkeypatch.setenv("GATE", make_fake_checkpoint(tmp_path / "gate.pth", GATE_CLASSES, GATE_LOGITS[gate], "gate-test"))
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("MAX_UPLOAD_BYTES", str(64 * 1024))
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)

        client = TestClient(app_module.create_app(Settings.from_env()))
        clients.append(client)
        return client.__enter__()

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def fake_api(make_fake_api):
    return make_fake_api()


# ======================================================================
# Modelos reais
# ======================================================================
REAL_MODELS = {
    "COFFEE": "models/Coffee/coffee.pth",
    "SOYBEAN": "models/Soybean/soybean_v2.pth",
    "WHEAT": "models/Wheat/wheat.pth",
    "GATE": "models/Gate/gate.pth",
}
REAL_MODELS_OK = all((ROOT / p).is_file() for p in REAL_MODELS.values())
FIXTURES_OK = (ROOT / "tests" / "fixtures" / "coffee_rust.jpg").is_file()

needs_models = pytest.mark.skipif(not (REAL_MODELS_OK and FIXTURES_OK), reason="modelos reais (./models) ou fixtures ausentes")


@pytest.fixture(scope="session")
def real_settings():
    if not (REAL_MODELS_OK and FIXTURES_OK):
        pytest.skip("modelos reais (./models) ou fixtures ausentes")
    with pytest.MonkeyPatch.context() as mp:
        for var in CULTURE_VARS + ("GATE", "CORS_ORIGINS", "MAX_UPLOAD_BYTES", "MAX_UPLOAD_MB", "DEVICE"):
            mp.delenv(var, raising=False)
        for var, path in REAL_MODELS.items():
            mp.setenv(var, str(ROOT / path))
        settings = Settings.from_env()  # TOMATO/CORN ficam desligados nos testes de uso
    return settings


@pytest.fixture(scope="session")
def real_api(real_settings):
    """API real, carregada uma vez por sessão (os modelos levam alguns segundos para subir)."""
    with pytest.MonkeyPatch.context() as mp:
        for var in CULTURE_VARS:
            mp.delenv(var, raising=False)
        for var, path in REAL_MODELS.items():
            mp.setenv(var, str(ROOT / path))
        client = TestClient(app_module.create_app(real_settings))
        client.__enter__()
    yield client
    client.__exit__(None, None, None)
