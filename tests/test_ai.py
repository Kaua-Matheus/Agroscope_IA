"""
Testes de comportamento da IA (modelos REAIS). Ver tests/CASOS_DE_TESTE.md.

- Grupo `models`: rodam com as fotos de tests/fixtures (segundos).
- Grupo `dataset`: medem o acerto nos conjuntos de teste completos do HD de dados (minutos).
  Não rodam por padrão:  pytest -m dataset
"""
import os
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from conftest import ROOT, needs_models
from core.expert import ExpertModel
from core.imaging import Load_Image
from core.leaf import Leaf_BBox
from helpers import (blur, downscale, fake_document, fake_noise, fake_sky, fake_wall, fixture, jpeg_bytes,
                     recompress, scale_brightness, solid_image)

LEAVES = [  # (cultura, foto, classe correta)
    ("coffee", "coffee_healthy.jpg", "Healthy"),
    ("coffee", "coffee_phoma.jpg", "Phoma"),
    ("coffee", "coffee_rust.jpg", "Rust"),
    ("soybean", "soy_rust.jpg", "Rust"),
    ("soybean", "soy_blight.jpg", "Bacterial_Blight"),
]


def up(img :Image.Image):
    return Load_Image(jpeg_bytes(img, quality=95))


@pytest.fixture(scope="session")
def experts(real_api):
    return real_api.app.state.registry.experts


@pytest.fixture(scope="session")
def gate(real_api):
    return real_api.app.state.gate


# ======================================================================
# Modelos (fixtures locais)
# ======================================================================
@needs_models
@pytest.mark.models
class TestCheckpoints:
    def test_metadados_dos_modelos(self, experts):
        """Todo modelo carregado tem classes únicas; modelos v2 trazem arquitetura e tamanho de entrada."""
        for key, e in experts.items():
            assert len(e.classes) >= 2 and len(set(e.classes)) == len(e.classes), key
            assert any(c.lower() in ("healthy", "saudavel") for c in e.classes), key
            if not e.legacy:
                assert e.info.get("architecture") and e.info.get("img_size"), key

    def test_triagem_carregada(self, gate):
        assert gate.enabled and gate.classes == ["leaf_ok", "leaf_poor", "not_leaf"]


@needs_models
@pytest.mark.models
class TestEspecialistas:
    @pytest.mark.parametrize("culture, image, expected", LEAVES)
    def test_acerta_as_fotos_de_referencia(self, experts, culture, image, expected):
        d = experts[culture].predict(up(fixture(image)))
        assert d["predicted_class"] == expected
        assert d["confidence"] >= 0.6, "acerto com certeza baixa demais"

    def test_deterministico(self, experts):
        a = experts["coffee"].predict(up(fixture("coffee_phoma.jpg")))
        b = experts["coffee"].predict(up(fixture("coffee_phoma.jpg")))
        assert a["probabilities"] == b["probabilities"]

    @pytest.mark.parametrize("transform", [
        lambda im: ImageOps.mirror(im),
        lambda im: ImageOps.flip(im),
        lambda im: im.rotate(90, expand=True),
    ], ids=["espelho", "de_cabeca_para_baixo", "girada_90"])
    @pytest.mark.parametrize("culture, image, expected", LEAVES)
    def test_invariante_a_orientacao(self, experts, transform, culture, image, expected):
        """Folha não tem lado 'certo' na foto: girar ou espelhar não muda o diagnóstico."""
        assert experts[culture].predict(up(transform(fixture(image))))["predicted_class"] == expected

    @pytest.mark.parametrize("degrade", [
        lambda im: blur(im, 1.5),
        lambda im: recompress(im, 50),
        lambda im: scale_brightness(im, 0.6),
        lambda im: downscale(im, 512),
    ], ids=["desfoque_leve", "jpeg_50", "mais_escura", "resolucao_512"])
    @pytest.mark.parametrize("culture, image, expected", LEAVES)
    def test_robusto_a_defeitos_leves(self, experts, degrade, culture, image, expected):
        """Defeitos que a triagem deixa passar não podem mudar o diagnóstico."""
        assert experts[culture].predict(up(degrade(fixture(image))))["predicted_class"] == expected

    @pytest.mark.parametrize("culture, image, expected", LEAVES)
    def test_posicao_da_folha_na_foto_nao_importa(self, experts, culture, image, expected):
        """
            Regressão do problema original ("o modelo só olha o centro"): a mesma folha no canto
            esquerdo, no centro ou no canto direito de uma foto maior tem o mesmo diagnóstico.
        """
        img = fixture(image)
        leaf = img.crop(Leaf_BBox(img))
        leaf.thumbnail((500, 500))
        for x in (20, (1600 - leaf.width) // 2, 1600 - leaf.width - 20):
            canvas = Image.new("RGB", (1600, 1000), (232, 232, 236))
            canvas.paste(leaf, (x, (1000 - leaf.height) // 2))
            assert experts[culture].predict(up(canvas))["predicted_class"] == expected, f"x={x}"


@needs_models
@pytest.mark.models
class TestGradCAM:
    @pytest.mark.parametrize("culture, image", [(c, i) for c, i, _ in LEAVES])
    def test_atencao_fica_na_folha(self, experts, culture, image):
        """A maior parte do mapa de calor cai sobre a folha (e não no fundo/bordas)."""
        e = experts[culture]
        u = up(fixture(image))
        region = u.image.crop(Leaf_BBox(u.image))
        idx = e.classes.index(e.predict(u)["predicted_class"])
        cam = e._gradcam(region, idx)
        # máscara da folha na mesma resolução do mapa (saturação: fundo claro tem saturação baixa)
        hsv = np.asarray(region.resize(cam.shape[::-1]).convert("HSV"))
        leaf_mask = hsv[..., 1] > 60
        assert (cam * leaf_mask).sum() / cam.sum() >= 0.6

    def test_mapas_diferentes_para_fotos_diferentes(self, experts):
        """Regressão do bug antigo (hooks removidos -> todas as fotos com o mesmo mapa)."""
        e = experts["coffee"]
        cams = []
        for name in ("coffee_healthy.jpg", "coffee_phoma.jpg", "coffee_rust.jpg"):
            u = up(fixture(name))
            region = u.image.crop(Leaf_BBox(u.image))
            cams.append(e._gradcam(region.resize((224, 224)), 0))
        assert not np.allclose(cams[0], cams[1], atol=0.05) and not np.allclose(cams[1], cams[2], atol=0.05)


@needs_models
@pytest.mark.models
class TestTriagem:
    @pytest.mark.parametrize("image", ["coffee_healthy.jpg", "coffee_phoma.jpg", "coffee_rust.jpg",
                                       "soy_rust.jpg", "soy_blight.jpg", "wheat_leaf.jpg"])
    def test_folhas_aceitas(self, gate, image):
        assert gate.check(up(fixture(image)))["verdict"] == "ok"

    @pytest.mark.parametrize("make", [
        fake_document, fake_sky, fake_wall, fake_noise,
        lambda: solid_image((800, 600), (20, 20, 20)),
        lambda: solid_image((800, 600), (60, 140, 60)),  # verde liso: cor de folha, mas não é folha
    ], ids=["documento", "ceu", "parede", "ruido", "preto", "verde_liso"])
    def test_nao_folhas_recusadas(self, gate, make):
        assert gate.check(up(make()))["verdict"] == "not_leaf"

    def test_imagem_toda_branca_recusada(self, gate):
        """Quadro todo branco parece foto estourada: é recusado como qualidade baixa (ou não-folha) — nunca analisado."""
        assert gate.check(up(solid_image((800, 600), (240, 240, 240))))["verdict"] in ("not_leaf", "low_quality")

    @pytest.mark.parametrize("degrade", [
        lambda im: blur(im, 14),
        lambda im: scale_brightness(im, 2.6),
        lambda im: recompress(im, 3),
        lambda im: downscale(im, 40),
    ], ids=["desfoque_forte", "estourada", "jpeg_3", "40px"])
    def test_qualidade_inviavel(self, gate, degrade):
        assert gate.check(up(degrade(fixture("coffee_rust.jpg"))))["verdict"] == "low_quality"

    def test_foto_escura_e_aceita(self, gate):
        r = gate.check(up(scale_brightness(fixture("coffee_rust.jpg"), 0.2)))
        assert r["verdict"] == "ok" and "imagem escura" in r["warnings"]


# ======================================================================
# Conjuntos de teste completos (HD de dados) — pytest -m dataset
# ======================================================================
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/media/kaua-matheus/HD3201/Data/Agroscope"))
EXT = (".jpg", ".jpeg", ".png", ".webp")


def _labeled(folder :Path):
    return sorted((p, p.parent.name) for p in folder.rglob("*") if p.suffix.lower() in EXT)


def _need(path :Path):
    if not path.is_dir():
        pytest.skip(f"dados não encontrados: {path}")


@needs_models
@pytest.mark.dataset
@pytest.mark.parametrize("culture, folder, minimum", [
    ("coffee", "Coffee/Test_Coffee", 0.95),     # medido em out/2026: 97,8%
    ("soybean", "Soybean/Test_Soybean", 0.95),  # medido em out/2026: 99,0%
])
def test_acerto_no_teste_externo(experts, culture, folder, minimum):
    """Acerto mínimo de cada especialista nas fotos externas (nunca usadas no treino)."""
    _need(DATA_ROOT / folder)
    samples = _labeled(DATA_ROOT / folder)
    e = experts[culture]
    hits = [e.predict(Load_Image(p.read_bytes()))["predicted_class"] == y for p, y in samples]
    acc = float(np.mean(hits))
    print(f"\n{culture}: {acc:.4f} em {len(samples)} fotos")
    assert acc >= minimum


@needs_models
@pytest.mark.dataset
def test_acerto_tomate_v2(real_settings):
    """Tomate v2 (carregado à parte; medido em out/2026: 90% em 390 fotos — o modelo mais fraco, por isso o limite é mais baixo)."""
    path, folder = ROOT / "models/Tomato/tomato.pth", DATA_ROOT / "Tomato/Test_Tomato"
    _need(folder)
    if not path.is_file():
        pytest.skip("models/Tomato/tomato.pth ausente")
    e = ExpertModel("tomato", str(path), real_settings.device)
    samples = _labeled(folder)
    acc = float(np.mean([e.predict(Load_Image(p.read_bytes()))["predicted_class"] == y for p, y in samples]))
    print(f"\ntomato: {acc:.4f} em {len(samples)} fotos")
    assert acc >= 0.8


@needs_models
@pytest.mark.dataset
def test_triagem_aceita_folhas_externas(gate):
    """A triagem não pode barrar usuários com fotos boas: >= 98% das fotos externas aceitas."""
    folders = [DATA_ROOT / "Coffee/Test_Coffee", DATA_ROOT / "Soybean/Test_Soybean"]
    for f in folders:
        _need(f)
    samples = [p for f in folders for p, _ in _labeled(f)]
    ok = float(np.mean([gate.check(Load_Image(p.read_bytes()))["verdict"] == "ok" for p in samples]))
    print(f"\ntriagem: {ok:.4f} de {len(samples)} folhas aceitas")
    assert ok >= 0.98


@needs_models
@pytest.mark.dataset
def test_triagem_recusa_objetos(gate):
    """>= 97% de uma amostra de objetos/cenas (Caltech-101) recusada como não-folha."""
    neg = DATA_ROOT / "Negatives/101_ObjectCategories"
    _need(neg)
    plants = {"bonsai", "lotus", "sunflower", "water_lilly", "strawberry", "joshua_tree"}  # têm folhas na foto
    files = sorted(p for p in neg.rglob("*.jpg") if p.parent.name not in plants)
    random.Random(0).shuffle(files)
    rejected = float(np.mean([gate.check(Load_Image(p.read_bytes()))["verdict"] == "not_leaf" for p in files[:400]]))
    print(f"\ntriagem: {rejected:.4f} de 400 objetos recusados")
    assert rejected >= 0.97
