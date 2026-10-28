"""
Casos de uso de ponta a ponta com os modelos REAIS (./models).
Cada teste simula o que um usuário do aplicativo faria e verifica a resposta da API.
Ver tests/CASOS_DE_TESTE.md.
"""
import base64
import io
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from conftest import needs_models
from helpers import (blur, downscale, fake_document, fake_sky, fake_wall, fixture, jpeg_bytes, png_bytes,
                     scale_brightness)

pytestmark = [needs_models, pytest.mark.models]


def predict(client, culture, data, filename="foto.jpg", content_type="image/jpeg", **form):
    fields = {"culture": culture, **{k: str(v).lower() for k, v in form.items()}}
    return client.post("/predict", data=fields, files={"file": (filename, data, content_type)})


# ---------- informações ----------
def test_health(real_api):
    body = real_api.get("/health").json()
    assert set(body["cultures"]) == {"coffee", "soybean", "wheat"}
    assert body["gate"] is True


def test_cultures_mostra_tipo_de_modelo(real_api):
    r = real_api.get("/cultures").json()
    by = {c["culture"]: c for c in r["cultures"]}
    assert "LeafCrop" in by["coffee"]["preprocessing"]
    assert "legacy" in by["wheat"]["preprocessing"]
    assert "tomato" in r["unavailable"]


# ---------- diagnóstico ----------
@pytest.mark.parametrize("culture, image, expected", [
    ("coffee", "coffee_healthy.jpg", "Healthy"),
    ("coffee", "coffee_phoma.jpg", "Phoma"),
    ("coffee", "coffee_rust.jpg", "Rust"),
    ("soybean", "soy_rust.jpg", "Rust"),
    ("soybean", "soy_blight.jpg", "Bacterial_Blight"),
])
def test_diagnostico_correto(real_api, culture, image, expected):
    """Produtor envia foto de folha (doente ou saudável) e recebe o diagnóstico certo."""
    r = predict(real_api, culture, jpeg_bytes(fixture(image)))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok" and body["gate"]["verdict"] == "ok"
    assert body["diagnosis"]["predicted_class"] == expected
    assert body["diagnosis"]["is_healthy"] == (expected == "Healthy")
    assert body["diagnosis"]["leaf_bbox"] is not None  # folha recortada antes do modelo


def test_apelidos_em_portugues(real_api):
    assert predict(real_api, "Soja", jpeg_bytes(fixture("soy_rust.jpg"))).json()["culture_key"] == "soybean"
    assert predict(real_api, "Café", jpeg_bytes(fixture("coffee_rust.jpg"))).json()["culture_key"] == "coffee"


def test_campos_antigos_para_o_front(real_api):
    body = predict(real_api, "coffee", jpeg_bytes(fixture("coffee_rust.jpg"))).json()
    assert body["expert"]["predict"] == body["diagnosis"]["predicted_class"].upper()
    assert body["expert"]["predict_confidence"] == pytest.approx(body["diagnosis"]["confidence"] * 100, abs=0.01)


def test_modelo_antigo_trigo(real_api):
    r = predict(real_api, "trigo", jpeg_bytes(fixture("wheat_leaf.jpg")))
    assert r.status_code == 200, r.text
    d = r.json()["diagnosis"]
    assert d["model"]["legacy_preprocessing"] is True and d["leaf_bbox"] is None


def test_mapa_de_calor(real_api):
    exp = predict(real_api, "coffee", jpeg_bytes(fixture("coffee_phoma.jpg")), explain=True).json()["diagnosis"]["explanation"]
    assert Image.open(io.BytesIO(base64.b64decode(exp["heatmap_png_base64"]))).format == "PNG"
    assert 0 <= exp["peak_xy"][0] <= 1 and 0 <= exp["peak_xy"][1] <= 1


def test_foto_de_celular_rotacionada(real_api):
    """Celular grava a foto deitada + tag EXIF de rotação; o resultado deve ser o mesmo."""
    exif = Image.Exif()
    exif[274] = 6
    data = jpeg_bytes(fixture("coffee_rust.jpg").rotate(90, expand=True), exif=exif.tobytes())
    r = predict(real_api, "coffee", data)
    assert r.status_code == 200, r.text
    assert r.json()["diagnosis"]["predicted_class"] == "Rust"


def test_png_com_transparencia(real_api):
    data = png_bytes(fixture("coffee_healthy.jpg").convert("RGBA"))
    assert predict(real_api, "coffee", data, filename="folha.png", content_type="image/png").status_code == 200


# ---------- triagem: não é folha ----------
@pytest.mark.parametrize("make", [fake_document, fake_sky, fake_wall], ids=["documento", "ceu", "parede"])
def test_nao_folha_recusada(real_api, make):
    r = predict(real_api, "coffee", jpeg_bytes(make()))
    assert r.status_code == 422, r.text
    assert r.json()["reason"] == "not_leaf"


def test_nao_folha_nao_passa_nem_com_force(real_api):
    assert predict(real_api, "coffee", jpeg_bytes(fake_document()), force=True).status_code == 422


# ---------- triagem: qualidade ----------
def test_foto_tremida_recusada(real_api):
    r = predict(real_api, "coffee", jpeg_bytes(blur(fixture("coffee_rust.jpg"), 14)))
    assert r.status_code == 422, r.text
    assert r.json()["reason"] == "low_quality" and "foto" in r.json()["message"]


def test_foto_tremida_com_force(real_api):
    r = predict(real_api, "coffee", jpeg_bytes(blur(fixture("coffee_rust.jpg"), 14)), force=True)
    assert r.status_code == 200, r.text
    d = r.json()["diagnosis"]
    assert d["status"] == "review" and "qualidade da imagem" in d["review_reasons"]


def test_foto_escura_passa_com_aviso(real_api):
    """Foto escura NÃO derruba o acerto (medido), então passa — só com aviso."""
    r = predict(real_api, "coffee", jpeg_bytes(scale_brightness(fixture("coffee_rust.jpg"), 0.2)))
    assert r.status_code == 200, r.text
    assert "imagem escura" in r.json()["gate"]["warnings"]


@pytest.mark.parametrize("make", [
    lambda: jpeg_bytes(scale_brightness(fixture("coffee_rust.jpg"), 2.6)),
    lambda: jpeg_bytes(fixture("coffee_rust.jpg"), quality=3),
    lambda: jpeg_bytes(fixture("coffee_rust.jpg").resize((64, 64))),
], ids=["estourada", "compressao_extrema", "minuscula"])
def test_qualidade_inviavel_recusada(real_api, make):
    r = predict(real_api, "coffee", make())
    assert r.status_code == 422, r.text
    assert r.json()["reason"] == "low_quality"


@pytest.mark.parametrize("make", [
    lambda: jpeg_bytes(blur(fixture("coffee_rust.jpg"), 1.5), quality=60),
    lambda: jpeg_bytes(downscale(fixture("coffee_rust.jpg"), 400)),
], ids=["desfoque_leve", "resolucao_media"])
def test_defeito_leve_passa(real_api, make):
    """Defeitos leves não atrapalham o diagnóstico, então o usuário não é barrado."""
    assert predict(real_api, "coffee", make()).status_code == 200


# ---------- erros de entrada ----------
def test_cultura_sem_modelo(real_api):
    assert predict(real_api, "tomate", jpeg_bytes(fixture("coffee_rust.jpg"))).status_code == 400


def test_arquivo_que_nao_e_imagem(real_api):
    assert predict(real_api, "coffee", b"isto nao e uma imagem", filename="a.txt", content_type="text/plain").status_code == 400


def test_imagem_corrompida(real_api):
    assert predict(real_api, "coffee", jpeg_bytes(fixture("coffee_rust.jpg"))[:300]).status_code == 400


# ---------- carga ----------
def test_varios_usuarios_ao_mesmo_tempo(real_api):
    """8 requisições simultâneas (com Grad-CAM) respondem igual a uma requisição isolada."""
    data = jpeg_bytes(fixture("coffee_phoma.jpg"))
    alone = predict(real_api, "coffee", data, explain=True).json()["diagnosis"]
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda _: predict(real_api, "coffee", data, explain=True), range(8)))
    for r in results:
        assert r.status_code == 200
        d = r.json()["diagnosis"]
        assert d["predicted_class"] == alone["predicted_class"]
        assert d["confidence"] == pytest.approx(alone["confidence"], abs=1e-3)
        assert d["explanation"]["peak_xy"] == alone["explanation"]["peak_xy"]
