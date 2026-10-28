"""
Testes rápidos da API com modelos FALSOS (não precisam dos .pth reais; rodam em segundos).
Verificam as regras da API: CORS, validação de entrada, códigos HTTP, triagem e formato da resposta.
"""
import base64
import io

from PIL import Image

import app as app_module
from core.settings import Settings
from helpers import jpeg_bytes, png_bytes, solid_image


def _image_bytes(size=(256, 256), color=(120, 180, 90), fmt="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def _post(client, culture="Tomato", data=None, filename="leaf.png", content_type="image/png", **form):
    fields = {k: str(v).lower() for k, v in form.items()}
    if culture is not None:
        fields["culture"] = culture
    return client.post("/predict", data=fields, files={"file": (filename, _image_bytes() if data is None else data, content_type)})


# --- CORS ---------------------------------------------------------------

def test_cors_allows_known_dev_origin(fake_api):
    origin = "http://localhost:3000"
    response = fake_api.get("/modelinfo", headers={"Origin": origin})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin


def test_cors_does_not_reflect_arbitrary_origin(fake_api):
    response = fake_api.get("/modelinfo", headers={"Origin": "https://evil.example.com"})
    # A rota responde (não é um browser), mas sem o header que autorizaria o browser a ler a resposta.
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_rejects_arbitrary_origin(fake_api):
    response = fake_api.options("/predict", headers={
        "Origin": "https://evil.example.com",
        "Access-Control-Request-Method": "POST",
    })
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_accepts_known_origin(fake_api):
    origin = "http://localhost:5173"
    response = fake_api.options("/predict", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin


def test_cors_does_not_allow_credentials(fake_api):
    response = fake_api.get("/modelinfo", headers={"Origin": "http://localhost:3000"})
    assert "access-control-allow-credentials" not in response.headers


def test_cors_preflight_rejects_disallowed_header(fake_api):
    response = fake_api.options("/predict", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "x-custom-header",
    })
    # allow_headers é uma lista explícita (não "*"): header fora da lista é recusado.
    assert response.status_code == 400


def test_cors_origins_configurable_via_env(monkeypatch):
    """CORS_ORIGINS sobrescreve a lista padrão (usado em produção)."""
    monkeypatch.setenv("CORS_ORIGINS", "https://agroscope-nu.vercel.app, https://api.example.com")
    settings = Settings.from_env()
    assert settings.cors_origins == ["https://agroscope-nu.vercel.app", "https://api.example.com"]
    app = app_module.create_app(settings)
    cors = next(m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware")
    assert cors.kwargs["allow_origins"] == settings.cors_origins


def test_cors_production_origin_allowed(make_fake_api):
    client = make_fake_api(env={"CORS_ORIGINS": "https://agroscope-nu.vercel.app"})
    r = client.get("/health", headers={"Origin": "https://agroscope-nu.vercel.app"})
    assert r.headers.get("access-control-allow-origin") == "https://agroscope-nu.vercel.app"
    r = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert "access-control-allow-origin" not in r.headers


# --- Rotas de informação ---------------------------------------------------

def test_modelinfo_returns_expected_structure(fake_api):
    response = fake_api.get("/modelinfo")
    assert response.status_code == 200
    body = response.json()
    for culture in ("tomato", "wheat", "soybean", "coffee"):
        assert body["models"][culture]["class_names"] == ["Healthy", "Diseased"]
        assert body["models"][culture]["num_classes"] == 2


def test_health_lists_loaded_cultures(fake_api):
    body = fake_api.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["cultures"]) == {"coffee", "soybean", "tomato", "wheat"}
    assert body["gate"] is False  # sem GATE configurado -> só regras simples


def test_cultures_reports_unavailable_and_preprocessing(fake_api):
    body = fake_api.get("/cultures").json()
    assert "corn" in body["unavailable"]
    by = {c["culture"]: c for c in body["cultures"]}
    assert "LeafCrop" in by["coffee"]["preprocessing"]   # modelo v2
    assert "legacy" in by["wheat"]["preprocessing"]      # modelo v1


# --- /predict: sucesso e formato ------------------------------------------

def test_predict_accepts_valid_image(fake_api):
    response = _post(fake_api)
    assert response.status_code == 200
    body = response.json()
    assert body["culture"] == "Tomato"           # devolve como foi enviado (compatibilidade)
    assert body["culture_key"] == "tomato"
    assert body["expert"]["predict"] == "DISEASED"
    assert body["expert"]["predict_confidence"] > 0


def test_predict_response_fields(fake_api):
    body = _post(fake_api, culture="coffee").json()
    assert body["status"] == "ok"
    d = body["diagnosis"]
    assert set(d) >= {"predicted_class", "confidence", "probabilities", "ranking", "margin", "entropy",
                      "is_healthy", "status", "review_reasons", "leaf_bbox", "model", "explanation", "latency_ms"}
    assert abs(sum(d["probabilities"].values()) - 1) < 1e-3
    assert d["ranking"][0] == d["predicted_class"]
    assert body["expert"]["predict_confidence"] == round(d["confidence"] * 100, 2)
    assert body["gate"]["verdict"] == "ok" and body["gate"]["method"] == "rules"
    assert body["image"]["width"] == 256 and body["metadata"]["api_version"] == app_module.API_VERSION


def test_predict_culture_aliases(fake_api):
    for sent, key in (("Café", "coffee"), ("SOJA", "soybean"), ("trigo", "wheat"), ("Tomate", "tomato")):
        body = _post(fake_api, culture=sent).json()
        assert body["culture_key"] == key, sent


def test_predict_legacy_model_flag(fake_api):
    d = _post(fake_api, culture="wheat").json()["diagnosis"]
    assert d["model"]["legacy_preprocessing"] is True
    assert d["leaf_bbox"] is None


def test_predict_explain_returns_png(fake_api):
    exp = _post(fake_api, culture="coffee", explain=True).json()["diagnosis"]["explanation"]
    assert Image.open(io.BytesIO(base64.b64decode(exp["heatmap_png_base64"]))).format == "PNG"


def test_predict_png_with_alpha(fake_api):
    data = png_bytes(Image.new("RGBA", (256, 256), (100, 160, 80, 128)))
    assert _post(fake_api, data=data).status_code == 200


def test_predict_accepts_octet_stream(fake_api):
    """Alguns clientes mobile mandam a foto como application/octet-stream."""
    r = _post(fake_api, data=jpeg_bytes(solid_image()), filename="foto", content_type="application/octet-stream")
    assert r.status_code == 200


def test_predict_low_confidence_marked_for_review(make_fake_api):
    client = make_fake_api(logits=(0.0, 0.1))  # ~52% de certeza
    d = _post(client).json()["diagnosis"]
    assert d["status"] == "review" and "confiança baixa" in d["review_reasons"]


# --- /predict: validação de entrada ---------------------------------------

def test_predict_rejects_unsupported_culture(fake_api):
    response = _post(fake_api, culture="Corn")  # milho sem modelo configurado
    assert response.status_code == 400
    assert "coffee" in response.json()["detail"]["available"]


def test_predict_rejects_unknown_culture(fake_api):
    assert _post(fake_api, culture="banana").status_code == 400


def test_predict_rejects_non_image_content_type(fake_api):
    response = _post(fake_api, data=b"not an image", filename="leaf.txt", content_type="text/plain")
    assert response.status_code == 400


def test_predict_rejects_corrupted_image(fake_api):
    response = _post(fake_api, data=b"\x89PNGnot-a-real-png-body")
    assert response.status_code == 400


def test_predict_rejects_empty_file(fake_api):
    assert _post(fake_api, data=b"").status_code == 400


def test_predict_rejects_oversized_upload(fake_api):
    limit = fake_api.app.state.settings.max_upload_bytes
    response = _post(fake_api, data=b"0" * (limit + 1))
    assert response.status_code == 413


def test_predict_missing_culture_is_rejected(fake_api):
    assert _post(fake_api, culture=None).status_code == 422


def test_predict_missing_file_is_rejected(fake_api):
    assert fake_api.post("/predict", data={"culture": "coffee"}).status_code == 422


# --- /predict: triagem ------------------------------------------------------

def test_rules_gate_rejects_tiny_image(fake_api):
    """Sem modelo de triagem, ainda assim imagens minúsculas (< 128px) são recusadas."""
    r = _post(fake_api, data=_image_bytes(size=(32, 32)))
    assert r.status_code == 422
    assert r.json()["reason"] == "low_quality"


def test_gate_not_leaf_is_rejected(make_fake_api):
    client = make_fake_api(gate="not_leaf")
    r = _post(client, culture="coffee")
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "rejected" and body["reason"] == "not_leaf"
    assert body["message"] == app_module.REJECTION_MESSAGES["not_leaf"]
    assert "diagnosis" not in body and "expert" not in body


def test_gate_not_leaf_ignores_force(make_fake_api):
    client = make_fake_api(gate="not_leaf")
    assert _post(client, culture="coffee", force=True).status_code == 422


def test_gate_low_quality_is_rejected(make_fake_api):
    client = make_fake_api(gate="poor")
    r = _post(client, culture="coffee")
    assert r.status_code == 422 and r.json()["reason"] == "low_quality"


def test_gate_low_quality_with_force_is_analyzed_for_review(make_fake_api):
    client = make_fake_api(gate="poor")
    r = _post(client, culture="coffee", force=True)
    assert r.status_code == 200
    d = r.json()["diagnosis"]
    assert d["status"] == "review" and "qualidade da imagem" in d["review_reasons"]


def test_gate_model_ok(make_fake_api):
    client = make_fake_api(gate="ok")
    body = _post(client, culture="coffee").json()
    assert body["gate"]["method"] == "model" and body["gate"]["verdict"] == "ok"
    assert client.get("/health").json()["gate"] is True


def test_gate_missing_file_falls_back_to_rules(make_fake_api):
    client = make_fake_api(env={"GATE": "/nao/existe/gate.pth"})
    assert client.get("/health").json()["gate"] is False


# --- Configuração -----------------------------------------------------------

def test_settings_defaults(monkeypatch):
    for var in ("MAX_UPLOAD_BYTES", "MAX_UPLOAD_MB", "SERVER_PORT", "SERVER_HOST", "CORS_ORIGINS", "DEVICE"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env()
    assert s.max_upload_bytes == 8 * 1024 * 1024
    assert (s.host, s.port) == ("0.0.0.0", 5000)
    assert "http://localhost:5173" in s.cors_origins


def test_settings_upload_limit_in_mb(monkeypatch):
    monkeypatch.delenv("MAX_UPLOAD_BYTES", raising=False)
    monkeypatch.setenv("MAX_UPLOAD_MB", "2")
    assert Settings.from_env().max_upload_bytes == 2 * 1024 * 1024


def test_api_fails_without_any_model(monkeypatch):
    """Sem nenhum especialista configurado, a API não sobe (melhor falhar cedo do que responder errado)."""
    import pytest
    from starlette.testclient import TestClient
    for var in ("COFFEE", "SOYBEAN", "WHEAT", "TOMATO", "CORN", "GATE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DEVICE", "cpu")
    with pytest.raises(RuntimeError, match="Nenhum modelo"):
        with TestClient(app_module.create_app(Settings.from_env())):
            pass


def test_model_paths_with_quotes_are_accepted(make_fake_api, monkeypatch):
    """`docker run --env-file` não remove aspas do .env (COFFEE='./models/...'); a API precisa tolerar."""
    make_fake_api()  # cria os checkpoints falsos e as variáveis de ambiente
    import os
    for var in ("COFFEE", "SOYBEAN", "TOMATO", "WHEAT"):
        monkeypatch.setenv(var, f"'{os.environ[var]}'")
    from starlette.testclient import TestClient
    with TestClient(app_module.create_app(Settings.from_env())) as quoted:
        assert set(quoted.get("/health").json()["cultures"]) == {"coffee", "soybean", "tomato", "wheat"}
