"""
API do Agroscope. Para rodar (dentro desta pasta):

    python app.py

Fluxo do /predict:
    foto -> leitura (EXIF, redução) -> triagem (é folha? qualidade ok?) -> especialista da cultura -> resposta

Configuração: variáveis de ambiente / .env (ver core/settings.py e README.md).
"""

import datetime as dt
import logging
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from core.gate import Gate
from core.imaging import InvalidImage, Load_Image
from core.registry import Registry
from core.settings import Settings

API_VERSION = "2.1"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("agroscope")

REJECTION_MESSAGES = {
    "not_leaf": "A imagem não parece ser de uma folha. Envie uma foto de uma folha da planta.",
    "low_quality": "A imagem não tem qualidade suficiente para análise. Tire outra foto com boa luz, sem tremer e com a folha em foco.",
}


def create_app(settings :Settings=None) -> FastAPI:
    """
        Cria uma instância da API. Cada instância tem seus próprios modelos e configuração,
        o que permite aos testes subir várias APIs isoladas (com modelos reais ou falsos).
    """
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app :FastAPI):
        """Carrega os modelos uma única vez, na subida da API."""
        log.info("Carregando modelos em %s...", settings.device)
        registry = Registry(settings.device, settings.review_threshold, settings.confident_threshold)
        if not registry.experts:
            raise RuntimeError(f"Nenhum modelo especialista foi carregado. Verifique COFFEE/SOYBEAN/WHEAT/TOMATO/CORN no .env: {registry.skipped}")

        gate_path = settings.gate_path
        if gate_path and not os.path.isfile(gate_path):
            log.warning("GATE não encontrado (%s); triagem só por regras simples", gate_path)
            gate_path = None
        gate = Gate(gate_path, settings.device, settings.gate_not_leaf_threshold, settings.gate_poor_threshold)

        app.state.registry, app.state.gate = registry, gate
        log.info("Culturas: %s | triagem: %s", list(registry.experts), "modelo" if gate.enabled else "regras")
        yield

    app = FastAPI(title="Agroscope IA", version=API_VERSION, lifespan=lifespan)
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,               # a API não usa cookies/sessão
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Accept"],
    )

    @app.middleware("http")
    async def limit_upload_size(request :Request, call_next):
        """Recusa uploads grandes pelo Content-Length, antes de ler o corpo inteiro."""
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > settings.max_upload_bytes:
            return JSONResponse({"detail": "Request body too large."}, status_code=413)
        return await call_next(request)

    # ---------- informações ----------
    @app.get("/health")
    def Health(request :Request):
        state = request.app.state
        return {
            "status": "ok",
            "api_version": API_VERSION,
            "device": settings.device.type,
            "cultures": list(state.registry.experts),
            "gate": state.gate.enabled,
        }

    @app.get("/cultures")
    def Cultures(request :Request):
        state = request.app.state
        return {
            "cultures": [e.describe() for e in state.registry.experts.values()],
            "unavailable": state.registry.skipped,
            "gate": state.gate.describe(),
        }

    @app.get("/modelinfo")
    def ModelInfo(request :Request):
        """Mantido por compatibilidade com o front antigo."""
        return {
            "models": {
                key: {"class_names": e.classes, "num_classes": len(e.classes), "model_info": e.info}
                for key, e in request.app.state.registry.experts.items()
            },
        }

    # ---------- diagnóstico ----------
    def run_prediction(state, data :bytes, culture_key :str, explain :bool, force :bool):
        up = Load_Image(data)
        gate = state.gate.check(up)
        if gate["verdict"] == "not_leaf" or (gate["verdict"] == "low_quality" and not force):
            return None, gate, up

        diagnosis = state.registry.experts[culture_key].predict(up, explain=explain)
        if gate["verdict"] == "low_quality":  # force=True: analisa mesmo assim, mas marca para revisão
            diagnosis["status"] = "review"
            diagnosis["review_reasons"].append("qualidade da imagem")
        return diagnosis, gate, up

    @app.post("/predict")
    async def Predict(
        request :Request,
        culture :str = Form(...),
        file :UploadFile = File(...),
        explain :bool = Form(False),   # inclui o mapa de calor (Grad-CAM) em base64
        force :bool = Form(False),     # analisa mesmo com qualidade baixa (nunca se não for folha)
    ):
        t0 = time.perf_counter()
        state = request.app.state
        culture_key = state.registry.resolve(culture)
        if culture_key is None:
            raise HTTPException(400, {"message": f"Cultura '{culture}' não disponível.", "available": list(state.registry.experts)})

        if file.content_type and not (file.content_type.startswith("image/") or file.content_type == "application/octet-stream"):
            raise HTTPException(400, "Only images are acceptable.")

        data = await file.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"File too large (máx. {settings.max_upload_bytes // 1024} KB).")
        if not data:
            raise HTTPException(400, "Arquivo vazio.")

        try:
            diagnosis, gate, up = await run_in_threadpool(run_prediction, state, data, culture_key, explain, force)
        except InvalidImage as e:
            raise HTTPException(400, str(e))

        common = {
            "culture": culture,             # como foi enviado (compatibilidade com o front antigo)
            "culture_key": culture_key,     # nome interno normalizado (coffee, soybean, ...)
            "gate": gate,
            "image": {"width": up.original_size[0], "height": up.original_size[1], "format": up.format, "taken_at": up.taken_at},
            "metadata": {
                "api_version": API_VERSION,
                "processed_at": dt.datetime.now().isoformat(timespec="seconds"),
                "device": settings.device.type,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
        }

        if diagnosis is None:
            return JSONResponse(status_code=422, content={
                "status": "rejected",
                "reason": gate["verdict"],
                "message": REJECTION_MESSAGES[gate["verdict"]],
                **common,
            })

        return {
            "status": "ok",
            # Campos antigos, mantidos para o front atual continuar funcionando
            "expert": {
                "predict": diagnosis["predicted_class"].upper(),
                "predict_confidence": round(diagnosis["confidence"] * 100, 2),
            },
            "diagnosis": diagnosis,
            **common,
        }

    return app


if load_dotenv():
    log.info(".env carregado")

APP = create_app()


if __name__ == "__main__":
    s = APP.state.settings
    uvicorn.run(APP, host=s.host, port=s.port)
