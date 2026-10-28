"""
Para rodar a aplicação, você deve estar dentro da pasta /

    python app.py

Fluxo do /predict:
    foto -> leitura (EXIF, redução) -> triagem (é folha? qualidade ok?) -> especialista da cultura -> resposta

Configuração (.env):
    COFFEE, SOYBEAN, WHEAT, TOMATO, CORN   caminhos dos especialistas (cultura sem caminho é ignorada)
    GATE                                   caminho do modelo de triagem (opcional)
    CORS_ORIGINS                           origens separadas por vírgula
    MAX_UPLOAD_MB (8), API_HOST (0.0.0.0), API_PORT (5000)
    REVIEW_THRESHOLD (0.6), CONFIDENT_THRESHOLD (0.8), GATE_NOT_LEAF_THRESHOLD (0.5), GATE_POOR_THRESHOLD (0.5)
"""

import datetime as dt
import logging
import os
import time
from contextlib import asynccontextmanager

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from core.gate import Gate
from core.imaging import InvalidImage, Load_Image
from core.registry import Registry

API_VERSION = "2.0"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("agroscope")

if load_dotenv():
    log.info(".env carregado")
else:
    log.warning(".env não encontrado; usando apenas variáveis de ambiente")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "8")) * 1024 * 1024)

REJECTION_MESSAGES = {
    "not_leaf": "A imagem não parece ser de uma folha. Envie uma foto de uma folha da planta.",
    "low_quality": "A imagem não tem qualidade suficiente para análise. Tire outra foto com boa luz, sem tremer e com a folha em foco.",
}

STATE = {}


@asynccontextmanager
async def lifespan(app :FastAPI):
    """Carrega os modelos uma única vez, na subida da API."""
    log.info("Carregando modelos em %s...", DEVICE)
    STATE["registry"] = Registry(
        DEVICE,
        review_threshold=float(os.getenv("REVIEW_THRESHOLD", "0.6")),
        confident_threshold=float(os.getenv("CONFIDENT_THRESHOLD", "0.8")),
    )
    if not STATE["registry"].experts:
        raise RuntimeError("Nenhum modelo especialista foi carregado. Verifique COFFEE/SOYBEAN/WHEAT/TOMATO/CORN no .env")

    gate_path = os.getenv("GATE")
    if gate_path and not os.path.isfile(gate_path):
        log.warning("GATE não encontrado (%s); triagem só por regras simples", gate_path)
        gate_path = None
    STATE["gate"] = Gate(
        gate_path, DEVICE,
        not_leaf_threshold=float(os.getenv("GATE_NOT_LEAF_THRESHOLD", "0.5")),
        poor_threshold=float(os.getenv("GATE_POOR_THRESHOLD", "0.5")),
    )
    log.info("Culturas: %s | triagem: %s", list(STATE["registry"].experts), "modelo" if STATE["gate"].enabled else "regras")
    yield
    STATE.clear()


APP = FastAPI(title="Agroscope IA", version=API_VERSION, lifespan=lifespan)

origins = [o.strip() for o in os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
).split(",") if o.strip()]

APP.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@APP.middleware("http")
async def limit_upload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES:
                return JSONResponse(
                    {"detail": "Request body too large."}, status_code=413
                )
        except ValueError:
            pass

    return await call_next(request)


# Routes
## Debug / informações
@APP.get("/health")
def Health():
    registry, gate = STATE["registry"], STATE["gate"]
    return {
        "status": "ok",
        "api_version": API_VERSION,
        "device": DEVICE.type,
        "cultures": list(registry.experts),
        "gate": gate.enabled,
    }


@APP.get("/cultures")
def Cultures():
    registry = STATE["registry"]
    return {
        "cultures": [e.describe() for e in registry.experts.values()],
        "unavailable": registry.skipped,
        "gate": STATE["gate"].describe(),
    }


@APP.get("/modelinfo")
def ModelInfo():
    """Mantido por compatibilidade com o front antigo."""
    return {
        "models": {
            key: {
                "class_names": e.classes,
                "num_classes": len(e.classes),
                "model_info": e.info,
            }
            for key, e in STATE["registry"].experts.items()
        },
    }


## Predict
def _run_prediction(data :bytes, culture_key :str, explain :bool, force :bool):
    t0 = time.perf_counter()
    up = Load_Image(data)
    gate = STATE["gate"].check(up)

    if gate["verdict"] == "not_leaf" or (gate["verdict"] == "low_quality" and not force):
        return None, gate, up, t0

    diagnosis = STATE["registry"].experts[culture_key].predict(up, explain=explain)
    if gate["verdict"] == "low_quality":  # force=True: analisa mesmo assim, mas marca para revisão
        diagnosis["status"] = "review"
        diagnosis["review_reasons"].append("qualidade da imagem")
    return diagnosis, gate, up, t0


@APP.post("/predict")
async def Predict(
    culture: str = Form(...),
    file: UploadFile = File(...),
    explain: bool = Form(False),   # inclui o mapa de calor (Grad-CAM) em base64
    force: bool = Form(False),     # analisa mesmo com qualidade baixa (nunca se não for folha)
):
    registry = STATE["registry"]
    culture_key = registry.resolve(culture)
    if culture_key is None:
        raise HTTPException(400, {"message": f"Cultura '{culture}' não disponível.", "available": list(registry.experts)})

    if file.content_type and not (file.content_type.startswith("image/") or file.content_type == "application/octet-stream"):
        raise HTTPException(415, "Only images are acceptable.")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (máx. {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    if not data:
        raise HTTPException(400, "Arquivo vazio.")

    try:
        diagnosis, gate, up, t0 = await run_in_threadpool(_run_prediction, data, culture_key, explain, force)
    except InvalidImage as e:
        raise HTTPException(415, str(e))

    image_info = {"width": up.original_size[0], "height": up.original_size[1], "format": up.format, "taken_at": up.taken_at}
    metadata = {
        "api_version": API_VERSION,
        "processed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "device": DEVICE.type,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }

    if diagnosis is None:
        return JSONResponse(status_code=422, content={
            "status": "rejected",
            "reason": gate["verdict"],
            "message": REJECTION_MESSAGES[gate["verdict"]],
            "culture": culture_key,
            "gate": gate,
            "image": image_info,
            "metadata": metadata,
        })

    return {
        "status": "ok",
        "culture": culture_key,
        # Campos antigos, mantidos para o front atual continuar funcionando
        "expert": {
            "predict": diagnosis["predicted_class"].upper(),
            "predict_confidence": round(diagnosis["confidence"] * 100, 2),
        },
        "diagnosis": diagnosis,
        "gate": gate,
        "image": image_info,
        "metadata": metadata,
    }


if __name__ == '__main__':
    uvicorn.run(APP, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "5000")))
