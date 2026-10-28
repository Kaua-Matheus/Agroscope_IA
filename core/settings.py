"""
    Configuração da API, lida das variáveis de ambiente (.env) em um único lugar.
"""
import os
from dataclasses import dataclass, field

import torch

DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"


def _float(name :str, default :float) -> float:
    return float(os.getenv(name, default))


@dataclass
class Settings:
    device :torch.device
    cors_origins :list = field(default_factory=list)
    max_upload_bytes :int = 8 * 1024 * 1024
    host :str = "0.0.0.0"
    port :int = 5000
    gate_path :str = None
    review_threshold :float = 0.6
    confident_threshold :float = 0.8
    gate_not_leaf_threshold :float = 0.5
    gate_poor_threshold :float = 0.5

    @classmethod
    def from_env(cls) -> "Settings":
        device = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

        # MAX_UPLOAD_BYTES (bytes) tem prioridade sobre MAX_UPLOAD_MB
        if os.getenv("MAX_UPLOAD_BYTES"):
            max_bytes = int(os.getenv("MAX_UPLOAD_BYTES"))
        else:
            max_bytes = int(_float("MAX_UPLOAD_MB", 8) * 1024 * 1024)

        return cls(
            device=torch.device(device),
            cors_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()],
            max_upload_bytes=max_bytes,
            host=os.getenv("SERVER_HOST", "0.0.0.0"),
            port=int(os.getenv("SERVER_PORT", "5000")),
            gate_path=(os.getenv("GATE") or "").strip().strip("'\"") or None,
            review_threshold=_float("REVIEW_THRESHOLD", 0.6),
            confident_threshold=_float("CONFIDENT_THRESHOLD", 0.8),
            gate_not_leaf_threshold=_float("GATE_NOT_LEAF_THRESHOLD", 0.5),
            gate_poor_threshold=_float("GATE_POOR_THRESHOLD", 0.5),
        )
