"""
    Culturas disponíveis na API.

    Cada cultura é configurada por uma variável de ambiente com o caminho do checkpoint
    (COFFEE, SOYBEAN, WHEAT, TOMATO, CORN). Cultura sem variável ou com arquivo ausente
    é apenas ignorada (com aviso no log), em vez de derrubar a aplicação.
"""
import logging
import os
import unicodedata

import torch

from core.expert import ExpertModel

log = logging.getLogger("agroscope")

CULTURES = {
    # chave interna: (variável de ambiente, apelidos aceitos na requisição)
    "coffee":  ("COFFEE",  ["cafe", "café"]),
    "soybean": ("SOYBEAN", ["soja", "soy"]),
    "wheat":   ("WHEAT",   ["trigo"]),
    "tomato":  ("TOMATO",  ["tomate"]),
    "corn":    ("CORN",    ["milho", "maize"]),
}


def _norm(text :str) -> str:
    text = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(c for c in text if not unicodedata.combining(c))


ALIASES = {}
for key, (_, aliases) in CULTURES.items():
    for name in [key, *aliases]:
        ALIASES[_norm(name)] = key


class Registry:
    def __init__(self, device :torch.device, review_threshold :float=0.6, confident_threshold :float=0.8):
        self.experts :dict[str, ExpertModel] = {}
        self.skipped :dict[str, str] = {}
        for key, (env, _) in CULTURES.items():
            path = (os.getenv(env) or "").strip().strip("'\"")  # docker --env-file não remove aspas
            if not path:
                self.skipped[key] = f"variável {env} não definida"
                continue
            if not os.path.isfile(path):
                self.skipped[key] = f"arquivo não encontrado: {path}"
                log.warning("Cultura %s ignorada: %s", key, self.skipped[key])
                continue
            self.experts[key] = ExpertModel(key, path, device, review_threshold, confident_threshold)
            log.info("Cultura %s carregada (%s, v%s)", key, path, self.experts[key].version)

    def resolve(self, culture :str):
        """Nome enviado pelo front (ex.: 'Café', 'Soybean') -> chave interna, ou None."""
        key = ALIASES.get(_norm(culture or ""))
        return key if key in self.experts else None
