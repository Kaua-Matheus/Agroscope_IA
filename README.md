# Sobre o Projeto
Visamos entender quais doenças podem estar atacando uma planta utilizando Redes Neurais Convolucionais (visão computacional) para analisar detalhes sobre as folhas das plantas e então identificar padrões de doenças.

## IAs - ConvNext_Tiny Pre Train
Esse modelo é utilizado somente após a identificação da cultura específica, onde então a imagem é enviada a sua IA especialista, que irá trabalhar somente com informações sobre determinada cultura, isso melhora a predição e diminui o processamento.

## Objetivos Futuros
- Identificar os pontos de falha e possiveis melhorias do modelo atual.
- Encontrar mais datasets de diferentes culturas com uma boa quantidade e variedade de imagens.
    - Tomate

---

## Aplicação (API) — `Project/IA`

### Como funciona
```
Foto ──► Triagem ──► Especialista da cultura ──► Resposta
         (é folha?      (doença + certeza +
          qualidade ok?)  mapa de atenção)
```
1. **Triagem**: um modelo pequeno verifica se a foto é de uma folha e se tem qualidade para análise. Fotos que não são de folha, ou muito tremidas, estouradas, comprimidas ou pequenas, são **recusadas com uma mensagem** para o usuário tirar outra foto. Foto escura **não** é recusada (medimos que não atrapalha o diagnóstico).
2. **Especialista**: um modelo por cultura dá o diagnóstico. Os modelos novos (café, soja) recortam a folha antes de analisar; os antigos (trigo, tomate, milho) continuam com o preparo com que foram treinados.

### Como rodar
```bash
cd Project/IA
python app.py                                   # sobe em http://0.0.0.0:5000
python -m unittest discover -s tests -v         # 31 casos de uso
```
Modelos e opções ficam no `Project/.env`:

| Variável | Para quê |
|---|---|
| `COFFEE`, `SOYBEAN`, `WHEAT`, `TOMATO`, `CORN` | Caminho do modelo de cada cultura. Cultura sem modelo é ignorada (não derruba a API) |
| `GATE` | Modelo de triagem. Sem ele, só regras simples (resolução) são verificadas |
| `CORS_ORIGINS` | Endereços do front, separados por vírgula |
| `MAX_UPLOAD_MB` | Tamanho máximo da foto (padrão 8) |
| `API_HOST`, `API_PORT` | Endereço e porta (padrão `0.0.0.0:5000`) |
| `REVIEW_THRESHOLD`, `CONFIDENT_THRESHOLD` | Abaixo de 0,6 de certeza = "revisar"; acima de 0,8 = "confiante" |

### Rotas
| Rota | O que faz |
|---|---|
| `GET /health` | Status, culturas carregadas e se a triagem está ativa |
| `GET /cultures` | Classes, versão e tipo de cada modelo |
| `GET /modelinfo` | Formato antigo (compatibilidade) |
| `POST /predict` | Diagnóstico. Campos: `culture`, `file`, `explain` (mapa de calor, opcional) e `force` (analisar mesmo com qualidade ruim, opcional) |

`culture` aceita nome em inglês ou português, com ou sem acento: `coffee`/`café`, `soybean`/`soja`, `wheat`/`trigo`, `tomato`/`tomate`, `corn`/`milho`.

### Respostas do `/predict`

**200 — analisada**
```json
{
  "status": "ok",
  "culture": "coffee",
  "expert": { "predict": "PHOMA", "predict_confidence": 93.2 },
  "diagnosis": {
    "predicted_class": "Phoma", "confidence": 0.932,
    "probabilities": { "Healthy": 0.029, "Phoma": 0.932, "Rust": 0.039 },
    "is_healthy": false,
    "status": "confident",
    "review_reasons": [],
    "leaf_bbox": [274, 142, 692, 870],
    "explanation": null
  },
  "gate": { "verdict": "ok", "warnings": [], "scores": { "leaf": 0.99, "not_leaf": 0.01, "poor_quality": 0.01 } },
  "image": { "width": 1024, "height": 1024, "taken_at": "2025:11:29 11:52:18" },
  "metadata": { "api_version": "2.0", "latency_ms": 69 }
}
```
- `expert.predict` e `expert.predict_confidence` são os **mesmos campos de antes**; o front atual continua funcionando.
- `diagnosis.status`: `confident` (certeza ≥ 80%), `uncertain` (60–80%) ou `review` (abaixo de 60%, ou foto com qualidade ruim enviada com `force`). Sugestão: em `review`, pedir outra foto ou indicar avaliação de um agrônomo.
- `gate.warnings`: dicas que não bloqueiam (ex.: "imagem escura").
- Com `explain=true`, `diagnosis.explanation.heatmap_png_base64` traz o mapa de atenção pronto para exibir (`<img src="data:image/png;base64,...">`).

**422 — recusada pela triagem**
```json
{
  "status": "rejected",
  "reason": "low_quality",
  "message": "A imagem não tem qualidade suficiente para análise. Tire outra foto com boa luz, sem tremer e com a folha em foco.",
  "gate": { "verdict": "low_quality", "reasons": ["imagem sem qualidade para análise (desfoque, exposição ou compressão)"] }
}
```
`reason` é `not_leaf` ou `low_quality`. A `message` já pode ser mostrada ao usuário. Em `low_quality` o front pode oferecer "analisar mesmo assim" (reenviar com `force=true`); em `not_leaf` não.

**Outros erros:** `400` cultura inexistente ou arquivo vazio (a resposta lista as culturas disponíveis), `413` arquivo grande demais, `415` arquivo que não é imagem ou está corrompido.

### Mudanças em relação à versão anterior (out/2026)
- Recorte da folha antes do diagnóstico e remoção do corte central, que causava o "modelo olhando só o centro".
- Triagem de imagens (folha / não-folha / qualidade).
- **Café** adicionado. **Soja** trocada pelo modelo novo: as classes mudaram para `Bacterial_Blight`, `Healthy` e `Rust` (antes: `Healthy`, `Soybean_Rust`, `Target_Spot`). O modelo antigo continua em `models/Soybean/soybean.pth`.
- Cultura sem modelo não derruba mais a API; nomes de cultura em português aceitos.
- Foto girada pelo celular (EXIF) é corrigida automaticamente.
- Respostas de erro com o código HTTP correto (antes, falhas internas retornavam erro 500 sem explicação).
