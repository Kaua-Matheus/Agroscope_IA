# Agroscope IA — API de diagnóstico de folhas

API que recebe a foto de uma folha e responde qual doença foi encontrada (ou se a folha está saudável), com o grau de certeza e um mapa mostrando onde a IA "olhou".

Os modelos são treinados e avaliados no laboratório (`Steps/`); esta pasta contém só o necessário para servi-los.

## Como funciona

```
Foto ──► Triagem ──────────► Especialista da cultura ──► Resposta
         é folha?             doença + certeza
         qualidade ok?        + mapa de atenção
```

1. **Triagem**: um modelo pequeno (16 MB) verifica se a foto é de uma folha e se tem qualidade para análise. Fotos que não são de folha, ou muito tremidas, estouradas, comprimidas ou pequenas, são **recusadas com uma mensagem** para o usuário tirar outra foto. Foto escura **não** é recusada: medimos que ela não atrapalha o diagnóstico.
2. **Especialista**: há um modelo por cultura. Os modelos novos (v2) recortam a folha antes de analisar; os antigos (v1) continuam com o preparo com que foram treinados.

### Modelos em uso

| Cultura | Arquivo | Versão | Classes | Acerto no teste externo |
|---|---|---|---|---|
| Café | `models/Coffee/coffee.pth` | v2 | Healthy, Phoma, Rust | 97,8% (315 fotos) |
| Soja | `models/Soybean/soybean_v2.pth` | v2 | Bacterial_Blight, Healthy, Rust | 99,0% (306 fotos) |
| Tomate | `models/Tomato/tomato.pth` | v2 | Bacterial_Spot, Healthy, Leaf_Mold | 90,0% (390 fotos) |
| Trigo | `models/Wheat/wheat.pth` | v1 | Brown_Rust, Healthy, Yellow_Rust | não reavaliado |
| Milho | `models/Corn/corn.pth` | v1 | Healthy, Rust_Blight, Rust_Common | não reavaliado |
| Triagem | `models/Gate/gate.pth` | — | leaf_ok, leaf_poor, not_leaf | 99,8% das folhas aceitas; 100% dos objetos recusados |

Os arquivos `.pth` não vão para o Git (regras no `.gitignore` da raiz do repositório). As versões completas (com o estado do otimizador, para continuar o treino) ficam em `Steps/Models/Saved/Expert/`.

---

## Como rodar

```bash
cp .env.example .env       # ajuste os caminhos dos modelos, se preciso
pip install -r requirements.txt
python app.py              # sobe em SERVER_HOST:SERVER_PORT (padrão 0.0.0.0:5000)
```

O backend chama esta API em `FLASK_API_URL` (padrão `http://localhost:5000`, ver `packages/backend/.env.example`). Mudou a porta aqui, mude lá também.

Documentação interativa das rotas: `http://localhost:<porta>/docs`.

### Docker

```bash
docker build -t agroscope-api .
docker run --env-file .env -p 5000:5000 agroscope-api
```
- A imagem usa PyTorch só para CPU (~1,5 GB a menos que a versão com GPU).
- O `.env` não entra na imagem; é passado na hora de rodar com `--env-file`.
- A porta do `-p` deve ser a mesma do `SERVER_PORT` do `.env`.

### Configuração (`.env`)

Modelo pronto em [`.env.example`](.env.example).

| Variável | Para quê | Padrão |
|---|---|---|
| `COFFEE`, `SOYBEAN`, `WHEAT`, `TOMATO`, `CORN` | Caminho do modelo de cada cultura. Cultura sem modelo é ignorada (não derruba a API) | — |
| `GATE` | Modelo de triagem. Sem ele, só a regra de resolução mínima é verificada | — |
| `SERVER_HOST`, `SERVER_PORT` | Endereço e porta | `0.0.0.0`, `5000` |
| `CORS_ORIGINS` | Endereços do front, separados por vírgula (ex.: `https://agroscope-nu.vercel.app`) | `localhost:3000` e `:5173` |
| `MAX_UPLOAD_MB` (ou `MAX_UPLOAD_BYTES`) | Tamanho máximo da foto | 8 MB |
| `DEVICE` | Força `cpu` ou `cuda` | automático |
| `REVIEW_THRESHOLD`, `CONFIDENT_THRESHOLD` | Certeza abaixo da qual o resultado é "revisar" / a partir da qual é "confiante" | `0.6`, `0.8` |
| `GATE_NOT_LEAF_THRESHOLD`, `GATE_POOR_THRESHOLD` | Sensibilidade da triagem | `0.5`, `0.5` |

### Adicionar uma cultura

As culturas são declaradas **só aqui**, em `CULTURES` (`core/registry.py`). Backend, front e app leem a lista da IA:

```
IA: GET /cultures ──► backend: GET /crops (cache) ──► front/app: seletor de cultura
```

1. Coloque o modelo em `models/<Cultura>/` e acrescente uma linha em `CULTURES`: chave, variável de ambiente, nome exibido e apelidos. Ex.: `"rice": ("RICE", "Arroz", ["arroz"])`.
2. Defina a variável (`RICE=./models/Rice/rice.pth`) e reinicie a IA.
3. Para o diagnóstico ser salvo no histórico, o backend precisa ter as doenças da cultura na tabela `sickness` (coluna `crop` = `code`, ex.: `RICE`; `name` = nome da classe do modelo). Isso é feito com uma migration de seed, como as `Seed*Sicknesses` do backend.

Cultura sem variável ou sem arquivo simplesmente não aparece (`unavailable` em `/cultures`), então dá para desligar uma cultura sem mexer em código.

**Ordem de subida:** a IA só aceita conexões depois de carregar os modelos, e o backend não espera por ela. O backend tenta ler as culturas no boot e, se a IA ainda não respondeu, tenta de novo na próxima requisição (até lá, `/crops` e `/predict` respondem "O serviço de análise ainda está iniciando"). Depois, relê a cada `CROP_CATALOG_TTL_MS` (padrão 5 min) e, se a IA cair, mantém a última lista conhecida.

---

## Rotas

| Rota | O que faz |
|---|---|
| `GET /health` | Status, culturas carregadas e se a triagem está ativa |
| `GET /cultures` | Culturas carregadas (`code`, `label`, classes, versão, preparo e métricas) e as indisponíveis. **É daqui que o backend tira a lista de culturas** |
| `GET /modelinfo` | Formato antigo (compatibilidade) |
| `POST /predict` | Diagnóstico (formulário `multipart/form-data`) |

Campos do `/predict`:

| Campo | Obrigatório | Descrição |
|---|---|---|
| `culture` | sim | Em inglês ou português, com ou sem acento: `coffee`/`café`, `soybean`/`soja`, `wheat`/`trigo`, `tomato`/`tomate`, `corn`/`milho` |
| `file` | sim | A foto (JPEG, PNG...) |
| `explain` | não | `true` para receber o mapa de atenção |
| `force` | não | `true` para analisar mesmo com qualidade baixa (nunca libera foto que não é folha) |

### 200 — foto analisada
```json
{
  "status": "ok",
  "culture": "Café",
  "culture_key": "coffee",
  "expert": { "predict": "PHOMA", "predict_confidence": 93.2 },
  "diagnosis": {
    "predicted_class": "Phoma",
    "confidence": 0.932,
    "probabilities": { "Healthy": 0.029, "Phoma": 0.932, "Rust": 0.039 },
    "is_healthy": false,
    "status": "confident",
    "review_reasons": [],
    "leaf_bbox": [274, 142, 692, 870],
    "explanation": null
  },
  "gate": { "verdict": "ok", "warnings": [], "scores": { "leaf": 0.99, "not_leaf": 0.01, "poor_quality": 0.01 } },
  "image": { "width": 1024, "height": 1024, "taken_at": "2025:11:29 11:52:18" },
  "metadata": { "api_version": "2.1", "latency_ms": 69 }
}
```
- `culture` volta como foi enviado e `expert.*` mantém o formato antigo, então **o front atual continua funcionando**.
- `diagnosis.status`:
  - `confident`: certeza de 80% ou mais.
  - `uncertain`: entre 60% e 80%.
  - `review`: abaixo de 60%, ou foto de qualidade ruim enviada com `force`. Sugestão: pedir outra foto ou indicar avaliação de um agrônomo.
- `gate.warnings`: dicas que não bloqueiam (ex.: "imagem escura").
- Com `explain=true`, `diagnosis.explanation.heatmap_png_base64` traz o mapa pronto para exibir: `<img src="data:image/png;base64,...">`.

### 422 — foto recusada pela triagem
```json
{
  "status": "rejected",
  "reason": "low_quality",
  "message": "A imagem não tem qualidade suficiente para análise. Tire outra foto com boa luz, sem tremer e com a folha em foco.",
  "gate": { "verdict": "low_quality", "reasons": ["imagem sem qualidade para análise (desfoque, exposição ou compressão)"] }
}
```
- `reason` é `not_leaf` ou `low_quality`, e `message` já pode ser mostrada ao usuário.
- Em `low_quality`, o front pode oferecer "analisar mesmo assim" (reenviar com `force=true`). Em `not_leaf`, não.

### Outros erros
| Código | Quando |
|---|---|
| 400 | Cultura inexistente ou sem modelo (a resposta lista as disponíveis); arquivo vazio, que não é imagem ou corrompido |
| 413 | Arquivo maior que o limite |
| 422 (sem `status`) | Faltou o campo `culture` ou `file` |

---

## Testes

```bash
pip install -r requirements-dev.txt
pytest                  # 142 testes, ~20 s (API + casos de uso + comportamento da IA)
pytest -m dataset -s    # acerto nos conjuntos de teste completos (precisa do HD de dados, ~5 min)
```
- Os testes da API (`tests/test_api.py`) usam modelos falsos e rodam sem os `.pth`, em cerca de 2 s. São eles que rodam no CI (job *Tests (IA)* em `.github/workflows/tests.yml`).
- Os testes ficam aqui (e não em outro pacote) porque testam o código Python deste pacote diretamente (`core/`, `create_app()`), como o `test/` do backend. O fluxo completo app → backend → IA é coberto pelo `packages/e2e-web`.
- Sem a pasta `models/`, os testes com modelos reais são pulados.
- Lista de casos, resultados esperados e decisões: [`tests/CASOS_DE_TESTE.md`](tests/CASOS_DE_TESTE.md).

---

## Estrutura

| Caminho | O que é |
|---|---|
| `app.py` | API: rotas, CORS, limite de upload e fluxo do `/predict` |
| `core/settings.py` | Leitura da configuração (`.env`) |
| `core/registry.py` | Quais culturas estão disponíveis e os nomes aceitos |
| `core/expert.py` | Especialista de uma cultura: diagnóstico, certeza e mapa de atenção |
| `core/gate.py` | Triagem: é folha? qualidade ok? |
| `core/imaging.py` | Leitura da foto (EXIF, redução) e medidas de qualidade |
| `core/leaf.py` | Localização da folha na foto |
| `core/nets.py` | Construção das redes e leitura dos `.pth` |
| `tests/` | Testes e fotos de referência (`fixtures/`) |
| `models/` | Modelos (fora do Git) |

---

## Histórico

### v2.1 (out/2026)
- **Culturas dinâmicas:** `GET /cultures` passou a devolver `code` e `label` de cada cultura. O backend lê essa rota (`GET /crops`) e front/app montam o seletor a partir dela; o milho entrou assim, sem lista fixa no backend/front/app.
- **Configuração:** a porta passou a ser lida de `SERVER_HOST`/`SERVER_PORT` (antes ficava fixa em 5000). O padrão continua 5000, que é onde o backend procura a IA; um `SERVER_PORT` diferente exige ajustar o `FLASK_API_URL` do backend.
- **CORS** mais restrito: sem envio de credenciais e só com os cabeçalhos necessários. As origens podem ser configuradas por `CORS_ORIGINS`.
- **Erros:** imagem inválida ou corrompida volta a responder **400**, como na API antiga.
- **Resposta:** `culture` volta a ser devolvido como foi enviado; o nome normalizado está em `culture_key`.
- **Tomate:** passou a usar o modelo v2.
- **`.env` sem aspas:** o `docker run --env-file` não remove aspas, então nenhum modelo era encontrado. A API agora tolera aspas e mostra por que cada cultura não carregou.
- **Docker:** sem cache do pip (imagem de 3,7 para 3,0 GB), usuário sem privilégios de administrador e verificação de saúde automática.
- **Modelos enxugados:** tomate, trigo e milho deixaram de guardar o estado do otimizador (de 215–307 MB para 106 MB cada).
- **Testes:** unificados no pytest, de 31 para 144 casos, com documentação.
- **Código morto removido:** `imports.py` e `utils/model.py`.
- **Dependências:** removidas `torchaudio` e `matplotlib`, que não eram usadas; OpenCV trocado pela versão para servidor (sem interface gráfica).

### v2.0 (out/2026)
- Recorte da folha antes do diagnóstico e remoção do corte central, que causava o "modelo olhando só o centro".
- Triagem de imagens (folha / não-folha / qualidade).
- **Café** adicionado. **Soja** trocada pelo modelo novo: as classes mudaram para `Bacterial_Blight`, `Healthy` e `Rust` (antes: `Healthy`, `Soybean_Rust`, `Target_Spot`).
- Cultura sem modelo não derruba mais a API; nomes de cultura em português aceitos.
- Foto girada pelo celular (EXIF) é corrigida automaticamente.

## Próximos passos
- Verificar se a folha é da cultura escolhida. Hoje uma folha de café enviada como "tomate" recebe um diagnóstico de tomate.
- Retreinar trigo e milho no pipeline novo (v2). O milho (v1) já está disponível para análise.
- Mostrar no front os avisos da triagem (`gate.warnings`) e oferecer "analisar mesmo assim" (`force`) nas recusas por qualidade. Hoje o backend já repassa a mensagem das recusas (422) ao usuário.
