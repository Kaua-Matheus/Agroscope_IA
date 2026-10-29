# Casos de Teste — Agroscope IA

Este documento lista o que é testado na API e nos modelos de IA, o resultado esperado de cada caso e onde ele está no código.

## Como rodar

```bash
pip install -r requirements-dev.txt

pytest                 # padrão: API + casos de uso + comportamento da IA (~20 s, 142 testes)
pytest tests/test_api.py   # só a API com modelos falsos (~2 s; não precisa dos .pth)
pytest -m dataset -s   # acerto nos conjuntos de teste completos do HD de dados (~5 min)
```

| Arquivo | O que testa | Precisa de | Quando rodar |
|---|---|---|---|
| `test_api.py` | Regras da API, com **modelos falsos** | nada | Sempre (inclusive em CI) |
| `test_use_cases.py` | Fluxos completos do usuário, com **modelos reais** | `models/` | Antes de publicar a API |
| `test_ai.py` (padrão) | Comportamento dos modelos com fotos de referência | `models/` | Ao trocar um modelo |
| `test_ai.py -m dataset` | Acerto mínimo nos conjuntos de teste completos | `models/` + HD de dados | Ao trocar um modelo |

Sem a pasta `models/`, os testes que precisam dos modelos reais são **pulados** (não falham).

As fotos de referência ficam em `tests/fixtures/`. Foram tiradas dos conjuntos de **teste externo** (fotos nunca usadas no treino):

| Foto | Cultura | Classe |
|---|---|---|
| `coffee_healthy.jpg` | Café | Healthy |
| `coffee_phoma.jpg` | Café | Phoma |
| `coffee_rust.jpg` | Café | Rust |
| `soy_rust.jpg` | Soja | Rust |
| `soy_blight.jpg` | Soja | Bacterial_Blight |
| `wheat_leaf.jpg` | Trigo | LeafBlight (classe que o modelo antigo não tem; usada só para testar o funcionamento) |

---

## 1. Testes de API (`test_api.py`)

Usam redes minúsculas no lugar dos modelos reais, com resposta fixa (ex.: sempre "Diseased"). Assim testam só as **regras da API**, rápido e sem depender dos pesos.

### 1.1 CORS (quem pode chamar a API pelo navegador)
| ID | Caso | Esperado |
|---|---|---|
| API-01 | Front local (`localhost:3000`) chama a API | Liberado (`access-control-allow-origin` igual à origem) |
| API-02 | Site desconhecido chama a API | Sem o cabeçalho que libera a leitura pelo navegador |
| API-03 | Pré-requisição (preflight) de site desconhecido | 400 |
| API-04 | Pré-requisição do front local (`localhost:5173`) | 200 |
| API-05 | Qualquer chamada | Nunca envia `access-control-allow-credentials` (a API não usa cookies) |
| API-06 | Pré-requisição pedindo cabeçalho fora da lista (`x-custom-header`) | 400 |
| API-07 | `CORS_ORIGINS` definido no `.env` | Substitui a lista padrão |
| API-08 | Origem de produção (`agroscope-nu.vercel.app`) configurada | Produção liberada; `localhost` deixa de ser liberado |

### 1.2 Rotas de informação
| ID | Caso | Esperado |
|---|---|---|
| API-10 | `GET /modelinfo` | Formato antigo: `models.<cultura>.class_names` e `num_classes` |
| API-11 | `GET /health` | Lista as culturas carregadas; `gate=false` quando não há modelo de triagem |
| API-12 | `GET /cultures` | Mostra as culturas indisponíveis (ex.: milho sem modelo) e o tipo de preparo de cada uma (v2 = recorte da folha; v1 = legado) |
| API-13 | `GET /cultures` (contrato lido pelo backend) | Cada cultura carregada traz `code` (ex.: `SOYBEAN`), `label` (ex.: `Soja`) e `classes` |
| API-14 | Modelo de milho configurado (`CORN`) | Milho aparece em `/cultures` (`code=CORN`, `label=Milho`), sai de `unavailable` e passa a ser aceito no `/predict` — sem mudar código |
| API-15 | `/predict` com o `code` vindo do backend (`COFFEE`, `SOYBEAN`, `TOMATO`, `WHEAT`) | 200 |

### 1.3 `/predict`: sucesso e formato da resposta
| ID | Caso | Esperado |
|---|---|---|
| API-20 | Imagem válida para "Tomato" | 200; `culture="Tomato"` (como enviado), `culture_key="tomato"`, `expert.predict="DISEASED"` |
| API-21 | Campos da resposta | `diagnosis` completo; probabilidades somam 1; `expert.predict_confidence` = certeza × 100 |
| API-22 | Nomes em português, maiúsculas e acentos (`Café`, `SOJA`, `trigo`, `Tomate`) | Reconhecidos |
| API-23 | Cultura com modelo antigo (trigo) | `legacy_preprocessing=true` e sem recorte da folha |
| API-24 | `explain=true` | Mapa de calor em PNG válido (base64) |
| API-25 | PNG com transparência | 200 |
| API-26 | Foto enviada como `application/octet-stream` (comum em apps mobile) | 200 |
| API-27 | Certeza baixa (~52%) | `status="review"` com motivo "confiança baixa" |

### 1.4 `/predict`: entradas inválidas
| ID | Caso | Esperado |
|---|---|---|
| API-30 | Cultura sem modelo configurado ("Corn") | 400, com a lista de culturas disponíveis |
| API-31 | Cultura inexistente ("banana") | 400 |
| API-32 | Arquivo que não é imagem (`text/plain`) | 400 |
| API-33 | Imagem corrompida | 400 |
| API-34 | Arquivo vazio | 400 |
| API-35 | Arquivo maior que o limite | 413 |
| API-36 | Requisição sem `culture` | 422 |
| API-37 | Requisição sem `file` | 422 |

### 1.5 `/predict`: triagem
| ID | Caso | Esperado |
|---|---|---|
| API-40 | Sem modelo de triagem, imagem 32×32 | 422 `low_quality` (regra de resolução mínima: 128 px) |
| API-41 | Triagem diz "não é folha" | 422 `not_leaf`, com mensagem pronta para o usuário e **sem** diagnóstico |
| API-42 | "Não é folha" + `force=true` | Continua 422 (o `force` nunca libera não-folha) |
| API-43 | Triagem diz "qualidade ruim" | 422 `low_quality` |
| API-44 | "Qualidade ruim" + `force=true` | 200, mas `status="review"` com motivo "qualidade da imagem" |
| API-45 | Triagem com modelo, foto boa | `gate.method="model"`, `verdict="ok"` |
| API-46 | Caminho do modelo de triagem inválido | A API sobe usando só as regras simples |

### 1.6 Configuração
| ID | Caso | Esperado |
|---|---|---|
| API-50 | Sem variáveis definidas | Limite de 8 MB, `0.0.0.0:5000`, origens locais de desenvolvimento |
| API-51 | `MAX_UPLOAD_MB=2` | Limite de 2 MB (`MAX_UPLOAD_BYTES` tem prioridade, se definido) |
| API-52 | Nenhum modelo especialista configurado | A API **não sobe** (erro claro em vez de responder errado) |
| API-53 | Caminhos dos modelos entre aspas no `.env` (`COFFEE='./models/...'`), como chegam pelo `docker run --env-file` | Modelos carregados normalmente |

---

## 2. Casos de uso de ponta a ponta (`test_use_cases.py`)

API completa com os **modelos reais**, simulando o que o usuário faz no aplicativo.

| ID | Caso de uso | Esperado |
|---|---|---|
| UC-01 | API no ar | `/health` lista café, soja e trigo; triagem ativa |
| UC-02 | Consultar os modelos | Café usa recorte da folha; trigo usa o preparo antigo; tomate aparece como indisponível |
| UC-03 a 07 | Foto de folha de café (saudável / Phoma / ferrugem) e de soja (ferrugem / crestamento) | Diagnóstico correto, triagem `ok` e folha recortada |
| UC-08 | Front envia "Soja" ou "Café" | Reconhecido |
| UC-09 | Front antigo lê `expert.predict` e `expert.predict_confidence` | Mesmos valores do diagnóstico novo |
| UC-10 | Cultura com modelo antigo (trigo) | Funciona, com o preparo antigo |
| UC-11 | Pedir o mapa de calor | PNG válido; ponto principal dentro da imagem |
| UC-12 | Foto do celular salva "deitada" com marcação EXIF de rotação | Mesmo diagnóstico da foto na posição certa |
| UC-13 | PNG com transparência | 200 |
| UC-14 a 16 | Foto de documento, céu ou parede | 422 `not_leaf` |
| UC-17 | Documento + `force=true` | Continua 422 |
| UC-18 | Foto muito tremida/desfocada | 422 `low_quality`, mensagem pedindo nova foto |
| UC-19 | Foto tremida + `force=true` | 200, marcada para revisão |
| UC-20 | Foto escura | **Aceita**, com aviso "imagem escura" (medimos que escuridão não atrapalha o diagnóstico) |
| UC-21 a 23 | Foto estourada, compressão extrema (JPEG qualidade 3) ou minúscula (64 px) | 422 `low_quality` |
| UC-24 e 25 | Desfoque leve ou resolução média (400 px) | Aceita (não atrapalha o diagnóstico) |
| UC-26 | Cultura sem modelo carregado | 400 |
| UC-27 | Arquivo de texto | 400 |
| UC-28 | Imagem corrompida | 400 |
| UC-29 | 8 usuários ao mesmo tempo, com mapa de calor | Todos recebem o mesmo resultado de uma chamada isolada |

---

## 3. Testes de IA (`test_ai.py`)

Verificam se os modelos **se comportam** como esperado. Rode sempre que um modelo for trocado ou retreinado.

### 3.1 Modelos carregados
| ID | Caso | Esperado |
|---|---|---|
| IA-01 | Metadados de cada especialista | Classes únicas, incluindo uma classe "saudável"; modelos v2 informam arquitetura e tamanho de entrada |
| IA-02 | Modelo de triagem | Classes `leaf_ok`, `leaf_poor`, `not_leaf` |

### 3.2 Especialistas
| ID | Caso | Esperado |
|---|---|---|
| IA-10 | 5 fotos de referência (café e soja) | Diagnóstico correto, com certeza ≥ 60% |
| IA-11 | Mesma foto duas vezes | Mesmo resultado (determinístico) |
| IA-12 | Foto espelhada, de cabeça para baixo ou girada 90° (5 fotos × 3) | Mesmo diagnóstico: folha não tem lado "certo" |
| IA-13 | Defeitos leves: desfoque leve, JPEG 50, mais escura, 512 px (5 fotos × 4) | Mesmo diagnóstico: o que a triagem deixa passar não pode mudar o resultado |
| IA-14 | **Mesma folha no canto esquerdo, no centro ou no canto direito da foto** | Mesmo diagnóstico. Garante que o problema original ("o modelo só olha o centro") não volte |

### 3.3 Mapa de atenção (Grad-CAM)
| ID | Caso | Esperado |
|---|---|---|
| IA-20 | Onde o modelo "olha" (5 fotos) | ≥ 60% do mapa de calor sobre a folha, não sobre o fundo |
| IA-21 | Fotos diferentes | Mapas diferentes. Garante que o bug antigo, em que todas as fotos mostravam o mesmo mapa, não volte |

### 3.4 Triagem
| ID | Caso | Esperado |
|---|---|---|
| IA-30 | 6 fotos de folha (café, soja e trigo) | `ok` |
| IA-31 | Documento, céu, parede, ruído, tela preta, verde liso | `not_leaf` (o verde liso garante que a cor sozinha não basta) |
| IA-32 | Imagem toda branca | Recusada (`low_quality` ou `not_leaf`). O modelo a trata como foto estourada, e a mensagem "tire outra foto com boa luz" faz sentido |
| IA-33 | Desfoque forte, estourada, JPEG qualidade 3, 40 px | `low_quality` |
| IA-34 | Foto escura | `ok`, com aviso |

### 3.5 Acerto nos conjuntos completos (`pytest -m dataset`)

Precisam do HD de dados. O caminho padrão é `/media/kaua-matheus/HD3201/Data/Agroscope`; para outro local, defina a variável `DATA_ROOT`. Os mínimos ficam um pouco abaixo do valor medido, para pegar regressões sem falhar por variação normal.

| ID | Caso | Mínimo | Medido (out/2026) |
|---|---|---|---|
| IA-40 | Café: teste externo | 95% | 97,8% (315 fotos) |
| IA-41 | Soja: teste externo | 95% | 99,0% (306 fotos) |
| IA-42 | Tomate v2: teste externo | 80% | 90,0% (390 fotos) |
| IA-43 | Triagem: folhas boas aceitas (café e soja) | 98% | 99,8% (621 fotos) |
| IA-44 | Triagem: objetos e cenas recusados (Caltech-101, sem categorias de planta) | 97% | 100% (400 imagens) |

---

## Decisões registradas nos testes

- **Imagem inválida ou corrompida devolve 400** (e não 415), igual à API antiga, para não quebrar o front.
- **`culture` na resposta devolve o nome como foi enviado**, também por compatibilidade com o front antigo. O nome normalizado fica em `culture_key`.
- **Foto escura não é recusada:** escurecer a foto até 10% do brilho tirou 0 a 4 pontos do acerto, abaixo do limite de 5 pontos usado para recusar (ver `Steps/gate_lab`).
- **Os limites de qualidade da triagem** (o que é "inviável") vêm da medição de quanto cada defeito derruba o acerto dos especialistas, e não de valores escolhidos à mão.

## Ao adicionar uma cultura ou trocar um modelo

1. Coloque o `.pth` em `models/<Cultura>/` e a variável no `.env`.
2. Adicione 1 ou 2 fotos de referência (de teste externo) em `tests/fixtures/` e inclua-as em `LEAVES` (`test_ai.py`) e em `test_diagnostico_correto` (`test_use_cases.py`).
3. Adicione o conjunto de teste externo em `test_acerto_no_teste_externo`, com um mínimo um pouco abaixo do acerto medido.
4. Rode `pytest` e `pytest -m dataset`.
