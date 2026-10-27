# Agroscope IA — Laboratório de Modelos

Esta pasta (`Steps/`) é o **laboratório** do projeto: aqui os modelos de IA são treinados, testados e comparados antes de irem para a aplicação (`Main/Project/IA`).

O objetivo do projeto é identificar doenças em folhas de plantas a partir de uma foto. O usuário tira a foto, a IA responde qual doença encontrou (ou se a folha está saudável), com que grau de certeza, e mostra em qual parte da folha ela se baseou.

---

## Sumário
1. [Como funciona](#como-funciona)
2. [Situação atual por cultura](#situação-atual-por-cultura)
3. [Histórico do que foi feito](#histórico-do-que-foi-feito)
4. [Estudos e decisões](#estudos-e-decisões)
5. [Como treinar um modelo](#como-treinar-um-modelo)
6. [Organização das pastas](#organização-das-pastas)
7. [Limitações conhecidas](#limitações-conhecidas)
8. [Próximos passos](#próximos-passos)
9. [Como registrar um novo teste](#como-registrar-um-novo-teste)

---

## Como funciona

```
Foto da folha  →  Recorte da folha  →  Modelo de IA  →  Resultado
                  (tira o fundo)       (ConvNeXt)       doença + certeza + mapa de atenção
```

1. **Recorte da folha** — se a foto tem a folha sobre um fundo claro, o sistema encontra a folha e recorta só ela. Isso evita que a IA perca detalhes por causa do fundo.
2. **Modelo** — usamos o **ConvNeXt-Tiny**, um modelo já treinado pelo mercado em milhões de imagens gerais e que nós "especializamos" com fotos de folhas (técnica chamada *transfer learning*). Existe **um modelo por cultura** (café, soja, trigo...).
3. **Resultado** — além da doença, o sistema devolve:
   - o **grau de certeza** de cada classe;
   - um **status**: `confiante`, `incerto` ou `revisar` (quando a certeza é baixa ou a foto tem problema);
   - avisos de **qualidade da foto** (borrada, escura, clara demais);
   - um **mapa de calor** (Grad-CAM) mostrando onde a IA "olhou".

---

## Situação atual por cultura

"Teste externo" = fotos que a IA nunca viu durante o treino, tiradas em outras condições. É a métrica que mais importa: mostra como o modelo se sai no mundo real.

| Cultura | Classes | Teste externo | Data | Observação |
|---|---|---|---|---|
| Café | Saudável, Phoma, Ferrugem | **97,8%** (315 fotos) | 19/10/2026 | Usa fotos de folha inteira como dado extra (`EXTRA_PATH`) |
| Soja | Crestamento bacteriano, Saudável, Ferrugem | **99,0%** (306 fotos) | 26/10/2026 | |
| Trigo, Tomate, Milho | — | não reavaliado | — | Modelos antigos; precisam ser retreinados com o pipeline novo |

---

## Histórico do que foi feito

### Out/2026 — Correção do "modelo que só olhava o centro" (café)

**Problema relatado:** o mapa de atenção da IA ficava sempre no centro / meio-inferior da imagem.

**O que descobrimos:**
- **As fotos de treino e de teste eram muito diferentes.** O treino só tinha fotos bem de perto (a folha ocupando a imagem toda); o teste tinha a folha inteira, pequena, no meio de um fundo branco. A IA nunca tinha visto esse tipo de foto. Testando com todas as 315 fotos de teste, o modelo antigo acertava só **32%** — praticamente um chute. (O número de 80% que aparecia antes vinha de apenas 15 fotos.)
- **O mapa de atenção estava quebrado.** Um erro no código fazia todas as imagens mostrarem o mesmo mapa (o da última imagem calculada). Por isso parecia que a IA sempre olhava para o mesmo ponto.
- **Outros erros no treino:** a velocidade de aprendizado oscilava sem controle; as fotos de validação estavam sendo alteradas aleatoriamente como se fossem de treino; o corte central da imagem jogava fora as bordas da folha.
- **Um "atalho" nos dados:** quase todas as fotos de ferrugem do treino eram de baixa resolução (borradas). A IA podia aprender "foto borrada = ferrugem" em vez de olhar a doença.

**O que foi feito:**
- Recorte automático da folha antes de passar para a IA.
- Inclusão de fotos de folha inteira no treino (pasta `Coffee_5`), **sem** usar nenhuma das fotos de teste.
- Variações artificiais mais fortes nas fotos de treino (rotação, cor, desfoque, partes apagadas) para a IA não depender de atalhos.
- Correção de todos os erros de código citados.

**Resultado:** teste externo do café foi de **32% → 97,8%**. O mapa de atenção agora acompanha as lesões em cada foto.

### Out/2026 — Teste de dois modelos juntos (ConvNeXt + VGG16)

**Pergunta:** usar um segundo modelo (VGG16) para "ver detalhes" melhora o resultado?

**Resposta: não compensa.**
- A VGG16 sozinha foi **pior** que o ConvNeXt (97,5–97,8% contra 98,7%), é **5× maior** (512 MB) e **3× mais lenta**.
- Juntar os dois acertou 2 fotos a mais em 315 — diferença pequena demais para ser confiável. Juntar o ConvNeXt com **qualquer** outro modelo dá um ganho parecido; o ganho não vem da VGG.
- O mapa de atenção da VGG é mais "fino", mas olha mais para fundo e nervuras do que para as lesões.
- **Decisão:** manter um modelo só. O próprio grau de certeza do ConvNeXt já serve de alerta: marcando como "revisar" as fotos com certeza abaixo de 80% (cerca de 4% das fotos), ele sinaliza 6 de cada 7 erros.
- Se no futuro for usar dois modelos, a forma eficiente é em **cascata**: o segundo modelo só é chamado quando o primeiro está inseguro.

### Out/2026 — Balanceamento de classes e SMOTE

**Pergunta:** é preciso gerar imagens artificiais (SMOTE) para equilibrar as classes?

**Resposta: não.** Melhor do que criar imagens artificiais é parar de descartar fotos reais: usando todas as fotos da soja o acerto subiu de ~98,4% para ~99,7%. Detalhes em [Balanceamento de classes](#balanceamento-de-classes-smote).

### Out/2026 — Triagem de imagens e atualização da aplicação

**Pergunta:** como evitar que fotos que não são de folha, ou de qualidade ruim, recebam um diagnóstico?

**O que foi feito:**
- Criado um modelo pequeno de triagem que roda antes do especialista: diz se a foto é de folha e se tem qualidade para análise.
- A aplicação (`Main/Project/IA`) foi atualizada: recorte da folha, triagem, respostas detalhadas, suporte a modelos antigos e novos, e 31 testes de casos de uso.

**Resultado:** 100% das não-folhas recusadas; 95% das fotos inviáveis barradas; só 1,6% das fotos boas recusadas por engano. Detalhes em [Triagem da imagem](#triagem-da-imagem-folha--não-folha-e-qualidade).

**Decisão:** a triagem fica ativa na aplicação. A foto escura não é mais motivo de recusa (medimos que não atrapalha o diagnóstico).

---

## Estudos e decisões

### Comparação de modelos (café, mesmo método de treino, 315 fotos externas)

| Modelo | Acerto | Tamanho | Tempo por foto (CPU) | Conclusão |
|---|---|---|---|---|
| **ConvNeXt-Tiny** | **98,7%** | 106 MB | ~65 ms | **Modelo principal** |
| EfficientNet-B0 | 98,7% (soja: 99,0% contra 99,3% do ConvNeXt) | **15 MB** | **~27 ms** | Ótima alternativa leve (celular / servidor sem placa de vídeo); a "certeza" que ela informa é um pouco menos confiável |
| VGG16 | 97,5–97,8% | 512 MB | 200–390 ms | Não recomendado |

> Testes antigos (antes das correções de out/2026) estão no histórico do Git; os números não são comparáveis com os atuais porque o método de treino mudou.

### Balanceamento de classes (SMOTE)

**Contexto:** as classes não têm o mesmo número de fotos (soja: 660 de crestamento contra ~1.700 das outras; café: 1.074 saudáveis contra ~7.000 doentes). Hoje o notebook **descarta** fotos das classes maiores até todas ficarem do tamanho da menor.

**SMOTE não é recomendado:**
- O SMOTE cria exemplos "misturando" dois exemplos parecidos. Em fotos, isso gera imagens fantasmas (duas folhas sobrepostas e borradas), que não existem na natureza e podem ensinar coisas erradas.
- As variações artificiais que o treino já faz (rotação, cor, desfoque...) cumprem melhor esse papel.
- O problema real do projeto não é o desequilíbrio, e sim a **falta de fotos de campo**, e isso nenhuma técnica artificial resolve.

**O que funciona melhor: usar todas as fotos** e fazer o treino sortear as classes com a mesma frequência (*amostragem ponderada*), em vez de descartar fotos. Teste na soja (306 fotos externas):

| Modelo | Descartando fotos (atual) | Usando todas as fotos |
|---|---|---|
| ConvNeXt (tentativa 1) | 98,7% | **99,3%** |
| ConvNeXt (tentativa 2) | 98,0% | **100%** |
| EfficientNet-B0 | 98,4% | **99,0%** |

A melhora apareceu em todas as tentativas, e a classe menor não foi prejudicada. **Decisão sugerida:** trocar o descarte de fotos pela amostragem ponderada no notebook.

### Triagem da imagem (folha / não-folha e qualidade)

**Pergunta:** vale ter um modelo só para barrar fotos que não são de folha ou que estão ruins demais, antes do diagnóstico?

**Resposta: sim.** Sem triagem, qualquer foto (um documento, o céu, uma parede) recebe um diagnóstico de doença — o especialista sempre escolhe uma das classes que conhece.

**Como "imagem ruim" foi definida:** em vez de chutar limites, medimos quanto cada defeito derruba o acerto dos especialistas de café e soja. Só é recusado o que derruba **5 pontos ou mais**:

| Defeito | Leve | Moderado | Severo |
|---|---|---|---|
| Desfoque | passa | passa | recusa |
| Foto tremida | passa | recusa | recusa |
| Foto escura | passa | passa | passa (não atrapalha!) |
| Foto estourada (clara demais) | recusa | recusa | recusa |
| Resolução baixa | passa | passa | recusa |
| Compressão (ex.: reenviada por mensageiro) | passa | recusa | recusa |
| Ruído (granulado) | passa | recusa | recusa |

**Resultado (modelo de 16 MB, ~22 ms por foto em CPU):**

| | Modelo de triagem | Regras simples (cor/nitidez) | Certeza do especialista |
|---|---|---|---|
| Folhas reais aceitas | 99–100% | 62–100% | 51–98% |
| Objetos e fundos recusados (categorias nunca vistas) | 99–100% | 58–64% | 75–80% |
| Fotos boas recusadas por engano | **1,6%** | 75% | — |
| Fotos ruins barradas | **95%** | 63% | — |

- Folhas de uma **cultura que o modelo nunca viu** (trigo, em um teste à parte) foram aceitas em 88,5% dos casos. Ao adicionar uma cultura nova, inclua fotos dela no treino da triagem.
- **A triagem não confere se a folha é da cultura escolhida.** Uma folha de café enviada como "tomate" passa e recebe um diagnóstico de tomate.
- Detalhes e como refazer: [`gate_lab/`](gate_lab/README.md).

### Cruzar com dados de clima e localização

- **Faz sentido**: as doenças têm "climas preferidos" diferentes. A ferrugem do café gosta de 20–25 °C com folha molhada (novembro a março); a Phoma gosta de frio (15–20 °C), vento e altitude (abril a outubro).
- **Existem fontes gratuitas** de clima por localização (Open-Meteo, NASA POWER, INMET).
- **Hoje não é possível treinar a IA com clima**: nenhuma foto do projeto tem GPS e quase todas foram tiradas no mesmo dia e lugar.
- **Plano sugerido**:
  1. O aplicativo passa a guardar localização, data e clima de cada foto enviada (junto com a confirmação do usuário).
  2. Mostrar um "risco climático" de cada doença como informação complementar, sem alterar o diagnóstico da foto.
  3. Quando houver fotos suficientes de regiões e épocas diferentes, testar a combinação foto + clima.

---

## Como treinar um modelo

1. **Organize as fotos** em pastas, uma por classe (o nome da pasta vira o nome da classe):
   ```
   Base_Cultura/            Test_Cultura/
   ├── Healthy/             ├── Healthy/
   ├── Doenca_A/            ├── Doenca_A/
   └── Doenca_B/            └── Doenca_B/
   ```
   - `Base_...` = fotos de treino.
   - `Test_...` = fotos **separadas** para o teste externo. Nunca coloque a mesma foto nas duas.
   - As duas pastas precisam ter **as mesmas classes**.
2. **Configure o `.env`**:
   ```
   DATA_PATH="caminho/para/Base_Cultura"
   EVAL_PATH="caminho/para/Test_Cultura"
   MODEL_PATH="./Models/Saved/Expert/Cultura/cultura.pth"
   EXTRA_PATH="caminho/para/fotos_extras"   # opcional: fotos de outro tipo (ex.: folha inteira)
   ```
3. **Rode o `lab.ipynb`** do início ao fim. Ele treina, mostra os gráficos, avalia no teste externo, mostra os mapas de atenção e salva o modelo em `MODEL_PATH`.
4. **Registre o resultado** neste README (tabela de culturas + uma entrada no histórico).

Pontos de atenção:
- O que vale é o **acerto no teste externo**, não o do treino.
- Olhe os **mapas de atenção dos erros**: se a IA olha para o fundo ou para a borda, há um problema nos dados.
- Para usar o modelo na aplicação, use `utils/inference.py` (ele aplica o mesmo recorte de folha do treino). Sem isso, o problema do "olhar só o centro" volta.

---

## Organização das pastas

| Caminho | O que é |
|---|---|
| `lab.ipynb` | Notebook principal: treino, avaliação e diagnóstico do modelo de classificação |
| `yolo_lab.ipynb` | Experimentos com YOLO (detecção de folhas) |
| `utils/models.py` | Preparação das imagens para treino e teste |
| `utils/leaf.py` | Recorte automático da folha |
| `utils/inference.py` | Versão de laboratório do diagnóstico completo (a aplicação usa a sua própria cópia em `Main/Project/IA/core/`) |
| `gate_lab/` | Experimento e treino do modelo de triagem (folha / não-folha / qualidade) |
| `Models/Saved/Expert/` | Modelos treinados, um por cultura (não vai para o Git) |
| `.env` | Caminhos das pastas de dados e do modelo (não vai para o Git) |

---

## Limitações conhecidas

- **Fotos de campo** (folha no pé, fundo com outras folhas e terra) ainda são o ponto fraco: no café, cerca de 75–80% de acerto, contra ~98% nas fotos com fundo claro. A solução é **juntar mais fotos de campo** para o treino.
- **O teste externo do café** vem quase todo de uma mesma sessão de fotos. Um resultado de 98% aqui não garante o mesmo em outras fazendas.
- **Trigo, tomate e milho** ainda não foram retreinados com o método novo (a aplicação usa os modelos antigos com o preparo antigo).
- **A triagem não confere a cultura**: uma folha de outra planta é diagnosticada como se fosse da cultura escolhida.

---

## Próximos passos

- [x] Atualizar a aplicação para o pipeline novo (out/2026, com triagem e testes de uso).
- [ ] Ajustar o front para mostrar as recusas da triagem e os avisos (ver `Main/README.md`).
- [ ] Verificar a cultura da folha (ex.: incluir a cultura como saída da triagem).
- [ ] Trocar o descarte de fotos pela amostragem ponderada (ver [Balanceamento](#balanceamento-de-classes-smote)).
- [ ] Guardar uma cópia reduzida das fotos de treino. As fotos da soja têm 12–20 megapixels e o treino passa a maior parte do tempo só abrindo arquivos do HD (placa de vídeo a ~6% de uso). Reduzir as fotos uma vez antes de treinar deixa o treino várias vezes mais rápido.
- [ ] Retreinar trigo, tomate e milho com o método novo e preencher a tabela de culturas.
- [ ] Juntar fotos de campo (fundo natural) de cada doença, separando algumas para teste.
- [ ] Avaliar a EfficientNet-B0 como modelo principal se o servidor não tiver placa de vídeo, ou para rodar direto no celular.
- [ ] Guardar localização, data e clima das fotos enviadas pelo aplicativo (base para o estudo com clima).

---

## Como registrar um novo teste

Copie o modelo abaixo para a seção [Histórico](#histórico-do-que-foi-feito):

```markdown
### Mês/Ano — Título curto do teste

**Pergunta:** o que queríamos descobrir?

**O que foi feito:** o que mudou (dados, modelo, configuração).

**Resultado:** número no teste externo antes → depois, e quantas fotos foram usadas.

**Decisão:** o que fica, o que sai, e por quê.
```
