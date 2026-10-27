# Laboratório da triagem (folha / não-folha / qualidade)

Modelo leve (MobileNetV3, 16 MB) que roda **antes** do especialista e decide se a foto pode ser analisada:
`leaf_ok` (segue), `leaf_poor` (folha, mas sem qualidade) ou `not_leaf` (não é folha).

Resultados completos em [`results/`](results/). Resumo no [README principal](../README.md#triagem-da-imagem-folha--não-folha-e-qualidade).

## Como refazer

Rodar dentro de `Steps/gate_lab`, com o `.venv` do Steps:

```bash
# 1) Monta o conjunto (folhas de todas as culturas + Caltech-101 como "não-folha"), reduzido para 768px
NEG_PATH=/media/kaua-matheus/HD3201/Data/Agroscope/Negatives/101_ObjectCategories python build_cache.py

# 2) (opcional) Mede quanto cada defeito de imagem derruba o acerto dos especialistas
python expert_impact.py

# 3) Treina a triagem. Os níveis "ruins" vêm do passo 2: só é recusado o que derruba >= 5 pontos
python train_gate.py --epochs 8 --wheat --out gate.pth \
  --bad '{"gauss":[3],"motion":[2,3],"bright":[1,2,3],"lowres":[3],"jpeg":[2,3],"noise":[2,3]}' \
  --ok  '{"gauss":[1,2],"motion":[1],"dark":[1,2],"lowres":[1,2],"jpeg":[1],"noise":[1]}'

# 4) Avalia
python eval_gate.py gate.pth
```

Depois copie `gate.pth` para `Main/Project/IA/models/Gate/gate.pth`.

| Arquivo | O que faz |
|---|---|
| `build_cache.py` | Separa treino/teste. Teste usa **categorias e culturas nunca vistas** no treino |
| `degrade.py` | Defeitos sintéticos: desfoque, tremido, escura, estourada, baixa resolução, compressão, ruído (4 níveis) |
| `synth.py` | "Não-folhas" sintéticas: céu, papel, parede, cores lisas |
| `expert_impact.py` | Acerto dos especialistas de café e soja sob cada defeito |
| `train_gate.py` | Treino |
| `eval_gate.py` | Comparação com heurísticas e com a certeza do especialista |
