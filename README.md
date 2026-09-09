# Sobre
Repositório com o propósito de aprimorar o modelo IA por meio de testes e descobertas. A ideia é realizar testes em etapas enfim melhorar o modelo.

# Modelos
convnext_tiny:
- F1-Macro (0.9680)
- BACC (0.9694)
- Loss (0.1255)

efficientnet_b0:
- F1-Macro (0.9606)
- BACC (0.9624)
- Loss (0.1497)

densenet121:
- F1-Macro (0.9374)
- BACC (0.9409)
- Loss (0.2344)

vit_b_16:
- F1-Macro (0.9490)
- BACC (0.9499)
- Loss (0.1839)

resnet18:
- F1-Macro (0.9391)
- BACC (0.9394)
- Loss (0.1946)

# Otimizadores
`Ainda em pesquisa..`

# Camadas
A quantidade de camadas utilizadas no processo de treinamento do modelo é suficiente, atualmente o modelo sofre de overfitthing, justamente aumentar a quantidade de camadas utilizadas piora essa condição. Estamos em um nível ideal para treinamento.

# Testes Futuros
- O modelo sobre de Overfitting, eval x teste em treino tem muita discrepância.
- Testar resoluções maiores. Resize(320)/CenterCrop(288)
- Adicionar RandomRotation e RandomErasing
- Testar Label Smoothing (nn.CrossEntropyLoss(label_smoothing=0.1))