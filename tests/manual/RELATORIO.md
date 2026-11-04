# Relatório dos testes manuais da IA

Executado em 03/11/2026. Especificação no formato da planilha geral: `Casos_de_Teste_Agroscope_IA.xlsx`, na raiz do repositório.

**Resultado: 30 casos concluídos, nenhum pendente: 13 aprovados e 17 reprovados.**

## Como foi feito

- Cada caso enviou uma foto real à API da IA (`POST /predict`), com os modelos reais e a triagem ativa, simulando o que o cliente faria. O front e o backend não foram executados.
- "Cliente vê o diagnóstico" aplica a regra do backend (`PredictUseCase`): abaixo de 80% de confiança, ele pede nova foto.
- Os critérios foram gravados antes da execução, em `casos.json`. Para repetir: `python tests/manual/executar_casos.py "<pasta das imagens>" <URL da API>`.

## Principais conclusões

1. **8 de 30 casos mostrariam um diagnóstico errado ao cliente, com 85,9% a 100% de confiança** (CT-IA-12, 15, 17, 18, 19, 20, 22, 25). Destes, 6 vieram dos modelos v1 de trigo e milho.
2. **A IA não confere se a folha é da cultura escolhida.** Folha de milho enviada como trigo virou ferrugem parda com 100% (CT-IA-17), e tomate saudável enviado como milho virou ferrugem comum (CT-IA-18).
3. **Doenças fora do catálogo viram doenças do catálogo.** Requeima virou mofo das folhas com 93,2% (CT-IA-19), e cercosporiose do milho virou ferrugem comum com 99,3% (CT-IA-20).
4. **O limite de 80% esconde acertos:** 5 fotos tiveram a classe certa e receberiam pedido de nova foto (CT-IA-06, 07, 09, 28, 30). A confiança não separou bem acertos de erros nesta amostra.
5. **A triagem acertou 4 de 5 fotos que não eram de folha;** só a espiga de trigo passou (CT-IA-25). Em compensação, recusou 2 folhas doentes (CT-IA-08 e 11).

## Limites destes resultados

- Há uma foto por caso. Os resultados apontam riscos, mas **não medem a taxa de acerto**, e os números acima não representam proporções reais de uso.
- Rótulos menos seguros: CT-IA-02 (título da fonte; sintoma pequeno), CT-IA-06 (foto realçada), CT-IA-08 (fundo azul de divulgação) e CT-IA-15 (espécie sem confirmação). CT-IA-15 **não deve ser tratado como erro confirmado** antes da revisão de um agrônomo.
- CT-IA-08 teve a foto trocada depois da primeira rodada, porque a original parecia pimentão, não tomate. Com ela, a IA dava a classe certa com 71%; com a nova, recusa na triagem.
- Nenhuma foto é do seu campo. Fotos reais de clientes seriam o próximo passo mais útil.

## Origem das imagens

As 27 fotos vêm de iNaturalist, Wikimedia Commons e PlantDoc, fontes que não estão na lista de `Agroscope.md`. Três casos (CT-IA-16, 17 e 18) reutilizam fotos de outros casos, com a cultura errada.

Elas foram comparadas com os 177.317 arquivos de imagem de `HD3201/Data/Agroscope` (treino, teste e negativos): **nenhuma cópia exata** (SHA-256) e **nenhuma duplicata** por semelhança. A busca por dHash deu 12 alertas, todos da espiga de milho (CT-IA-23), contra imagens sem relação, como um piano e um íbis. A correlação de pixels ficou entre 0,41 e 0,70, abaixo do que se espera de uma duplicata. Dos arquivos, 1.735 não puderam ser comparados por semelhança: 1.632 são metadados `__MACOSX` e 103 estão vazios (0 byte).

**O que não pôde ser comparado, por não estar no HD:** Trigo 3 (Kaggle), Trigo 4 (`wgd66f8n6h`), os dois conjuntos "Novo" de café (`mfpxg4y65r` e `pmkbyjpf6k`) e os dados de treino do milho, que também não constam em `Agroscope.md`. As pastas `Base_Wheat` e `Test_Wheat` estão vazias. Para esses conjuntos, a garantia vem só da origem das fotos.

Fontes, autores e licenças de cada foto estão em `imagens/FONTES.csv` (com cópia na pasta do HD). Várias têm licença CC BY-NC, e as do PlantDoc vêm de um dataset coletado da web, cujas fotos podem ter direitos próprios (algumas trazem o nome do autor). Por isso as imagens ficam no HD, fora do Git. Os caminhos indicados nos casos são relativos a `/media/kaua-matheus/HD3201/Data/Caso Testes Agroscope`.

## Casos


### Capacidade: diagnóstico por cultura e classe

**CT-IA-01 — Aprovado.** Verificar se folha de café saudável é reconhecida como saudável.

- Entrada: Cultura: Café (COFFEE); Foto: Folha de café saudável, foto aproximada (`Café/Healthy/CT-IA-01_cafe_saudavel.jpg`)
- Esperado: A IA aceita a foto e devolve a classe Healthy com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Triagem aceitou (folha: 98,3%). Modelo: Healthy com 92,9%. Cliente vê o diagnóstico.

**CT-IA-02 — Reprovado.** Verificar se a IA identifica a mancha de Phoma no café.

- Entrada: Cultura: Café (COFFEE); Foto: Folha de café com Phoma (`Café/Phoma/CT-IA-02_cafe_phoma.jpg`)
- Esperado: A IA aceita a foto e devolve a classe Phoma com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Triagem aceitou (folha: 98,1%). Modelo: Rust com 62,7%. Abaixo de 80%: backend pede nova foto.
- Observação: Classe tirada do título da foto na fonte; o sintoma visível é pequeno (borda necrosada). Vale repetir com outra foto de Phoma.

**CT-IA-03 — Aprovado.** Verificar se a IA identifica a ferrugem do cafeeiro.

- Entrada: Cultura: Café (COFFEE); Foto: Folha de café com ferrugem (Hemileia vastatrix), foto de celular (`Café/Rust/CT-IA-03_cafe_ferrugem.jpeg`)
- Esperado: A IA aceita a foto e devolve a classe Rust com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Triagem aceitou (folha: 98,1%). Modelo: Rust com 90,5%. Cliente vê o diagnóstico.

**CT-IA-04 — Aprovado.** Verificar se soja saudável é reconhecida como saudável.

- Entrada: Cultura: Soja (SOYBEAN); Foto: Plantas de soja saudáveis em campo (`Soja/Healthy/CT-IA-04_soja_saudavel.jpeg`)
- Esperado: A IA aceita a foto e devolve a classe Healthy com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Triagem aceitou (folha: 99,1%). Modelo: Healthy com 88,1%. Cliente vê o diagnóstico.

**CT-IA-05 — Aprovado.** Verificar se a IA identifica a ferrugem asiática da soja.

- Entrada: Cultura: Soja (SOYBEAN); Foto: Trifólio de soja com ferrugem (Phakopsora pachyrhizi) (`Soja/Rust/CT-IA-05_soja_ferrugem.jpg`)
- Esperado: A IA aceita a foto e devolve a classe Rust com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Triagem aceitou (folha: 98,5%). Modelo: Rust com 89,9%. Cliente vê o diagnóstico.

**CT-IA-06 — Reprovado.** Verificar se a IA identifica o crestamento bacteriano da soja.

- Entrada: Cultura: Soja (SOYBEAN); Foto: Folha de soja com crestamento bacteriano (P. syringae pv. glycinea) (`Soja/Bacterial_Blight/CT-IA-06_soja_crestamento.jpg`)
- Esperado: A IA aceita a foto e devolve a classe Bacterial_Blight com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Triagem aceitou (folha: 96,5%). Modelo: Bacterial_Blight com 69,1%. Abaixo de 80%: backend pede nova foto.
- Observação: Foto atípica: contraste realçado e moldura oval. A classe estava certa, mas abaixo de 80%.

**CT-IA-07 — Reprovado.** Verificar se folha de tomate saudável é reconhecida como saudável.

- Entrada: Cultura: Tomate (TOMATO); Foto: Folha de tomate saudável sobre fundo claro (`Tomate/Healthy/CT-IA-07_tomate_saudavel.jpg`)
- Esperado: A IA aceita a foto e devolve a classe Healthy com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Triagem aceitou (folha: 92,8%). Modelo: Healthy com 77,1%. Abaixo de 80%: backend pede nova foto.
- Observação: Classe certa, abaixo de 80%: o cliente receberia pedido de nova foto.

**CT-IA-08 — Reprovado.** Verificar se a IA identifica a mancha bacteriana do tomate.

- Entrada: Cultura: Tomate (TOMATO); Foto: Folha de tomate com mancha bacteriana (`Tomate/Bacterial_Spot/CT-IA-08_tomate_mancha_bacteriana.jpg`)
- Esperado: A IA aceita a foto e devolve a classe Bacterial_Spot com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Recusada na triagem (not_leaf; folha: 13,2%). Mensagem: A imagem não parece ser de uma folha. Envie uma foto de uma folha da planta.
- Observação: Foto de divulgação com fundo azul e várias folhas, que pode ter pesado na recusa. A primeira foto escolhida (descartada por parecer pimentão) deu Bacterial_Spot com 71%. Vale repetir com foto de campo.

**CT-IA-09 — Reprovado.** Verificar se a IA identifica o mofo das folhas do tomate.

- Entrada: Cultura: Tomate (TOMATO); Foto: Folha de tomate com mofo (Fulvia fulva), foto de celular (`Tomate/Leaf_Mold/CT-IA-09_tomate_mofo.jpeg`)
- Esperado: A IA aceita a foto e devolve a classe Leaf_Mold com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Triagem aceitou (folha: 98,0%). Modelo: Leaf_Mold com 70,4%. Abaixo de 80%: backend pede nova foto.
- Observação: Classe certa, abaixo de 80%: o cliente receberia pedido de nova foto.

**CT-IA-10 — Aprovado.** Verificar se trigo saudável é reconhecido como saudável (modelo v1).

- Entrada: Cultura: Trigo (WHEAT); Foto: Plantas de trigo saudáveis, folhas e espiga verdes (`Trigo/Healthy/CT-IA-10_trigo_saudavel.jpg`)
- Esperado: A IA aceita a foto e devolve a classe Healthy com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Triagem aceitou (folha: 98,3%). Modelo: Healthy com 100,0%. Cliente vê o diagnóstico.

**CT-IA-11 — Reprovado.** Verificar se a IA identifica a ferrugem parda do trigo (modelo v1).

- Entrada: Cultura: Trigo (WHEAT); Foto: Folha de trigo com ferrugem parda (P. recondita f.sp. tritici) (`Trigo/Brown_Rust/CT-IA-11_trigo_ferrugem_parda.jpg`)
- Esperado: A IA aceita a foto e devolve a classe Brown_Rust com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Recusada na triagem (not_leaf; folha: 44,0%). Mensagem: A imagem não parece ser de uma folha. Envie uma foto de uma folha da planta.
- Observação: Close de folha cortada, sobre superfície de madeira. A triagem recusou uma folha doente real.

**CT-IA-12 — Reprovado.** Verificar se a IA identifica a ferrugem amarela do trigo (modelo v1).

- Entrada: Cultura: Trigo (WHEAT); Foto: Folha de trigo com ferrugem amarela (P. striiformis), foto de celular (`Trigo/Yellow_Rust/CT-IA-12_trigo_ferrugem_amarela.jpeg`)
- Esperado: A IA aceita a foto e devolve a classe Yellow_Rust com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Triagem aceitou (folha: 88,7%). Modelo: Brown_Rust com 100,0%. Cliente vê o diagnóstico.
- Observação: Erro grave: ferrugem errada com 100% de confiança.

**CT-IA-13 — Reprovado.** Verificar se milho saudável é reconhecido como saudável (modelo v1).

- Entrada: Cultura: Milho (CORN); Foto: Planta jovem de milho saudável (`Milho/Healthy/CT-IA-13_milho_saudavel.jpeg`)
- Esperado: A IA aceita a foto e devolve a classe Healthy com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Triagem aceitou (folha: 97,8%). Modelo: Rust_Common com 45,8%. Abaixo de 80%: backend pede nova foto.

**CT-IA-14 — Aprovado.** Verificar se a IA identifica a ferrugem comum do milho (modelo v1).

- Entrada: Cultura: Milho (CORN); Foto: Folha de milho com ferrugem comum (P. sorghi), foto de celular (`Milho/Rust_Common/CT-IA-14_milho_ferrugem_comum.jpeg`)
- Esperado: A IA aceita a foto e devolve a classe Rust_Common com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Triagem aceitou (folha: 99,1%). Modelo: Rust_Common com 96,2%. Cliente vê o diagnóstico.

**CT-IA-15 — Reprovado.** Verificar se a IA identifica a ferrugem polissora do milho (modelo v1).

- Entrada: Cultura: Milho (CORN); Foto: Folha de milho com ferrugem polissora (P. polysora), segurada na mão; espécie indicada pelo autor, sem confirmação da comunidade (`Milho/Rust_Blight/CT-IA-15_milho_ferrugem_polissora.jpg`)
- Esperado: A IA aceita a foto e devolve a classe Rust_Blight com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Triagem aceitou (folha: 98,4%). Modelo: Rust_Common com 98,6%. Cliente vê o diagnóstico.
- Observação: Espécie sem confirmação na fonte, e as duas ferrugens são difíceis de separar por foto. Confirmar o rótulo com um agrônomo antes de considerar erro do modelo.


### Erro do cliente: cultura errada selecionada

**CT-IA-16 — Aprovado.** Verificar o que acontece quando o cliente envia folha de café com a cultura Soja selecionada.

- Entrada: Cultura: Soja (SOYBEAN); Foto: Mesma foto do CT-IA-03 (café com ferrugem), enviada como Soja (`Café/Rust/CT-IA-03_cafe_ferrugem.jpeg`)
- Esperado: A IA não entrega diagnóstico confiante ao cliente: recusa a foto na triagem ou responde com confiança < 80% (o backend então pede nova foto).
- Obtido: Triagem aceitou (folha: 98,1%). Modelo: Bacterial_Blight com 69,3%. Abaixo de 80%: backend pede nova foto.

**CT-IA-17 — Reprovado.** Verificar o que acontece quando o cliente envia folha de milho com a cultura Trigo selecionada.

- Entrada: Cultura: Trigo (WHEAT); Foto: Mesma foto do CT-IA-14 (milho com ferrugem), enviada como Trigo (`Milho/Rust_Common/CT-IA-14_milho_ferrugem_comum.jpeg`)
- Esperado: A IA não entrega diagnóstico confiante ao cliente: recusa a foto na triagem ou responde com confiança < 80% (o backend então pede nova foto).
- Obtido: Triagem aceitou (folha: 99,1%). Modelo: Brown_Rust com 100,0%. Cliente vê o diagnóstico.
- Observação: Erro grave: folha de milho diagnosticada como trigo doente com 100%. A IA não confere se a folha é da cultura escolhida.

**CT-IA-18 — Reprovado.** Verificar o que acontece quando o cliente envia folha de tomate com a cultura Milho selecionada.

- Entrada: Cultura: Milho (CORN); Foto: Mesma foto do CT-IA-07 (tomate saudável), enviada como Milho (`Tomate/Healthy/CT-IA-07_tomate_saudavel.jpg`)
- Esperado: A IA não entrega diagnóstico confiante ao cliente: recusa a foto na triagem ou responde com confiança < 80% (o backend então pede nova foto).
- Obtido: Triagem aceitou (folha: 92,8%). Modelo: Rust_Common com 85,9%. Cliente vê o diagnóstico.
- Observação: Erro grave: folha de tomate saudável diagnosticada como milho doente.


### Capacidade: problema fora do catálogo do modelo

**CT-IA-19 — Reprovado.** Verificar a resposta para requeima do tomate, doença que o modelo não conhece.

- Entrada: Cultura: Tomate (TOMATO); Foto: Folha de tomate com requeima (Phytophthora infestans) (`_Outros/Fora_do_Catalogo/CT-IA-19_tomate_requeima.jpg`)
- Esperado: A IA não entrega diagnóstico confiante ao cliente: recusa a foto na triagem ou responde com confiança < 80% (o backend então pede nova foto).
- Obtido: Triagem aceitou (folha: 98,3%). Modelo: Leaf_Mold com 93,2%. Cliente vê o diagnóstico.
- Observação: Erro grave: requeima informada como mofo das folhas, com manejo que seria o da doença errada.

**CT-IA-20 — Reprovado.** Verificar a resposta para cercosporiose do milho, doença que o modelo não conhece.

- Entrada: Cultura: Milho (CORN); Foto: Folha de milho com cercosporiose (gray leaf spot) (`_Outros/Fora_do_Catalogo/CT-IA-20_milho_cercosporiose.jpg`)
- Esperado: A IA não entrega diagnóstico confiante ao cliente: recusa a foto na triagem ou responde com confiança < 80% (o backend então pede nova foto).
- Obtido: Triagem aceitou (folha: 87,1%). Modelo: Rust_Common com 99,3%. Cliente vê o diagnóstico.
- Observação: Erro grave: doença fora do catálogo informada como ferrugem comum com 99,3%.

**CT-IA-21 — Aprovado.** Verificar a resposta para cercosporiose do café, doença que o modelo não conhece.

- Entrada: Cultura: Café (COFFEE); Foto: Folha de café com Cercospora coffeicola, foto de celular (`_Outros/Fora_do_Catalogo/CT-IA-21_cafe_cercospora.jpg`)
- Esperado: A IA não entrega diagnóstico confiante ao cliente: recusa a foto na triagem ou responde com confiança < 80% (o backend então pede nova foto).
- Obtido: Triagem aceitou (folha: 98,8%). Modelo: Rust com 56,3%. Abaixo de 80%: backend pede nova foto.

**CT-IA-22 — Reprovado.** Verificar a resposta para folha de soja com manchas claras que não são ferrugem nem crestamento.

- Entrada: Cultura: Soja (SOYBEAN); Foto: Trifólio de soja com manchas claras de causa não identificada na fonte (`_Outros/Fora_do_Catalogo/CT-IA-22_soja_manchas_claras.jpg`)
- Esperado: A IA não entrega diagnóstico confiante ao cliente: recusa a foto na triagem ou responde com confiança < 80% (o backend então pede nova foto).
- Obtido: Triagem aceitou (folha: 98,4%). Modelo: Healthy com 93,2%. Cliente vê o diagnóstico.
- Observação: Causa das manchas não identificada na fonte; a resposta 'saudável' ignora um sintoma visível.


### Erro do cliente: foto que não é de folha

**CT-IA-23 — Aprovado.** Verificar se a espiga de milho é recusada.

- Entrada: Cultura: Milho (CORN); Foto: Espiga de milho descascada segurada na mão (`_Outros/Nao_e_Folha/CT-IA-23_espiga_de_milho.jpg`)
- Esperado: A triagem recusa a foto (422) e o cliente recebe a mensagem pedindo outra foto; nenhum diagnóstico é gerado.
- Obtido: Recusada na triagem (not_leaf; folha: 4,2%). Mensagem: A imagem não parece ser de uma folha. Envie uma foto de uma folha da planta.

**CT-IA-24 — Aprovado.** Verificar se grãos de café são recusados.

- Entrada: Cultura: Café (COFFEE); Foto: Grãos de café torrados em uma tigela (`_Outros/Nao_e_Folha/CT-IA-24_graos_de_cafe.png`)
- Esperado: A triagem recusa a foto (422) e o cliente recebe a mensagem pedindo outra foto; nenhum diagnóstico é gerado.
- Obtido: Recusada na triagem (not_leaf; folha: 1,9%). Mensagem: A imagem não parece ser de uma folha. Envie uma foto de uma folha da planta.

**CT-IA-25 — Reprovado.** Verificar se a espiga de trigo é recusada.

- Entrada: Cultura: Trigo (WHEAT); Foto: Espiga verde de trigo segurada na mão (`_Outros/Nao_e_Folha/CT-IA-25_espiga_de_trigo.jpeg`)
- Esperado: A triagem recusa a foto (422) e o cliente recebe a mensagem pedindo outra foto; nenhum diagnóstico é gerado.
- Obtido: Triagem aceitou (folha: 96,8%). Modelo: Brown_Rust com 93,2%. Cliente vê o diagnóstico.
- Observação: Erro grave: espiga verde sem sintomas passou pela triagem e recebeu diagnóstico de ferrugem.

**CT-IA-26 — Aprovado.** Verificar se folha seca de outra espécie é recusada.

- Entrada: Cultura: Café (COFFEE); Foto: Folha seca de bordo segurada na mão, enviada como Café (`_Outros/Nao_e_Folha/CT-IA-26_folha_seca_de_bordo.jpg`)
- Esperado: A triagem recusa a foto (422) e o cliente recebe a mensagem pedindo outra foto; nenhum diagnóstico é gerado.
- Obtido: Recusada na triagem (not_leaf; folha: 15,3%). Mensagem: A imagem não parece ser de uma folha. Envie uma foto de uma folha da planta.

**CT-IA-27 — Aprovado.** Verificar se a foto da lavoura inteira é recusada.

- Entrada: Cultura: Milho (CORN); Foto: Lavoura de milho vista de longe, sem nenhuma folha em destaque (`_Outros/Nao_e_Folha/CT-IA-27_lavoura_de_milho.jpg`)
- Esperado: A triagem recusa a foto (422) e o cliente recebe a mensagem pedindo outra foto; nenhum diagnóstico é gerado.
- Obtido: Recusada na triagem (not_leaf; folha: 19,2%). Mensagem: A imagem não parece ser de uma folha. Envie uma foto de uma folha da planta.


### Erro do cliente: enquadramento da foto

**CT-IA-28 — Reprovado.** Verificar se a mão segurando a folha atrapalha o diagnóstico.

- Entrada: Cultura: Café (COFFEE); Foto: Folha de café com ferrugem segurada entre os dedos, fundo escuro (`Café/Rust/CT-IA-28_cafe_ferrugem_na_mao.jpeg`)
- Esperado: A IA aceita a foto e devolve a classe Rust com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Triagem aceitou (folha: 98,9%). Modelo: Rust com 76,3%. Abaixo de 80%: backend pede nova foto.
- Observação: Classe certa, abaixo de 80%: o cliente receberia pedido de nova foto.

**CT-IA-29 — Aprovado.** Verificar a resposta quando o cliente fotografa o pé inteiro de longe.

- Entrada: Cultura: Café (COFFEE); Foto: Cafeeiro inteiro em campo aberto, folhas pequenas na foto (`_Outros/Enquadramento/CT-IA-29_cafeeiro_de_longe.jpg`)
- Esperado: A IA não entrega diagnóstico confiante ao cliente: recusa a foto na triagem ou responde com confiança < 80% (o backend então pede nova foto).
- Obtido: Recusada na triagem (not_leaf; folha: 19,2%). Mensagem: A imagem não parece ser de uma folha. Envie uma foto de uma folha da planta.

**CT-IA-30 — Reprovado.** Verificar se folhas fotografadas contra o céu são reconhecidas.

- Entrada: Cultura: Café (COFFEE); Foto: Ramo de café com folhas sem sintomas contra o céu (`Café/Healthy/CT-IA-30_cafe_contra_o_ceu.jpg`)
- Esperado: A IA aceita a foto e devolve a classe Healthy com confiança ≥ 80% (limite a partir do qual o backend exibe o diagnóstico ao cliente).
- Obtido: Triagem aceitou (folha: 97,9%). Modelo: Healthy com 60,5%. Abaixo de 80%: backend pede nova foto.
- Observação: Classe certa, abaixo de 80%: o cliente receberia pedido de nova foto.

