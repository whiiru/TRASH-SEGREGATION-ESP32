# TRASH-SEGREGATION-ESP32

Sistema de segregação de lixo usando reconhecimento de imagem. O projeto treina um classificador binário (orgânico vs reciclável) com transfer learning (MobileNetV2), faz a inferência em um computador (webcam) e comunica a decisão a um ESP32 para acionar um servo motor que direciona o lixo.

---

## Dataset utilizado

O dataset usado como referência neste projeto é:
- Garbage Classification (Mendeley): https://data.mendeley.com/datasets/n3gtgm9jxj/2
---

## Estrutura esperada do dataset

O script de treinamento (`treinar_modelo.py`) espera a estrutura:

dataset/
- organico/
- reciclavel/

Cada pasta contém as imagens correspondentes.

---

## Dependências
Instale as dependências necessárias (recomendado criar um virtualenv):

pip install tensorflow pillow numpy scikit-learn matplotlib

Observação: Para treinar com GPU, instale a versão adequada do TensorFlow compatível com sua GPU/driver.

---

## Treinamento
1. Baixe e organize o dataset (veja seção anterior).
2. Abra treinar_modelo.py e ajuste a variável DATASET_DIR para apontar para a pasta dataset criada.
    Por exemplo: DATASET_DIR = r"/caminho/para/seu/projeto/dataset"
3. Execute:
   python treinar_modelo.py

O script faz:

- Data augmentation (com validation_split = 0.2);
- Treinamento da "cabeça" (somente as camadas finais);
- Fine-tuning das últimas N camadas da base MobileNetV2;
- EarlyStopping e ModelCheckpoint (arquivo melhor_modelo.keras).

Arquivos gerados:

- modelo_lixo.keras — modelo salvo ao final do script.
- melhor_modelo.keras — checkpoint do melhor modelo segundo val_loss.

O script também imprime o mapeamento interno de classes (gerador_treino.class_indices). Verifique esse dicionário e garanta que a ordem das classes bate com a configuração de inferência (ex.: ["organico","reciclavel"]).

---

##Integração com ESP32

Este repositório assume que:

A inferência é realizada em um computador (onde roda a webcam e o modelo).
O computador envia um comando ao ESP32.
O ESP32 recebe a decisão e aciona um servo motor para direcionar o lixo para o lado correspondente.
A implementação exata do firmware do ESP32 e o protocolo de comunicação ficam a cargo do usuário; inclua no ESP32 um handler simples que receba comandos como LEFT, RIGHT ou ORGANICO, RECICLAVEL e mova o servo para a posição desejada.

---

##Créditos e licença

Dataset: Garbage Classification — Mendeley Data (https://data.mendeley.com/datasets/n3gtgm9jxj/2)
Modelo base: MobileNetV2 (pretrained ImageNet)

--- 
