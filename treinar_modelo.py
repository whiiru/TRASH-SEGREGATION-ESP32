"""
Treinamento do Modelo — Sistema de Segregação de Lixo
============================================================
Usa transfer learning com MobileNetV2 para treinar um classificador
binário: organico vs reciclavel.

PASSO 1 — Organize seu dataset nesta estrutura de pastas:

    dataset/
        organico/
            img001.jpg
            img002.jpg
            ...
        reciclavel/
            img001.jpg
            img002.jpg
            ...

Se você for usar um dataset público como TrashNet ou o "Garbage
Classification" do Kaggle (que vêm com classes tipo cardboard, glass,
metal, paper, plastic, trash), você precisa REMAPEAR as pastas em organico 
e reciclavel

Instalação:
    pip install tensorflow pillow numpy scikit-learn matplotlib

Execução:
    python treinar_modelo.py
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ===================== CONFIGURAÇÕES =====================
DATASET_DIR = r""          # Localização da pasta com subpastas organico/ e reciclavel/
TAMANHO_IMAGEM = (224, 224)
BATCH_SIZE = 32

# Números de épocas são  TETOS máximos — o EarlyStopping vai parar
# antes disso automaticamente assim que val_loss parar de melhorar.
EPOCAS_CABECA = 20
EPOCAS_FINETUNE = 20

# Quantas camadas finais da base destravar no fine-tuning.
# Reduzido de 30 para 15: menos parâmetros livres = menos overfitting
# com datasets pequenos.
CAMADAS_FINETUNE = 15

# Quantas épocas sem melhora no val_loss até parar o treino
PACIENCIA_EARLY_STOPPING = 4

MODELO_SAIDA = "modelo_lixo.keras"
CHECKPOINT_PATH = "melhor_modelo.keras"  # sempre guarda o melhor val_loss visto até agora

# ===================== DATA AUGMENTATION =====================
datagen_treino = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    channel_shift_range=20.0,
    validation_split=0.2,
)

gerador_treino = datagen_treino.flow_from_directory(
    DATASET_DIR,
    target_size=TAMANHO_IMAGEM,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

gerador_validacao = datagen_treino.flow_from_directory(
    DATASET_DIR,
    target_size=TAMANHO_IMAGEM,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

print("Classes detectadas (nessa ordem o modelo vai gerar as saídas):")
print(gerador_treino.class_indices)
# Confirme que bate com CLASSES = ["organico", "reciclavel"]
# no servidor_classificador.py (ordem alfabética das pastas)

# ===================== MODELO (TRANSFER LEARNING) =====================
base_model = MobileNetV2(
    input_shape=(*TAMANHO_IMAGEM, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # congela a base inicialmente

modelo = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.5),  # aumentado de 0.3 para 0.5 (mais regularização)
    layers.Dense(2, activation="softmax"),  # 2 classes
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ===================== CALLBACKS ANTI-OVERFITTING =====================
# EarlyStopping: interrompe o treino quando val_loss para de melhorar,
# e restaura os pesos da melhor época (não fica com a última, que pode
# já estar overfitada).
parar_cedo = EarlyStopping(
    monitor="val_loss",
    patience=PACIENCIA_EARLY_STOPPING,
    restore_best_weights=True,
    verbose=1,
)

# ModelCheckpoint: salva em disco o melhor modelo visto até agora,
# como segurança extra além do restore_best_weights.
salvar_melhor = ModelCheckpoint(
    CHECKPOINT_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

print("\n=== Fase 1: treinando apenas a camada final ===")
modelo.fit(
    gerador_treino,
    validation_data=gerador_validacao,
    epochs=EPOCAS_CABECA,
    callbacks=[parar_cedo, salvar_melhor],
)

# ===================== FINE-TUNING (opcional, melhora acurácia) =====================
print(f"\n=== Fase 2: fine-tuning das últimas {CAMADAS_FINETUNE} camadas da base ===")
base_model.trainable = True
# Congela tudo menos as últimas N camadas (reduzido para diminuir overfitting)
for camada in base_model.layers[:-CAMADAS_FINETUNE]:
    camada.trainable = False

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # LR bem menor
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Callbacks recriados para monitorar a Fase 2 de forma independente
parar_cedo_finetune = EarlyStopping(
    monitor="val_loss",
    patience=PACIENCIA_EARLY_STOPPING,
    restore_best_weights=True,
    verbose=1,
)
salvar_melhor_finetune = ModelCheckpoint(
    CHECKPOINT_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

modelo.fit(
    gerador_treino,
    validation_data=gerador_validacao,
    epochs=EPOCAS_FINETUNE,
    callbacks=[parar_cedo_finetune, salvar_melhor_finetune],
)

# ===================== AVALIAÇÃO E EXPORTAÇÃO =====================
# Graças ao restore_best_weights=True, "modelo" já está com os pesos
# da época de MENOR val_loss, não da última época treinada.
perda, acuracia = modelo.evaluate(gerador_validacao)
print(f"\nAcurácia final de validação (melhor época, não a última): {acuracia * 100:.2f}%")
print(f"Perda final de validação: {perda:.4f}")

modelo.save(MODELO_SAIDA)
print(f"\nModelo salvo em: {MODELO_SAIDA}")
print(f"Checkpoint do melhor modelo também disponível em: {CHECKPOINT_PATH}")
print("Coloque o modelo final na mesma pasta do webcam_classificador.py")
