"""
Classificador com Câmera USB — Sistema de Segregação de Lixo
============================================================
Roda no PC ou Raspberry Pi onde a câmera USB está conectada.
Captura frames em tempo real, classifica com o modelo treinado
(ver treinar_modelo.py) e envia comando HTTP pro ESP32 mover o servo.

Instalação:
    pip install opencv-python tensorflow numpy requests

Execução:
    python webcam_classificador.py

Pressione 'q' na janela de vídeo para sair.
"""

import time
import cv2
import numpy as np
import requests
import tensorflow as tf #Obs: Tensorflow está disponível para python 3.13

# ===================== CONFIGURAÇÕES =====================
INDICE_CAMERA = 0          # 0 = primeira câmera USB detectada; troque se tiver mais de uma
MODELO_PATH = "melhor_modelo.keras" # Modelo usado, deixe na mesma pasta do programa
TAMANHO_IMAGEM = (224, 224)
CLASSES = ["organico", "reciclavel"]  # ordem alfabética das pastas do treino

ESP32_IP = "192.168.*.***"   # IP impresso no Monitor Serial do ESP32 TROQUE AQUI 
ESP32_URL = f"http://{ESP32_IP}/mover"

CONFIANCA_MINIMA = 0.65     # abaixo disso, não move o servo (evita decisão errada)
INTERVALO_CLASSIFICACAO = 2.0  # segundos entre classificações
COOLDOWN_APOS_MOVER = 3.0   # segundos de pausa após mover o servo (evita reclassificar o mesmo objeto)

# ===================== CARREGAR MODELO =====================
print("Carregando modelo...")
modelo = tf.keras.models.load_model(MODELO_PATH)
print("Modelo carregado.")


def preprocessar(frame_bgr: np.ndarray) -> np.ndarray: # Processa a imagem lida, usa a biblioteca cv2
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_redimensionado = cv2.resize(frame_rgb, TAMANHO_IMAGEM)
    array = frame_redimensionado.astype(np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def enviar_comando_servo(classe: str) -> bool: #Envia o comando para o servidor
    try:
        resposta = requests.get(ESP32_URL, params={"classe": classe}, timeout=3)
        if resposta.status_code == 200:
            print(f"[ESP32] comando enviado com sucesso: {classe}")
            return True
        else:
            print(f"[ESP32] erro HTTP {resposta.status_code}: {resposta.text}")
    except requests.exceptions.RequestException as e:
        print(f"[ESP32] falha na comunicação: {e}")
    return False


def main():
    captura = cv2.VideoCapture(INDICE_CAMERA) #Abertura da camera
    if not captura.isOpened():
        print("Não foi possível abrir a câmera USB. Verifique o INDICE_CAMERA.")
        return

    print("Câmera iniciada. Pressione 'q' para sair.")
    ultima_classificacao = 0.0
    pausado_ate = 0.0

    while True:
        ok, frame = captura.read()
        if not ok:
            print("Falha ao ler frame da câmera.")
            break

        agora = time.time()
        classe_atual = None
        confianca_atual = None

        if agora >= pausado_ate and (agora - ultima_classificacao) >= INTERVALO_CLASSIFICACAO:
            ultima_classificacao = agora

            entrada = preprocessar(frame)
            predicao = modelo.predict(entrada, verbose=0)[0]
            indice = int(np.argmax(predicao))
            confianca = float(predicao[indice])
            classe_atual = CLASSES[indice]
            confianca_atual = confianca

            print(f"Predição: {classe_atual} (confiança: {confianca:.2f})")

            if confianca >= CONFIANCA_MINIMA:
                sucesso = enviar_comando_servo(classe_atual)
                if sucesso:
                    pausado_ate = agora + COOLDOWN_APOS_MOVER
            else:
                print("Confiança baixa, servo não foi movido.")

        # ===== Overlay de debug na janela de vídeo =====
        texto = f"{classe_atual} ({confianca_atual:.2f})" if classe_atual else "aguardando..."
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Segregacao de Lixo - Camera USB", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
