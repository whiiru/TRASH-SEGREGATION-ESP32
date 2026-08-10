/*
 * Sistema de Segregação Automática de Lixo
 * ESP32 — Controlador do Servo (via WiFi)
 *
 * O ESP32 sobe um servidor HTTP simples
 * na rede local e espera receber comandos do script Python que roda
 * no PC/Raspberry Pi (onde a câmera USB está conectada).
 *
 * Endpoint:
 *   GET http://<IP_DO_ESP32>/mover?classe=organico
 *   GET http://<IP_DO_ESP32>/mover?classe=reciclavel
 *
 * Bibliotecas necessárias:
 *  - ESP32Servo (by Kevin Harrington / madhephaestus)
 *  - (WiFi.h e WebServer.h já vêm com o core do ESP32)
 *
 * Board: qualquer ESP32 genérico (DevKit, etc.)
 */

#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>

// ===================== CONFIGURAÇÕES =====================
const char* WIFI_SSID     = ""; //Altere baseado no nome e senha da rede 
const char* WIFI_PASSWORD = ""; //Wifi TEM que ser 2.4gGhz, o ESP32 não consegue se conectar a redes 5G

const int PINO_SERVO = 13; // qualquer do ESP32

const int ANGULO_ORGANICO     = 0; //divisões de lixo 
const int ANGULO_RECICLAVEL = 90;

WebServer servidor(80); //abre o servidor na porta 80
Servo servoLixo;

void tratarMover() {
  if (!servidor.hasArg("classe")) { 
    servidor.send(400, "application/json", "{\"erro\":\"parametro 'classe' ausente\"}");
    return;
  }
//Envia para o servidor a classe reconhecida 
  String classe = servidor.arg("classe");
  if (classe == "ORGANICO") {
    Serial.println(">> Movendo servo para 0° (organico)"); //0 graus se organico, 90 se reciclável 
    servoLixo.write(ANGULO_ORGANICO);
    servidor.send(200, "application/json", "{\"status\":\"ok\",\"angulo\":0}");
  } else if (classe == "RECICLAVEL") {
    Serial.println(">> Movendo servo para 90° (reciclavel)");
    servoLixo.write(ANGULO_RECICLAVEL);
    servidor.send(200, "application/json", "{\"status\":\"ok\",\"angulo\":90}");
  } else { 
    servidor.send(400, "application/json", "{\"erro\":\"classe invalida\"}");
  }
}

//Notificação do servidor online
void tratarSaude() {
  servidor.send(200, "application/json", "{\"status\":\"online\"}");
}

void setup() {
  //Inicia a porta serial na frequência 115200 (alterar na Arduino IDE)
  Serial.begin(115200);
  delay(1000);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("Conectando ao WiFi %s", WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi conectado!");
  Serial.print("IP do ESP32 (use este IP no script Python): "); //IP do servidor do esp32
  Serial.println(WiFi.localIP()); 

  servoLixo.setPeriodHertz(50);
  servoLixo.attach(PINO_SERVO, 500, 2400);
  servoLixo.write(ANGULO_ORGANICO); // posição inicial

  servidor.on("/mover", tratarMover);
  servidor.on("/saude", tratarSaude);
  servidor.begin();
  Serial.println("Servidor HTTP iniciado na porta 80.");
}

void loop() {
  servidor.handleClient();
}
