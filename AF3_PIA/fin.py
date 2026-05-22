import cv2
import serial
import time

#Conexión con ESP32
esp32 = serial.Serial('COM3', 115200)
time.sleep(2)

#Detección y carga de modelo

clasificador = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
modelo = cv2.face.LBPHFaceRecognizer_create()
modelo.read("modelo_rostro.xml")

#Uso de cámara
camara = cv2.VideoCapture(1)
time.sleep(2)

#Procesamiento y funcionamiento de caja
caja_abierta = False
tiempo_apertura = 15
momento_apertura = 0

#Hacemos condiciones para que se activen servos
frames_autorizados = 0
frames_requeridos = 30

print("Sistema iniciado")

while True:
    ret, frame = camara.read()
    if not ret:
        break
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detección de caras
    caras = clasificador.detectMultiScale(
        gris,
        scaleFactor=1.2,
        minNeighbors=7
    )
    autorizado = False
    for (x, y, w, h) in caras:
        rostro = gris[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (200, 200))
        label, confianza = modelo.predict(rostro)
        texto = "NO AUTORIZADO"
        color = (0, 0, 255)

        if confianza < 45:

            autorizado = True

            texto = "AUTORIZADO"
            color = (0, 255, 0)

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            color,
            2
        )
        cv2.putText(
            frame,
            f"{texto} ({int(confianza)})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    #Confirmación de usuario

    if autorizado and not caja_abierta:

        frames_autorizados += 1
    else:

        if not caja_abierta:
            frames_autorizados = 0

    progreso = int((frames_autorizados / frames_requeridos) * 100)
    if progreso > 100:
        progreso = 100

    cv2.putText(
        frame,
        f"VERIFICANDO: {progreso}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    #Apertura de caja

    if frames_autorizados >= frames_requeridos and not caja_abierta:

        esp32.write(b'1')

        caja_abierta = True

        momento_apertura = time.time()

        print("CAJA ABIERTA")

    #Se pone un contador para volver a cerrar la caja de forma automática

    if caja_abierta:

        tiempo_restante = int(
            tiempo_apertura - (time.time() - momento_apertura)
        )
        cv2.putText(
            frame,
            f"CIERRE EN: {max(tiempo_restante, 0)}s",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )
        
        if time.time() - momento_apertura >= tiempo_apertura:

            esp32.write(b'0')

            caja_abierta = False

            frames_autorizados = 0

            print("CAJA CERRADA")

    cv2.imshow("Reconocimiento Facial", frame)

    tecla = cv2.waitKey(1)

    # Establecemos tecla esc para cerrar
    if tecla == 27:
        break
esp32.write(b'0')
camara.release()
cv2.destroyAllWindows()
esp32.close()