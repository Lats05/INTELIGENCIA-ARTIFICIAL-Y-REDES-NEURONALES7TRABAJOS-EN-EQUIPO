import cv2
import os

# Carpeta de dataset
carpeta = "dataset/autorizado"
os.makedirs(carpeta, exist_ok=True)

# Contador de imágenes
contador = len(os.listdir(carpeta))
print(f"Imágenes actuales: {contador}")

# Clasificador
clasificador = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Funcionamiento
camara = cv2.VideoCapture(1)
cv2.waitKey(1000)
print("ESPACIO = guardar cara")
print("ESC = salir")

while True:

    ret, frame = camara.read()

    if not ret:
        print("Error al abrir cámara")
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    caras = clasificador.detectMultiScale(
        gris,
        scaleFactor=1.2,
        minNeighbors=7
    )
    for (x, y, w, h) in caras:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
    cv2.putText(
        frame,
        f"Fotos guardadas: {contador}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )
    cv2.imshow("Captura de Cara", frame)
    tecla = cv2.waitKey(1)
    if tecla == 27:
        break
    elif tecla == 32:
        if len(caras) == 0:
            print("No se detectó rostro")
            continue
        for (x, y, w, h) in caras:
            rostro = gris[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (200, 200))
            nombre = f"{carpeta}/rostro_{contador}.jpg"
            cv2.imwrite(nombre, rostro)
            print(f"Guardado: {nombre}")
            contador += 1
camara.release()
cv2.destroyAllWindows()