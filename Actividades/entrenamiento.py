import cv2
import os
import numpy as np

ruta_dataset = "dataset/autorizado"
caras = []
etiquetas = []

# Etiqueta de usuario autorizado
label = 0

# Lectura de imágenes
for archivo in os.listdir(ruta_dataset):
    ruta_imagen = os.path.join(ruta_dataset, archivo)
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    caras.append(imagen)
    etiquetas.append(label)

print("Entrenando modelo...")

#Entrenamiento de modelo
modelo = cv2.face.LBPHFaceRecognizer_create()
modelo.train(caras, np.array(etiquetas))

#Guardado
modelo.save("modelo_rostro.xml")
print("Modelo entrenado")