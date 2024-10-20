import matplotlib

matplotlib.use("TkAgg")
import cv2
import numpy as np
import matplotlib.pyplot as plt

examen_1 = cv2.imread("examen_1.png", cv2.IMREAD_GRAYSCALE)
examen_2 = cv2.imread("examen_2.png", cv2.IMREAD_GRAYSCALE)
examen_3 = cv2.imread("examen_3.png", cv2.IMREAD_GRAYSCALE)
examen_4 = cv2.imread("examen_4.png", cv2.IMREAD_GRAYSCALE)
examen_5 = cv2.imread("examen_5.png", cv2.IMREAD_GRAYSCALE)


def preprocess_image(img):
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh


def find_header_region(img):
    # Cropeamos la imagen en un 15%
    header_height = int(img.shape[0] * 0.15)
    header = img[:header_height, :]
    return header


def extract_text_above_line(img):
    # Preprocesamos la imagen
    header = find_header_region(img)
    thresh = preprocess_image(header)

    # Buscamos los contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # vamos a hacer de cuenta que el ultimo contorno es la linea
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

    if len(contours) == 0:
        print("No contours found.")
        return None

    line_contour = contours[0]

    x, y, w, h = cv2.boundingRect(line_contour)

    # Aca seleccionamos la seccion de la linea para arriba
    text_region = header[:y, :]

    return text_region


text_above_line = extract_text_above_line(examen_1)

plt.imshow(text_above_line, cmap="gray")
plt.axis("off")
plt.show()
