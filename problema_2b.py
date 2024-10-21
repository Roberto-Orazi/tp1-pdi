import cv2
import numpy as np
import matplotlib.pyplot as plt
import re


# Función para preprocesar la imagen (binaria/umbral)
def preprocess_image(img):
    if img is None:
        return None
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh


# Función para eliminar los textos "Name:", "Date:", "Class:"
def remove_text_labels(region):
    if region is None or region.size == 0:
        return None

    upper_region = region[: int(region.shape[0] * 0.3), :]
    if upper_region.size == 0:
        return region
    _, binary_region = cv2.threshold(upper_region, 150, 255, cv2.THRESH_BINARY_INV)

    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    detected_text = cv2.morphologyEx(
        binary_region, cv2.MORPH_CLOSE, text_kernel, iterations=2
    )

    contours, _ = cv2.findContours(
        detected_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 5 and w < region.shape[1] * 0.5:
            region[y : y + h, x : x + w] = 255

    return region


# Función para encontrar la región del encabezado
def find_header_region(img):
    header_height = int(img.shape[0] * 0.15)
    header = img[:header_height, :]
    return header


# Función para segmentar el encabezado en las 3 partes
def segment_header(header):
    width = header.shape[1]
    name_start = int(width * 0.10)
    name_end = int(width * 0.45)
    date_start = int(width * 0.52)
    date_end = int(width * 0.65)
    class_start = int(width * 0.75)
    class_end = width

    region_name = header[:, name_start:name_end]
    region_date = header[:, date_start:date_end]
    region_class = header[:, class_start:class_end]

    region_name = remove_text_labels(region_name)
    region_date = remove_text_labels(region_date)
    region_class = remove_text_labels(region_class)

    region_name = remove_underline(region_name)
    region_date = remove_underline(region_date)
    region_class = remove_underline(region_class)

    return region_name, region_date, region_class


# Función para detectar las líneas horizontales (guiones bajos) y recortar la parte superior
def remove_underline(region):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(
        region, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )

    contours, _ = cv2.findContours(
        detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        underline_y = min([cv2.boundingRect(contour)[1] for contour in contours])
        return region[:underline_y, :]

    return region


def correct_data(region_date):

    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
        region_date
    )

    output = cv2.cvtColor(region_date, cv2.COLOR_GRAY2BGR)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

    """    plt.figure(2)
    plt.imshow(output) plt.title("Componentes Conectados (Palabras Detectadas)") plt.show()"""

    letter_count = 0
    for i in range(1, num_labels):
        letter_count += 1

    cont = 0
    word_count = 1
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aux = x
        if i == 1:
            cont = aux
        else:
            pixel = aux - cont
            cont = aux
            if pixel >= 16:
                word_count += 1
    return letter_count == 8 and word_count == 1


def correct_class(region_class):

    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
        region_class
    )

    output = cv2.cvtColor(region_class, cv2.COLOR_GRAY2BGR)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
    """    plt.figure(2)
    plt.imshow(output) plt.title("Componentes Conectados (Palabras Detectadas)") plt.show()"""
    letter_count = 0
    for i in range(1, num_labels):
        letter_count += 1

    return letter_count == 1


def correct_name(region_name):
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
        region_name
    )

    # Mostrar los componentes conectados
    output = cv2.cvtColor(region_name, cv2.COLOR_GRAY2BGR)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

    """    plt.figure(2)
    plt.imshow(output) plt.title("Componentes Conectados (Palabras Detectadas)") plt.show()"""

    cont = 0
    word_count = 1
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aux = x
        if i == 1:
            cont = aux
        else:
            pixel = aux - cont
            cont = aux
            if pixel >= 16:
                word_count += 1
    return word_count == 2


# Cargamos las imágenes de examen
examen_1 = cv2.imread("examen_1.png", cv2.IMREAD_GRAYSCALE)
examen_2 = cv2.imread("examen_2.png", cv2.IMREAD_GRAYSCALE)
examen_3 = cv2.imread("examen_3.png", cv2.IMREAD_GRAYSCALE)
examen_4 = cv2.imread("examen_4.png", cv2.IMREAD_GRAYSCALE)
examen_5 = cv2.imread("examen_5.png", cv2.IMREAD_GRAYSCALE)
examenes = [examen_1, examen_2, examen_3, examen_4, examen_5]

for i, examen in enumerate(examenes, start=1):
    header_image = find_header_region(examen)

    binary_header = preprocess_image(header_image)

    region_name, region_date, region_class = segment_header(binary_header)

    nom = correct_name(region_name)
    date = correct_data(region_date)
    clas = correct_class(region_class)

    print(f"Examen numero {i}")

    if nom:
        print(f"Name: OK")
    else:
        print(f"Name: MAL")

    if date:
        print(f"Date: OK")
    else:
        print(f"Date: MAL")

    if clas:
        print(f"Class: OK")
    else:
        print(f"Class: MAL")
