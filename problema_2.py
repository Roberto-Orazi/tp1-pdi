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


# Función para preprocesar la imagen (binaria)/umbral
def preprocess_image(img):
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh


# Función para encontrar la región del encabezado (cortamos el 15% superior de la imagen)
def find_header_region(img):
    header_height = int(img.shape[0] * 0.15)
    header = img[:header_height, :]
    return header


# Función para extraer la región del texto por encima de la línea
def extract_text_above_line(img):
    # Preprocesamos la imagen
    header = find_header_region(img)
    thresh = preprocess_image(header)

    # Encontramos los contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Suponemos que el contorno más grande es la línea
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

    if len(contours) == 0:
        print("No contours found.")
        return None

    # Tomamos el contorno de la línea
    line_contour = contours[0]

    # Obtener las coordenadas de la línea
    x, y, w, h = cv2.boundingRect(line_contour)

    # Seleccionar la región del texto por encima de la línea
    text_region = header[:y, :]

    return text_region


# Función para segmentar el texto en 3 partes (Name, Date, Class)
def segment_header(header):
    # Ancho de la imagen
    width = header.shape[1]

    # Limites porcentuales
    name_end = int(width * 0.45)  # 0% a 45% para Name
    date_end = int(width * 0.65)  # 45% a 65% para Date
    class_end = width  # 65% a 100% para Class

    region_name = header[:, :name_end]
    region_date = header[:, name_end:date_end]
    region_class = header[:, date_end:class_end]

    return region_name, region_date, region_class


# Función para verificar los campos
def validate_fields(region_name, region_date, region_class):
    # Comprobar Name
    if cv2.countNonZero(region_name) > 0:
        # Convertir a texto y comprobar longitud
        name_content = cv2.countNonZero(region_name)
        if name_content > 25:
            name_status = "MAL"
        else:
            name_status = "OK"
    else:
        name_status = "MAL"

    # Comprobar Date
    if cv2.countNonZero(region_date) > 0:
        date_status = "OK"
    else:
        date_status = "MAL"

    # Comprobar Class
    if cv2.countNonZero(region_class) > 0:
        class_status = "OK"
    else:
        class_status = "MAL"

    return name_status, date_status, class_status


# Extraer la región del encabezado de la imagen
text_above_line = extract_text_above_line(examen_4)

if text_above_line is not None:
    # Segmentar en Name, Date y Class
    region_name, region_date, region_class = segment_header(text_above_line)

    # Validar los campos
    name_status, date_status, class_status = validate_fields(
        region_name, region_date, region_class
    )

    # Mostrar resultados
    print(f"Name: {name_status}")
    print(f"Date: {date_status}")
    print(f"Class: {class_status}")

    # Mostrar las tres regiones segmentadas
    plt.subplot(1, 3, 1)
    plt.imshow(region_name, cmap="gray")
    plt.title("Name")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(region_date, cmap="gray")
    plt.title("Date")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(region_class, cmap="gray")
    plt.title("Class")
    plt.axis("off")

    plt.show()
else:
    print("No se pudo extraer el texto correctamente.")
