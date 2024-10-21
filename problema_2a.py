import matplotlib

matplotlib.use("TkAgg")
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargamos las respuestas Correctas del examen
respuesta_correcta = ["B", "A", "B", "D", "D", "C", "B", "A", "D", "B"]


# Función para preprocesar la imagen
def preprocess_image(img):
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh


# Cargar las imágenes
examenes = [
    cv2.imread("examen_1.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("examen_2.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("examen_3.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("examen_4.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("examen_5.png", cv2.IMREAD_GRAYSCALE),
]

# Procesamos todas las imágenes
for examen_idx, img in enumerate(examenes):
    print(f"\nProcesando Examen {examen_idx + 1}:")

    # Preprocesamos la imagen
    thresh = preprocess_image(img)

    # Encontramos los contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenamos los contornos por área (de mayor a menor)
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

    # Nos aseguramos de que se detectaron suficientes contornos
    if len(contours) < 2:
        print(f"No se encontraron suficientes contornos en el Examen {examen_idx + 1}.")
        continue

    # Extraer las 2 mitades
    segmented_images = []
    for i in range(2):
        x, y, w, h = cv2.boundingRect(contours[i])
        segmented_img = img[y : y + h, x : x + w]
        segmented_images.append(segmented_img)

        # Mostrar cada mitad segmentada con título del Examen
        plt.subplot(1, 2, i + 1)
        h = plt.imshow(segmented_img, cmap="gray", vmin=0, vmax=255)
        plt.title(f"Examen {examen_idx, i+1}")
        plt.axis("off")
        plt.colorbar(h)

    plt.show()

    preguntas_segmentadas = []
    # Ahora que tenemos las dos mitades, segmentamos cada mitad en 5 filas
    for idx, segmented_img in enumerate(segmented_images):
        height = segmented_img.shape[0]  # Altura de la imagen
        step = height // 5  # Dividimos la altura en 5 partes

        print(f"\nSegmentando Examen {examen_idx + 1} - Mitad {idx+1}:")
        for i in range(5):
            # Extraemos cada fila
            question_segment = segmented_img[i * step : (i + 1) * step, :]
            height, width = question_segment.shape
            segmentadox2 = question_segment[10 : height - 10, 10 : width - 10]
            preguntas_segmentadas.append(segmentadox2)
            # Mostrar cada segmento de pregunta con título del Examen y la Pregunta

            plt.subplot(5, 2, i * 2 + (idx + 1))
            h = plt.imshow(segmentadox2, cmap="gray", vmin=0, vmax=255)
            plt.title(f"Examen {examen_idx + 1} - Pregunta {i + 1 + (idx * 5)}")
            plt.axis("off")
            plt.colorbar(h)

    plt.show()

# Localizar líneas y recortar
renglones = []
for idx, question_img in enumerate(preguntas_segmentadas):
    # Convertir la imagen a binaria
    question_thresh = preprocess_image(question_img)

    # Aplicar la Transformada de Hough para detectar líneas
    lines = cv2.HoughLinesP(
        question_thresh, 1, np.pi / 180, threshold=50, minLineLength=20
    )
    # Si se encuentran líneas, procesarlas
    if lines is not None:
        for line in lines:
            print(line)
            x1, y1, x2, y2 = line[0]
            # Calcular la posición y de la línea media
            line_y = (y1 + y2) // 2
            # Definir el recorte
            top = max(0, line_y - 14)  # 14 píxeles hacia arriba de la línea
            bottom = min(
                question_img.shape[0], line_y - 2
            )  # Dos pixeles hacia arriba de la línea

            # Recortar la imagen
            cropped_question = question_img[top:bottom, x1 : x2 + 10]
            renglones.append(cropped_question)
            # Mostrar la pregunta recortada

            plt.figure()
            plt.imshow(cropped_question, cmap="gray", vmin=0, vmax=255)
            plt.title(f"Recorte Examen {examen_idx + 1} - Pregunta {idx + 1}")
            plt.axis("off")
            plt.show()

    else:
        print(
            f"No se encontraron líneas en la Pregunta {idx + 1} del Examen {examen_idx + 1}."
        )


def preprocess_image(img):
    # Verificar si la imagen ya está en escala de grises
    if len(img.shape) == 3:  # Si tiene 3 dimensiones, entonces es una imagen en BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  # Ya está en escala de grises

    # Binarizar la imagen (umbral adaptativo si hay variaciones de iluminación)
    _, binaria = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return binaria


# Función para detectar la letra basada en los contornos y segmentos
def detectar_respuesta(renglon_img):
    binaria = preprocess_image(renglon_img)

    # Comprobar si hay píxeles blancos (potencial marca)
    if cv2.countNonZero(binaria) == 0:
        return "Sin respuesta"

    # Encontrar contornos y jerarquía
    contornos, jerarquia = cv2.findContours(
        binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    # Contar contornos cerrados (huecos internos)
    huecos = 0
    for i in range(len(contornos)):
        # Verificar si el contorno es un hijo (hueco interno)
        if jerarquia[0][i][3] != -1:
            huecos += 1

    # Analizar según el número de huecos
    if contar_segmentos(binaria) == 1:
        return "A"
    elif contar_segmentos(binaria) == 2:
        return "D"
    elif contar_segmentos(binaria) == 3:
        return "B"
    elif huecos == 0:
        # Si no hay huecos pero hay píxeles, es C
        return "C"
    else:
        # En caso de detección inesperada
        return "Error"


# Función para contar segmentos horizontales en una imagen binaria
def contar_segmentos(img_binaria):
    # Detectar líneas horizontales que atraviesen la imagen
    altura, ancho = img_binaria.shape
    print(altura)
    segmentaciones = []
    for y in range(0, altura, altura // 10):
        linea = img_binaria[y : y + 1, :]  # Tomar una fila de píxeles
        if (
            cv2.countNonZero(linea) > 4
        ):  # Si hay píxeles blancos, cuenta como un segmento
            segmentaciones.append(linea)
    return len(segmentaciones)


# Evaluar las respuestas de una lista de imágenes de renglones
respuestas_detectadas = []
for idx, renglon in enumerate(renglones):
    letra_marcada = detectar_respuesta(renglon)
    print(letra_marcada)
    respuestas_detectadas.append(letra_marcada)
    if letra_marcada == respuesta_correcta[idx]:
        print(f"Pregunta {idx + 1}: OK")
    else:
        print(f"Pregunta {idx + 1}: MAL")
