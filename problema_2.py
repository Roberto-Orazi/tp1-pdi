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


def correct_date(region_date):

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
exam_name = []
for i, examen in enumerate(examenes, start=1):
    header_image = find_header_region(examen)

    binary_header = preprocess_image(header_image)

    region_name, region_date, region_class = segment_header(binary_header)

    exam_name.append([i, region_name])

    nom = correct_name(region_name)
    date = correct_date(region_date)
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


import matplotlib

matplotlib.use("TkAgg")
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargamos las respuestas correctas
respuesta_correcta = ["B", "A", "B", "D", "D", "C", "B", "A", "D", "B"]


# Función para preprocesar la imagen
def preprocess_image(img):
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh


# Función para detectar la letra basada en los contornos y segmentos
def detect_answer(renglon_img):
    # Preprocesar la imagen
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
    if segments_count(binaria) == 1:
        return "A"
    elif segments_count(binaria) == 2:
        return "D"
    elif segments_count(binaria) == 3:
        return "B"
    elif huecos == 0:
        # Si no hay huecos pero hay píxeles, es C
        return "C"
    else:
        # En caso de detección inesperada
        return "Error"


# Función para contar segmentos horizontales en una imagen binaria
def segments_count(img_binaria):
    # Detectar líneas horizontales que atraviesen la imagen
    height, ancho = img_binaria.shape
    segmentations = []
    for y in range(
        0, height, height // 10
    ):  # Dividir la imagen en 10 secciones horizontales
        line = img_binaria[y : y + 1, :]  # Tomar una fila de píxeles
        if (
            cv2.countNonZero(line) > 4
        ):  # Si hay píxeles blancos, cuenta como un segmento
            segmentations.append(line)
    return len(segmentations)


# Cargar las imágenes
examenes = [
    cv2.imread("examen_1.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("examen_2.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("examen_3.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("examen_4.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("examen_5.png", cv2.IMREAD_GRAYSCALE),
]

# Procesamos todas las imágenes
respuestas_final = {}

for exam_idx, img in enumerate(examenes):
    # print(f"\nProcesando Examen {examen_idx + 1}:")

    # Preprocesamos la imagen
    thresh = preprocess_image(img)

    # Encontramos los contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenamos los contornos por área (de mayor a menor)
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

    # Asegurarse de que se detectaron suficientes contornos
    if len(contours) < 2:
        print(f"No se encontraron suficientes contornos en el Examen {exam_idx + 1}.")
        continue

    # Extraer las 2 mitades
    segmented_images = []
    for i in range(2):
        x, y, w, h = cv2.boundingRect(contours[i])
        segmented_img = img[y : y + h, x : x + w]
        segmented_images.append(segmented_img)

    # Segmentación de las preguntas
    segmented_questions = []
    for idx, segmented_img in enumerate(segmented_images):
        height = segmented_img.shape[0]
        step = height // 5

        print(f"\nSegmentando Examen {exam_idx + 1} - Mitad {idx + 1}:")
        for i in range(5):

            segment_question = segmented_img[i * step : (i + 1) * step, :]
            height, width = segment_question.shape
            doble_segment = segment_question[5 : height - 10, 10 : width - 10]
            segmented_questions.append(doble_segment)

    # Localizar líneas y recortar
    renglones = []
    for idx, question_img in enumerate(segmented_questions):
        # Convertir la imagen a binaria
        question_thresh = preprocess_image(question_img)

        # Aplicar la Transformada de Hough para detectar líneas
        lines = cv2.HoughLinesP(
            question_thresh, 1, np.pi / 180, threshold=50, minLineLength=20
        )
        # Si se encuentran líneas, procesarlas
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calcular la posición y de la línea media
                line_y = (y1 + y2) // 2
                # Definir el recorte
                top = max(0, line_y - 14)  # 14 píxeles hacia arriba de la línea
                bottom = min(
                    question_img.shape[0], line_y - 2
                )  # Dos pixeles hacia arriba de la línea

                # Recortar la imagen
                cropped_question = question_img[top:bottom, x1 : x2 + 15]
                renglones.append(cropped_question)
        else:
            print(
                f"No se encontraron líneas en la Pregunta {idx + 1} del Examen {exam_idx + 1}."
            )

    # Evaluar las respuestas de una lista de imágenes de renglones
    res_OK_MAL = []
    respuestas_detectadas = []

    for idx, renglon in enumerate(renglones):
        letra_marcada = detect_answer(renglon)
        respuestas_detectadas.append(letra_marcada)
        # Mostrar el resultado
        if letra_marcada == respuesta_correcta[idx]:
            res_OK_MAL.append("OK")
        else:
            res_OK_MAL.append("MAL")

    respuestas_final[exam_idx + 1] = res_OK_MAL

approved = {}
rejected = {}
for i in range(1, 6):
    nota = 0
    for j in respuestas_final[i]:
        if j == "OK":
            nota += 1
    if nota >= 6:
        approved[i] = nota
    else:
        rejected[i] = nota


# nombres de los examenes
exam_name

# notas de los examenes
approved
rejected

# Creamos una lista de resultados combinada con aprobados y desaprobados
results = []
for exam_number, name in exam_name:
    nota = None
    estado = None

    # Buscar la nota en el diccionario de aprobados
    if exam_number in approved:
        nota = approved[exam_number]
        estado = "Aprobado"
    # Buscar la nota en el diccionario de desaprobados
    elif exam_number in rejected:
        nota = rejected[exam_number]
        estado = "Desaprobado"

    # Añadir el resultado a la lista final
    results.append((name, exam_number, nota, estado))


# Función para generar la imagen de resultados
def generate_result_image(results, output_filename="resultados_examenes.png"):
    # Definir las dimensiones de cada bloque de nombre
    block_height, block_width = 110, 310  # Ajustar dimensiones para incluir el borde

    # Crear una imagen en blanco para almacenar todos los nombres
    img_results_height = block_height * len(results)
    img_results = np.ones((img_results_height, block_width, 3), dtype=np.uint8) * 255

    # Colores para diferenciar aprobados y desaprobados
    approved_color = (0, 255, 0)  # Verde
    rejected_color = (0, 0, 255)  # Rojo

    # Iterar sobre cada resultado y agregarlo a la imagen
    for idx, (name, exam_number, _, estado) in enumerate(results):
        # Extraer la imagen del nombre (crop)
        region_name = cv2.cvtColor(exam_name[exam_number - 1][1], cv2.COLOR_GRAY2BGR)

        # Redimensionar el nombre para que coincida con el bloque de la imagen de salida
        resized_name = cv2.resize(region_name, (block_width - 10, block_height - 10))

        # Determinar el color del borde según el estado (aprobado/desaprobado)
        color = approved_color if estado == "Aprobado" else rejected_color

        # Dibujar un borde alrededor del bloque de nombre
        bordered_name = cv2.copyMakeBorder(
            resized_name, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=color
        )

        # Calcular la posición donde se pegará el bloque en la imagen de resultados
        y_start = idx * block_height
        y_end = y_start + block_height

        # Agregar el bloque de nombre a la imagen de resultados
        img_results[y_start:y_end, :] = bordered_name

    # Guardar la imagen de resultados
    cv2.imwrite(output_filename, img_results)
    print(f"Imagen de resultados guardada como {output_filename}")


# Llamar a la función para generar la imagen de resultados
generate_result_image(results)
