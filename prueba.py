import cv2
import numpy as np

# A Define el umbral para considerar una opción marcada
fill_threshold = 500  # Ajusta este valor según sea necesario


def is_filled(segment):
    filled_pixels = cv2.countNonZero(segment)
    return filled_pixels > fill_threshold


def grade_exam(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    correct_answers = ["C", "B", "A", "D", "B", "B", "A", "B", "D", "D"]
    question_height, option_width = 30, 30  # ajustar según tu diseño

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = ["MAL"] * 10
    detected_answers = [[] for _ in range(10)]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # ajustar filtro de tamaño según sea necesario
            question_idx = y // question_height
            answer_idx = x // option_width
            if (
                0 <= question_idx < len(detected_answers) and 0 <= answer_idx < 4
            ):  # verificando índices válidos
                if is_filled(thresh[y : y + h, x : x + w]):
                    detected_answers[question_idx].append("ABCD"[answer_idx])

    for idx, answers in enumerate(detected_answers):
        if len(answers) == 1 and answers[0] == correct_answers[idx]:
            results[idx] = "OK"

    for i, result in enumerate(results):
        print(f"Pregunta {i + 1}: {result}")


grade_exam("examen_1.png")

# -------------------------------------------------------------------------
# B
import re


def validate_header(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Asegúrate de tener lógica para extraer los campos del encabezado. Aquí solo usamos ejemplos.
    name = "Nombre Extraído"
    date = "20240101"  # Ajusta según tu lógica de extracción
    class_ = "A"  # Ajusta según tu lógica de extracción

    name_ok = len(name.split()) >= 2 and len(name) <= 25
    date_ok = re.match(r"^\d{8}$", date)
    class_ok = re.match(r"^[A-Za-z0-9/]{1}$", class_)

    print(f'Name: {"OK" if name_ok else "MAL"}')
    print(f'Date: {"OK" if date_ok else "MAL"}')
    print(f'Class: {"OK" if class_ok else "MAL"}')


validate_header("examen_1.png")

# ---------------------------------------------------------------------------
# C
exam_files = ["examen_1.png", "examen_2.png", "examen_3.png"]

for exam_file in exam_files:
    print(f"\nEvaluando {exam_file}")
    grade_exam(exam_file)
    validate_header(exam_file)

# ------------------------------------------------------------------------------
# D

import cv2


def generate_output_image(exam_files):
    approved = []
    disapproved = []

    for exam_file in exam_files:
        result_image = cv2.imread(exam_file)
        # Asumiendo que tienes lógica para determinar el número de respuestas correctas
        correct_answers = 6  # Reemplaza con el recuento real
        if correct_answers >= 6:
            approved.append(result_image)
        else:
            disapproved.append(result_image)

    output_image = np.zeros(
        (1000, 1000, 3), dtype=np.uint8
    )  # ajustar según sea necesario
    # Combinar imágenes en output_image y anotar como "Aprobado" o "Desaprobado"

    cv2.imwrite("output_image.png", output_image)


generate_output_image(exam_files)
