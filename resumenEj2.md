# Descripción del Código: Procesamiento y Evaluación de Exámenes Digitales

Este código procesa automáticamente imágenes de exámenes en formato digital, detectando y evaluando encabezados y
respuestas marcadas. Luego, genera una imagen de resumen de resultados.

## Funciones del Código

### 1. Función `preprocess_image(img)`

```python
Copiar código
def preprocess_image(img):
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh
```

**Propósito:** Preprocesar la imagen para convertirla en binaria invertida.

**Detalles:**
- Convierte la imagen a binaria usando un umbral de 150.
- Utiliza `cv2.THRESH_BINARY_INV` para invertir la imagen (blanco y negro).
- Esto facilita la detección de texto o marcas en el examen.

### 2. Función `remove_text_labels(region)`

```python
Copiar código
def remove_text_labels(region):
    if region is None or region.size == 0:
        return None

    upper_region = region[: int(region.shape[0] * 0.3), :]
    if upper_region.size == 0:
        return region

    _, binary_region = cv2.threshold(upper_region, 150, 255, cv2.THRESH_BINARY_INV)
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    detected_text = cv2.morphologyEx(binary_region, cv2.MORPH_CLOSE, text_kernel, iterations=2)

    contours, _ = cv2.findContours(detected_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 5 and w < region.shape[1] * 0.5:
            region[y : y + h, x : x + w] = 255

    return region
```

**Propósito:** Eliminar etiquetas de texto como "Name:", "Date:", "Class:" en la parte superior de la región.

**Detalles:**
- Se selecciona la parte superior de la región para detectar y eliminar texto usando morfología de cierre
  (`cv2.MORPH_CLOSE`).
- Se identifican los contornos de los textos y se eliminan pintándolos de blanco.

### 3. Función `find_header_region(img)`

```python
Copiar código
def find_header_region(img):
    header_height = int(img.shape[0] * 0.15)
    header = img[:header_height, :]
    return header
```

**Propósito:** Extraer la región del encabezado de la imagen.

**Detalles:**
- Se corta la imagen al 15% de su altura total para obtener la sección del encabezado.

### 4. Función `segment_header(header)`

```python
Copiar código
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
```

**Propósito:** Dividir el encabezado en tres regiones: nombre, fecha y clase.

**Detalles:**
- Se segmentan las regiones según proporciones fijas de la anchura del encabezado.
- Se eliminan el texto y los guiones bajos en cada región antes de devolverlas.

### 5. Función `remove_underline(region)`

```python
Copiar código
def remove_underline(region):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(region, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        underline_y = min([cv2.boundingRect(contour)[1] for contour in contours])
        return region[:underline_y, :]

    return region
```

**Propósito:** Detectar y eliminar guiones bajos (líneas horizontales) en la región.

**Detalles:**
- Utiliza morfología de apertura para detectar líneas horizontales.
- Se recorta la imagen por encima de la primera línea detectada.

### 6. Funciones `correct_name(region_name)`, `correct_date(region_date)`, `correct_class(region_class)`

Estas funciones validan si las regiones de nombre, fecha y clase cumplen con los criterios correctos usando componentes
conectados.

**Ejemplo:** `correct_name(region_name)`

```python
Copiar código
def correct_name(region_name):
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(region_name)

    output = cv2.cvtColor(region_name, cv2.COLOR_GRAY2BGR)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

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
```

**Propósito:** Verificar si la región de nombre contiene dos palabras.

**Detalles:**
- Utiliza componentes conectados para detectar las palabras y su separación.
- Valida si hay dos palabras en la región.

### 7. Función `detect_answer(renglon_img)`

```python
Copiar código
def detect_answer(renglon_img):
    binaria = preprocess_image(renglon_img)
    if cv2.countNonZero(binaria) == 0:
        return "Sin respuesta"

    contornos, jerarquia = cv2.findContours(binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    huecos = 0
    for i in range(len(contornos)):
        if jerarquia[0][i][3] != -1:
            huecos += 1

    if segments_count(binaria) == 1:
        return "A"
    elif segments_count(binaria) == 2:
        return "D"
    elif segments_count(binaria) == 3:
        return "B"
    elif huecos == 0:
        return "C"
    else:
        return "Error"
```

**Propósito:** Detectar la letra marcada en una fila de preguntas.

**Detalles:**
- Preprocesa la imagen y cuenta los huecos internos para determinar la letra seleccionada (A, B, C, o D).

### 8. Función `segments_count(img_binaria)`

```python
Copiar código
def segments_count(img_binaria):
    height, ancho = img_binaria.shape
    segmentations = []
    for y in range(0, height, height // 10):
        line = img_binaria[y : y + 1, :]
        if cv2.countNonZero(line) > 4:
            segmentations.append(line)
    return len(segmentations)
```

**Propósito:** Contar los segmentos horizontales en la imagen binaria.

**Detalles:**
- Divide la imagen en 10 secciones horizontales y cuenta los segmentos con píxeles blancos.

### 9. Función `generate_result_image(results, output_filename)`

```python
Copiar código
def generate_result_image(results, output_filename="resultados_examenes.png"):
    block_height, block_width = 110, 310

    img_results_height = block_height * len(results)
    img_results = np.ones((img_results_height, block_width, 3), dtype=np.uint8) * 255

    approved_color = (0, 255, 0)
    rejected_color = (0, 0, 255)

    for idx, (name, exam_number, _, estado) in enumerate(results):
        region_name = cv2.cvtColor(exam_name[exam_number - 1][1], cv2.COLOR_GRAY2BGR)
        resized_name = cv2.resize(region_name, (block_width - 10, block_height - 10))
        color = approved_color if estado == "Aprobado" else rejected_color

        bordered_name = cv2.copyMakeBorder(resized_name, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=color)

        y_start = idx * block_height
        y_end = y_start + block_height

        img_results[y_start:y_end, :] = bordered_name

    cv2.imwrite(output_filename, img_results)
    print(f"Imagen de resultados guardada como {output_filename}")
```

**Propósito:** Crear una imagen que muestra los resultados de los exámenes (aprobados y desaprobados).

**Detalles:**
- Dibuja bloques con nombres de los exámenes, coloreados en verde o rojo según si el examen está aprobado o desaprobado.

