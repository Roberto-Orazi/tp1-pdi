# Resumen del Código: Ecualización Local de Histograma en Imágenes

Este código implementa un proceso de ecualización local de histograma para una imagen en escala de grises. La
ecualización de histograma mejora el contraste de la imagen, lo cual puede resaltar detalles que de otro modo estarían
ocultos.

## Funciones del Código

### 1. `local_histogram_equalization(img, window_size)`

Esta función aplica ecualización local de histograma a la imagen proporcionada, utilizando una ventana de tamaño
definido por `window_size`.

```python
def local_histogram_equalization(img, window_size):
    if len(img.shape) != 2:
        raise ValueError("Input image must be grayscale.")

    rows, cols = img.shape
    M, N = window_size
    half_M, half_N = M // 2, N // 2

    padded_img = cv2.copyMakeBorder(
        img, half_M, half_M, half_N, half_N, cv2.BORDER_REPLICATE
    )

    output_img = np.zeros_like(img)

    for i in range(rows):
        for j in range(cols):
            window = padded_img[i : i + M, j : j + N]

            hist, _ = np.histogram(window.flatten(), bins=256, range=[0, 256])
            cdf = hist.cumsum()

            cdf_min = cdf[cdf > 0].min()
            cdf_range = cdf.max() - cdf_min

            img_val = window[half_M, half_N]
            output_val = (cdf[img_val] - cdf_min) * 255 / cdf_range
            output_img[i, j] = np.clip(output_val, 0, 255)

    return output_img.astype(np.uint8)
```

**Detalles del Proceso:**
- Se asegura que la imagen de entrada esté en escala de grises.
- Se agrega un borde a la imagen para poder procesar las ventanas en los bordes.
- Se aplica la ecualización a cada ventana deslizante sobre la imagen original, mejorando el contraste de manera
  localizada.

### 2. Procesamiento de la Imagen

```python
img = cv2.imread("Imagen_con_detalles_escondidos.tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error reading the image. Check if the file path is correct.")
else:
    window_sizes = [(3, 3), (3, 50), (50, 3), (250, 250), (15, 15), (25, 25)]

    num_windows = len(window_sizes) + 1
    cols = 3
    rows = (num_windows + cols - 1) // cols

    plt.figure(figsize=(15, 10))

    plt.subplot(rows, cols, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Imagen Original")
    plt.axis("off")

    for i, window_size in enumerate(window_sizes, start=2):
        img_local_eq = local_histogram_equalization(img, window_size)

        plt.subplot(rows, cols, i)
        plt.imshow(img_local_eq, cmap="gray")
        plt.title(f"Ventana {window_size}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
```

- Se carga la imagen llamada `"Imagen_con_detalles_escondidos.tif"` en escala de grises.
- Si la imagen no se encuentra, se muestra un mensaje de error.
- Se define una lista de diferentes tamaños de ventanas para realizar la ecualización local.
- Para cada tamaño de ventana, se aplica la función de ecualización local y se visualizan los resultados usando
  `matplotlib`.

### Visualización

- Se utiliza `matplotlib` para mostrar la imagen original y las imágenes resultantes tras la aplicación de la
  ecualización local con diferentes tamaños de ventanas.
- Cada imagen procesada se muestra con su respectivo título indicando el tamaño de la ventana utilizada.

## Objetivo del Código

Este código tiene como objetivo mejorar el contraste de una imagen utilizando la técnica de ecualización local de
histograma con ventanas de diferentes tamaños. Esto puede ser útil para resaltar detalles específicos que pueden ser
difíciles de ver en la imagen original.

## Librerías Utilizadas

- **`matplotlib`**: Para visualizar la imagen original y las imágenes procesadas.
- **`cv2` (OpenCV)**: Para procesar la imagen y aplicar operaciones como agregar bordes y calcular histogramas.
- **`numpy`**: Para operaciones matemáticas y manejo de matrices.

