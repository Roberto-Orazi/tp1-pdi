import matplotlib

matplotlib.use("TkAgg")
import cv2
import numpy as np
import matplotlib.pyplot as plt

examen_1 = cv2.imread("examen_1.tif", cv2.IMREAD_GRAYSCALE)
examen_2 = cv2.imread("examen_2.tif", cv2.IMREAD_GRAYSCALE)
examen_3 = cv2.imread("examen_3.tif", cv2.IMREAD_GRAYSCALE)
examen_4 = cv2.imread("examen_4.tif", cv2.IMREAD_GRAYSCALE)
examen_5 = cv2.imread("examen_5.tif", cv2.IMREAD_GRAYSCALE)


def automatic_correction(img):
    """
    Corrections from the header:
        - Name: 2 Words, < 25 Chars
        - Date: 8 Chars with no space between. Ex: (01/01/24)
        - Class: 1 Char

    Correct Answers for the test:

    1.C - 2.B - 3.A - 4.D - 5.B -  6.B - 7.A - 8.B - 9.D - 10.D
    """
    correct_answers = ["C", "B", "A", "D", "B", "B", "A", "B", "D", "D"]
    if len(img.shape) != 2:
        raise ValueError("Input image must be grayscale.")
