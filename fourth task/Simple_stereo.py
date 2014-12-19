from cv2.cv import CV_TM_SQDIFF

__author__ = 'Andrew'

import numpy as np
import cv2
import cv2 as cv

RESULT_FILE_NAME = "result.txt"
DELTA = 1
BIG_VALUE = 10000
MATCHING_THRESHOLD = 100
PIXEL_THRESHOLD = 150

fst_image = cv2.imread("im2.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
snd_image = cv2.imread("im6.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)

s_x = fst_image.shape[0]
s_y = fst_image.shape[1]

# region file manipulations
def create_file (fileName):

    file = open(fileName, "w")
    file.write("")
    file.close()

def write_to_file (fileName, x, y, z):

    file = open (fileName, "a")
    file.write (str(x) + "," + str(y) + "," + str(z) + "\n")
    file.close()

create_file(RESULT_FILE_NAME)
# endregion

def calculate_x_y_z(u, v, d):
    x = u - s_x/2
    y = v - s_y/2
    z = 1000 / d

    return x, y, z

def get_pixel_with_neighs(image, x, y):

    result = np.zeros([2 * DELTA + 1], dtype=np.float32)

    for i in range(2 * DELTA + 1):
        shift = i - DELTA
        result[i] = image[x, y + shift]

    return result

def analyse_min(line):

    min_index = np.argmin(line)
    fst_min = line[min_index]

    line[min_index] = BIG_VALUE
    snd_min = np.min(line)

    if np.abs(snd_min - fst_min) <= MATCHING_THRESHOLD:
        min_index = -1

    return min_index

def find_bounds(one, two):
    rows, columns = one.shape
    for i in range(rows):
        for j in range(DELTA, columns - DELTA):
            if one[i, j] > PIXEL_THRESHOLD:
                line = two[i]
                template = one[i, j - DELTA: j + DELTA]

                result = cv.matchTemplate(line, template, CV_TM_SQDIFF)

                index = analyse_min(result)
                if index >= 0 and j != index:
                    shift = index - j
                    x, y, z = calculate_x_y_z(i, j, shift)
                    write_to_file(RESULT_FILE_NAME, x, y, z)

one = cv2.Sobel(fst_image, cv2.CV_32F, 1, 0, ksize=3, scale=1, delta=0)
two = cv2.Sobel(snd_image, cv2.CV_32F, 1, 0, ksize=3, scale=1, delta=0)

find_bounds(one, two)