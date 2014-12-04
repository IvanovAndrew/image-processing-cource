from cv2.cv import CV_TM_SQDIFF

__author__ = 'Andrew'

import numpy as np
import cv2
import cv2 as cv

RESULT_FILE_NAME = "result.txt"
DERIVATION_THRESHOLD = 20
DELTA = 1
BIG_VALUE = 10000
MATCHING_THRESHOLD = 100 * DELTA

fst_image = cv2.imread("im2.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
snd_image = cv2.imread("im6.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)

s_x = fst_image.shape[0]
s_y = fst_image.shape[1]
total_count = 0

print fst_image.shape

# region file manipulations
def create_file (fileName):
    file = open(fileName, "w")
    file.write("")
    file.close()

def write_to_file (fileName, x, y, z):
    file = open (fileName, "a")
    file.write (str(x) + " " + str(y) + " " + str(z) + "\n")
    file.close()

create_file(RESULT_FILE_NAME)
# endregion

def calculate_x_y_z(u, v, d):
    x = u - s_x/2
    y = v - s_y/2
    z = 1000 / d

    return x, y, z

def get_pixel_with_neighs(image, x, y):

    result = np.zeros([2 * DELTA + 1], dtype=np.uint8)

    for i in range(2 * DELTA + 1):
        shift = i - DELTA
        result[i] = image[x, y + shift]

    return result

def analyse_min(line):

    min_index = -1
    fst_min = BIG_VALUE
    snd_min = BIG_VALUE

    for index in range(len(line)):
        num = line[index]
        if num[0] < fst_min:
            snd_min = fst_min
            fst_min = num[0]
            min_index = index + 1
        elif num[0] < snd_min:
            snd_min = num[0]

    if np.abs(snd_min - fst_min) <= MATCHING_THRESHOLD:
        min_index = -1
    else:
        print "fst_min =", fst_min, "snd_min =", snd_min

    return min_index

def to_first_derivation(image):

    row, columns = image.shape

    result = np.zeros([row, columns - 1], dtype=np.uint8)

    for i in range(row):
        for j in range(1, columns):
            value = np.float(image[i, j]) - np.float(image[i, j - 1])
            if value <= DERIVATION_THRESHOLD:
                result[i, j - 1] = 0
            else:
                result[i, j - 1] = 255##int(value)

    return result

def find_bounds(one, two):
    rows, columns = one.shape
    for i in range(rows):
        for j in range(columns - DELTA):
            if j < DELTA:
                continue

            # diff = np.float(one[i, j]) - np.float(one[i, j - 1])
            # if np.abs(diff) >= DERIVATION_THRESHOLD:
            if one[i, j] > 0:
                line = two[i, :]
                template = get_pixel_with_neighs(one, i, j)

                result = cv.matchTemplate(line, template, CV_TM_SQDIFF)
                index = analyse_min(result)
                if index >= 0 and j != index:
                    shift = index - j
                    x, y, z = calculate_x_y_z(i, j, shift)
                    write_to_file(RESULT_FILE_NAME, x, y, z)

one = to_first_derivation(fst_image)
two = to_first_derivation(snd_image)
find_bounds(one, two)