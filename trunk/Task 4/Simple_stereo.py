__author__ = 'Andrew'

import numpy as np
import cv2
import cv2 as cv

RESULT_FILE_NAME = "result.txt"
DELTA = 2
MATCHING_THRESHOLD = 100
DERIVATION_THRESHOLD = 10

fst_image = cv2.imread("im2.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
snd_image = cv2.imread("im6.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)

one_colored = cv2.imread("im2.png", cv2.CV_LOAD_IMAGE_COLOR)

rows = fst_image.shape[0]
columns = fst_image.shape[1]

# region file manipulations
def create_file ():

    file = open(RESULT_FILE_NAME, "w")
    file.write("")
    file.close()

def write_to_file (x, y, z, color):
    r, g, b = color
    file = open (RESULT_FILE_NAME, "a")
    file.write (str(x) + " " + str(y) + " " + str(z) + " " + str(r) + " " + str(g) + " " + str(b) + "\n" )
    file.close()

# def write_to_file (x, y, z):
#     file = open (RESULT_FILE_NAME, "a")
#     file.write (str(x) + " " + str(y) + " " + str(z) + "\n" )
#     file.close()

create_file()
# endregion

def calculate_x_y_z(u, v, d):

    x = u - rows / 2
    y = v - columns / 2
    z = 1000 / d

    return x, y, z

def analyse_min(line):

    return np.argmin(line), np.min(line)

def calculate_derivation(line):

    res = np.zeros(len(line), dtype=int)
    for i in range(1, len(line)):
        res[i] = int(line[i]) - int(line[i - 1])

    return res

def find_bounds():

    for i in range(rows):
    # for i in range(300, 301):
        print "line", i, "is processed"

        der_vector = calculate_derivation(fst_image[i])
        xy_pair = np.zeros(columns, dtype=int)
        is_exist_minimum = np.zeros(columns, dtype=bool)
        minimum_values = np.zeros(columns, dtype=int)

        for j in range(DELTA, columns - DELTA):
            if np.abs(der_vector[j]) > DERIVATION_THRESHOLD:
                template = fst_image[i, j - DELTA: j + DELTA + 1]
                line = snd_image[i]
                result = cv.matchTemplate(line, template, cv2.cv.CV_TM_SQDIFF)

                index, min_value = analyse_min(result)
                if min_value < MATCHING_THRESHOLD:
                    # print min_value
                    if is_exist_minimum[index]:

                        if min_value < minimum_values[index]:
                            xy_pair[index] = j
                            minimum_values[index] = min_value
                    else:
                        is_exist_minimum[index] = True
                        minimum_values[index] = min_value
                        xy_pair[index] = j

        for j in range(columns):
            if is_exist_minimum[j] and xy_pair[j] != j:
                s = xy_pair[j]

                shift = j - s
                x, y, z = calculate_x_y_z(i, s, shift)
                # x, y, z = i, j, shift
                write_to_file(x, y, z, one_colored[i, s])
                # write_to_file(x, y, z)

        print ""

find_bounds()