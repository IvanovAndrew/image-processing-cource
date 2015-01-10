__author__ = 'Andrew'

import numpy as np
import cv2
import cv2 as cv

RESULT_FILE_NAME = "result.txt"
DELTA = 2
UNIQUE_MIN_THRESHOLD = 35
DERIVATION_THRESHOLD = 25

NEED_COLORED = True

if NEED_COLORED:
    fst_image = cv2.imread("im2.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    snd_image = cv2.imread("im6.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    one_colored = cv2.imread("im2.png", cv2.CV_LOAD_IMAGE_COLOR)
else:
    fst_image = cv2.imread("disp2.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    snd_image = cv2.imread("disp6.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)

rows = fst_image.shape[0]
columns = fst_image.shape[1]

print fst_image.shape

# region file manipulations
def create_file ():

    file = open(RESULT_FILE_NAME, "w")
    file.write("")
    file.close()

def write_to_file (x, y, z, color):
    b, g, r = color
    file = open (RESULT_FILE_NAME, "a")
    file.write (str(x) + " " + str(y) + " " + str(z) + " " + str(r) + " " + str(g) + " " + str(b) + "\n" )
    file.close()

def write_to_file_gray (x, y, z):
    file = open (RESULT_FILE_NAME, "a")
    file.write (str(x) + " " + str(y) + " " + str(z) + "\n" )
    file.close()

create_file()
# endregion

def calculate_x_y_z(u, v, d):

    x = u - rows / 2
    y = v - columns / 2
    z = 5000 / float(d)

    return x, y, z

def analyse_min(line):

    fst_ind, fst_min = np.argmin(line), np.min(line)
    line[fst_ind] = line[fst_ind] * 100

    snd_ind, snd_min = np.argmin(line), np.min(line)
    if (fst_ind != snd_ind) and (np.abs(fst_min - snd_min) < UNIQUE_MIN_THRESHOLD):
        return -1

    return fst_ind

def calculate_derivation(line):

    res = np.zeros(len(line), dtype=int)
    for i in range(1, len(line)):
        res[i] = int(line[i]) - int(line[i - 1])

    return res

def find_bounds():

    for i in range(DELTA, rows):
    # for i in range(rows/2, rows):
        print "line", i, "is processed"

        der_vector = calculate_derivation(fst_image[i])

        for j in range(DELTA, columns - DELTA):
        # for j in range(DELTA, columns/2):
            if np.abs(der_vector[j]) > DERIVATION_THRESHOLD:
                template = fst_image[i - DELTA : i + DELTA + 1, j - DELTA: j + DELTA + 1]
                line = snd_image[i - DELTA : i + DELTA + 1, 0 : j]
                result = cv.matchTemplate(line, template, cv2.cv.CV_TM_SQDIFF)

                index = analyse_min(result[0])
                if index > 0:
                    shift = index - j
                    if shift != 0:
                        x, y, z = calculate_x_y_z(i, j, shift)
                        # x, y, z = i, j, shift

                        if NEED_COLORED:
                            write_to_file(x, y, z, one_colored[i, j])
                        else:
                            write_to_file_gray(x, y, z)
                    # else:
                    #     print "shift <= 0"

        print ""

find_bounds()