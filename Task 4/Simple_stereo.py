__author__ = 'Andrew'

import numpy as np
import cv2
import cv2 as cv

RESULT_FILE_NAME = "result.txt"
DELTA = 2
MATCHING_THRESHOLD = 500 * DELTA
PIXEL_THRESHOLD = 100

one_colored = cv2.imread("im2.png", cv2.CV_LOAD_IMAGE_COLOR)

fst_image = cv2.cvtColor(one_colored, cv.COLOR_RGB2GRAY)
snd_image = cv2.imread("im6.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)

s_x = fst_image.shape[0]
s_y = fst_image.shape[1]

# region file manipulations
def create_file (fileName):

    file = open(fileName, "w")
    file.write("")
    file.close()

def write_to_file (fileName, x, y, z, color):
    r, g, b = color
    file = open (fileName, "a")
    file.write (str(x) + " " + str(y) + " " + str(z) + " " + str(r) + " " + str(g) + " " + str(b) + "\n" )
    file.close()

create_file(RESULT_FILE_NAME)
# endregion

def calculate_x_y_z(u, v, d):

    x = u - s_x / 2
    y = v - s_y / 2
    z = 1000 / d

    return x, y, z

def analyse_min(line):

    min_index = np.argmin(line)
    fst_min = line[min_index]

    line[min_index] = 1000000
    snd_min = np.min(line)

    if np.abs(snd_min - fst_min) <= MATCHING_THRESHOLD:
        min_index = -1

    return min_index

def find_bounds(one, two):

    rows, columns = one.shape
    for i in range(rows):
    # for i in range(300, 301):
        for j in range(DELTA, columns - DELTA):
            if one[i, j] > PIXEL_THRESHOLD:
                line = two[i]
                template = one[i, j - DELTA: j + DELTA + 1]

                result = cv.matchTemplate(line, template, cv2.cv.CV_TM_SQDIFF_NORMED)

                index = analyse_min(result)
                if index >= 0 and j != index:
                    shift = index - j
                    x, y, z = calculate_x_y_z(i, j, shift)
                    # x, y, z = i, j, shift
                    write_to_file(RESULT_FILE_NAME, x, y, z, one_colored[i, j])

                    # print (x, y) , one[i, j - DELTA: j + DELTA + 1], " -> ", two[i, index - DELTA : index + DELTA + 1]
                else:
                    if j == index:
                        print "j = index"
                    else:
                        print "index = -1"
        print i

one = np.abs(cv2.Sobel(fst_image, cv2.CV_32F, 0, 1, ksize=3))
two = np.abs(cv2.Sobel(snd_image, cv2.CV_32F, 0, 1, ksize=3))

# cv.imwrite("one sobel.png", one)
# cv.imwrite("two sobel.png", two)
find_bounds(one, two)