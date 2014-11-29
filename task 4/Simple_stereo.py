__author__ = 'Andrew'

import numpy as np
import cv2

RESULT_FILE_NAME = "result.txt"

fst_image = cv2.cvtColor(cv2.imread("im2.png"), cv2.COLOR_BGR2GRAY)
snd_image = cv2.cvtColor(cv2.imread("im6.png"), cv2.COLOR_BGR2GRAY)

s_x = fst_image.shape[0]
s_y = fst_image.shape[1]


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

def analyse_line(num):
    for j in range(s_y):
        if j == 0:
            continue

        if (fst_image[j])

for line in range(s_x):
    analyse_line(line)
