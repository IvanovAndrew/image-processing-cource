__author__ = 'Andrew'

import numpy as np
import cv2

BLACK = [0, 0, 0]
GRAY = [128, 128, 128]
RED = [0, 0, 255]
BLUE = [255, 0, 0]
GREEN = [0, 255, 0]
WHITE = [255, 255, 255]

number_to_color = dict([(0, GRAY), (1, GRAY),(2, GRAY),(3, GRAY),(4, GRAY),(5, GRAY),
                        (6, BLUE),(7, BLUE),(8, BLUE), (9, BLUE),(10, BLUE),(11, BLUE),
                        (12, GREEN),(13, GREEN),(14, GREEN),(15, GREEN),(16, GREEN),(17, GREEN),(18, GREEN),
                        (19, RED)])

FRAMES_COUNT = 50
GOOD_TRACKS_COUNT = 20
LAST_POINTS = 5

OBJECT_SIZE = 3

IMAGE_SIZE = FRAMES_COUNT * OBJECT_SIZE * 4 + 3
IMAGE_NAME = "frame_"
EXTENSION = ".png"

first = []
second = []
third = []
fourth = []
fifth = []

# region Image creating
def create_map():

    image = np.empty([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=int)
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            image[i, j] = BLACK
    return image
# endregion

# region Image saving
def save_image(image, suffix):

    str_suffix = str(suffix)
    if len(str_suffix) == 1:
        str_suffix = "0" + str_suffix
    name = IMAGE_NAME + str_suffix + EXTENSION
    cv2.imwrite(name, image)
# endregion

# region Drawing object points
def fill_object_points(image):
    point1, point2 = first
    point3, point4 = second
    point5, point6 = third
    point7, point8 = fourth
    point9, point10 = fifth

    list = [point1, point2, point3, point4, point5, point6, point7, point8, point9, point10]

    for point in list:
        plot_object_point(image, point)

    return image

def plot_object_point(image, point):

    color = WHITE
    x, y = point
    image[x - 1, y - 1] = color
    image[x - 1, y] = color
    image[x - 1, y + 1] = color

    image[x, y - 1] = color
    image[x, y] = color
    image[x, y + 1] = color

    image[x + 1, y - 1] = color
    image[x + 1, y] = color
    image[x + 1, y + 1] = color
# endregion

# region Drawing tracks
def plot_track_point(image, point, color):

    x, y = point
    image[x, y] = color

def draw_straight_line(image, one, two, color):
    x_start, y_start = one
    x_end, y_end = two

    x0, y0 = x_start, y_start

    delta_x = x_end - x_start
    delta_y = y_end - y_start

    if np.abs(delta_x) > np.abs(delta_y):
        step = delta_y/float(delta_x)

        for i in range(1, np.abs(delta_x)):
            x = x0 + i
            y = y0 + int(step * i)
            # print (x, y)

            plot_track_point(image, (x, y), color)

    else:
        step = delta_x / float(delta_y)

        if step < 0:
            y0 = y_end

        for i in range(1, np.abs(delta_y)):
            x = x0 + int(step * i)
            y = y0 + i
            # print (x, y)

            plot_track_point(image, (x, y), color)

def draw_tracks(tracks):

    image = create_map()
    count = 0

    for track in tracks:
        color = number_to_color.get(count)

        point1, point2, point3, point4, point5 = track
        draw_straight_line(image, point1, point2, color)
        draw_straight_line(image, point2, point3, color)
        draw_straight_line(image, point3, point4, color)
        draw_straight_line(image, point4, point5, color)

        count += 1

    image = fill_object_points(image)

    return image
# endregion

# region Hypothesis
def calculate_sum(vector):

    res = 0
    for num in vector:
        res += num

    return res

def calculate_scalar_production(one, two):

    res = 0
    for i in range(len(one)):
        res += one[i] * two[i]

    return res

def calculate_rating(k, b, x_vector, y_vector):

    res = 0
    for i in range(len(x_vector)):
        temp = k * x_vector[i] + b - y_vector[i]
        res += temp * temp

    return res

def analyse_hypothesis(hypothesis):
    p0, p1, p2, p3, p4 = hypothesis
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    x_vector = [x0, x1, x2, x3, x4]
    y_vector = [y0, y1, y2, y3, y4]

    ## sum of x_i
    X = calculate_sum(x_vector)

    ## sum of y_i
    Y = calculate_sum(y_vector)

    ## sum of x_i^2
    Q = calculate_scalar_production(x_vector, x_vector)

    ## sum of x_i * y_i
    S = calculate_scalar_production(x_vector, y_vector)

    ## straight line: y = k * x + b
    ## koefficient k
    k = (5 * S - X * Y) / float(5 * Q - X * X)

    ## straight line: y = k * x + b
    ## koefficient b
    b = (Q * Y - S * X) / float(5 * Q - X * X)

    rating = calculate_rating(k, b, x_vector, y_vector)
    return rating

def generate_hypothesis():
    hypoToRating = {}
    for point1 in first:
        for point2 in second:
            for point3 in third:
                for point4 in fourth:
                    for point5 in fifth:
                        hypothesis = (point1, point2, point3, point4, point5)
                        rating = analyse_hypothesis (hypothesis)
                        hypoToRating[hypothesis] = rating

    items = sorted(hypoToRating.items(), key=lambda x: x[1])
    count = 0
    result = []
    for item in items:
        if count == GOOD_TRACKS_COUNT:
            break

        points, rating = item
        # print "track", count, ": points ", points, "rating ", rating

        result.append(points)
        count += 1

    result.reverse()
    return result
# endregion

# region source tracks
def get_next_coordinates_1(t):

    return t * (OBJECT_SIZE * 4) + 1, IMAGE_SIZE/2 - t * (OBJECT_SIZE * 2)

def get_next_coordinates_2(t):

    return t * (OBJECT_SIZE * 4) + 1, IMAGE_SIZE/2 + t * (OBJECT_SIZE * 2)
# endregion

for i in range(FRAMES_COUNT):
    print "##### ", i, "starts #####"
    point1 = get_next_coordinates_1(i)
    point2 = get_next_coordinates_2(i)

    first, second, third, fourth, fifth = second, third, fourth, fifth, (point1, point2)

    if i >= LAST_POINTS:
        good_tracks = generate_hypothesis()

        image = draw_tracks(good_tracks)

        save_image(image, i)

    print "#####", i, "ends #####"