__author__ = 'Andrew'

import numpy as np
import cv2

BLACK = 0
GRAY = 128
RED = 200
WHITE = 255
IMAGE_SIZE = 600
FRAMES_COUNT = 10
GOOD_TRACKS_COUNT = 20
IMAGE_NAME = "frame_"
EXTENSION = ".png"

first = []
second = []
third = []
fourth = []
fifth = []

# region source map creating
def create_map():
    image = np.empty([IMAGE_SIZE, IMAGE_SIZE], dtype=type(int))
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            image[i, j] = BLACK
    return image
# endregion

# region source tracks
def get_next_coordinates_1(t):

    return 5*t + 1, IMAGE_SIZE/2 - t/4

def get_next_coordinates_2(t):

    return 5*t + 1, IMAGE_SIZE/2 + t/4
# endregion

def draw_straight_line(image, one, two, color):
    x0, y0 = one
    x1, y1 = two

    delta_x = x1 - x0
    step = (y1 - y0)/float(delta_x)

    for i in range(delta_x + 1):
        x = x0 + i
        y = y0 + int(step * i)

        plot_point(image, (x, y), color)

def plot_point(image, point, color):

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

def update_points(one, two, three, four, five, newPoint):

    one = two
    two = three
    three = four
    four = five
    five = newPoint
    return one, two, three, four, five

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

    return result
# endregion

def save_image(image, suffix):
    name = IMAGE_NAME + str(suffix) + EXTENSION
    cv2.imwrite(name, image)

def draw_tracks(tracks):

    is_first = True
    image = create_map()

    for track in tracks:
        color = GRAY
        if is_first:
            color = RED
            is_first = False

        point1, point2, point3, point4, point5 = track
        draw_straight_line(image, point1, point2, color)
        draw_straight_line(image, point2, point3, color)
        draw_straight_line(image, point3, point4, color)
        draw_straight_line(image, point4, point5, color)

    return image

for i in range(FRAMES_COUNT):
    print "##### ", i, "starts #####"
    point1 = get_next_coordinates_1(i)
    point2 = get_next_coordinates_2(i)

    first, second, third, fourth, fifth = \
        update_points(first, second, third, fourth, fifth, (point1, point2))

    print "first = ", first
    print "second = ", second
    good_tracks = generate_hypothesis()

    image = draw_tracks(good_tracks)

    save_image(image, i)

    print "#####", i, "ends #####"