__author__ = 'Andrew'

import numpy as np
##import cv2

BLACK = 0
GRAY = 128
WHITE = 255
IMAGE_SIZE = 600
FRAMES_COUNT = 100
GOOD_TRACKS_COUNT = 20

first = []
second = []
third = []
fourth = []
fifth = []

def create_map():

    image = np.empty([IMAGE_SIZE, IMAGE_SIZE], dtype=np.uint8)
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            image[i, j] = BLACK
    return image

def get_next_coordinates_1(t):

    return 5*t + 1, IMAGE_SIZE/2 - t/4

def get_next_coordinates_2(t):

    return 5*t + 1, IMAGE_SIZE/2 + t/4

def plot_point(map, point):

    x, y = point
    map[x - 1, y - 1] = WHITE
    map[x - 1, y] = WHITE
    map[x - 1, y + 1] = WHITE

    map[x, y - 1] = WHITE
    map[x, y] = WHITE
    map[x, y + 1] = WHITE

    map[x + 1, y - 1] = WHITE
    map[x + 1, y] = WHITE
    map[x + 1, y + 1] = WHITE

# points = []
# source = create_map()

def init_points():

    first.append  (get_next_coordinates_1(0), get_next_coordinates_2(0))
    second.append (get_next_coordinates_1(1), get_next_coordinates_2(1))
    third.append  (get_next_coordinates_1(2), get_next_coordinates_2(2))
    fourth.append (get_next_coordinates_1(3), get_next_coordinates_2(3))
    fifth.append  (get_next_coordinates_1(4), get_next_coordinates_2(4))

def update_points(one, two, three, four, five, newPoint):

    one = two
    two = three
    three = four
    four = five
    five = newPoint
    return one, two, three, four, five

def calculate_sum(vector):

    res = 0
    for num in vector:
        res += num

    return res

def calculate_scalar_production(one, two):

    res = 0
    for i in range(len(one)):
        res += one.index(i) * two.index(i)

    return res

def calculate_rating(k, b, x_vector, y_vector):

    res = 0
    for i in range(len(x_vector)):
        temp = k * x_vector.index(i) + b - y_vector.index(i)
        res += temp * temp

    return res

def analyse_hypothesis(p0, p1, p2, p3, p4):

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

def generate_tracks():
    hypoToRating = {}
    for point1 in first:
        for point2 in second:
            for point3 in third:
                for point4 in fourth:
                    for point5 in fifth:
                        hypothesis = (point1, point2, point3, point4, point5)
                        rating = analyse_hypothesis (hypothesis)
                        hypoToRating[hypothesis] = rating
    hypoToRating.items().sort()


for i in range(5, FRAMES_COUNT):
    point1 = get_next_coordinates_1(i)
    point2 = get_next_coordinates_2(i)

    first, second, third, fourth, fifth = \
        update_points(first, second, third, fourth, fifth, (point1, point2))

    generate_tracks()
    # points.append(point1)
    # points.append(point2)
    # print "one =", point1, "two = ", point2

# for point in points:
#     plot_point(source, point)

# black_image = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
# cv2.imwrite("tracks.png", black_image)