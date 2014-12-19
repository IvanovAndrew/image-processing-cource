__author__ = 'Andrew'

import numpy as np
import cv2
import copy

FRAMES_COUNT = 100
GOOD_HYPO_COUNT = 20
LAST_POINTS = 5

OBJECT_SIZE = 3

IMAGE_SIZE = FRAMES_COUNT * (OBJECT_SIZE * 3) + 2
IMAGE_NAME = "frame_"
EXTENSION = ".png"

#region colors
BLACK = [0, 0, 0]

BLUE           = [255, 0, 0]
DARK_TURQUOISE = [209, 206, 0]
DARK_GREEN     = [0, 100, 0]
GREEN          = [0, 255, 0]
GRAY           = [128, 128, 128]
SADDLE_BROWN   = [19, 69, 139]
INDIAN_RED     = [92, 92, 205]
BEIGE          = [220, 245, 245]
MAGENTA        = [255, 0, 255]
YELLOW         = [0, 255, 255]
NAVY_BLUE      = [128, 0, 0]
PEACH_PUFF1     = [185, 218, 255]
MISTYROSE1     = [225, 228, 255]
SLATEBLUE1     = [255, 111, 131]
CYAN          = [255, 255, 0]
SPRING_GREEN1  = [127, 255, 0]
DARK_GOLDENROD4 = [8, 101, 139]
PLUM1          = [255, 187, 255]
DARK_ORCHID1   = [255, 62, 191]
RED            = [0, 0, 255]
WHITE          = [255, 255, 255]

number_to_color = dict([
                        (0, GRAY),
                        (1, DARK_GOLDENROD4),
                        (2, DARK_TURQUOISE),
                        (3, MISTYROSE1),
                        (4, DARK_GREEN),
                        (5, GREEN),
                        (6, PLUM1),
                        (7, SLATEBLUE1),
                        (8, BLUE),
                        (9, CYAN),
                        (10, SADDLE_BROWN),
                        (11, PEACH_PUFF1),
                        (12, INDIAN_RED),
                        (13, NAVY_BLUE),
                        (14, BEIGE),
                        (15, SPRING_GREEN1),
                        (16, MAGENTA),
                        (17, YELLOW),
                        (18, DARK_ORCHID1),
                        (19, RED)
                    ])
# endregion

hypotheses = []

class Hypothesis:

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def draw(self, image, color):

        self.first.draw(image, color)
        self.second.draw(image, color)

    def get_total_points(self):

        total_points = len(self.first.path)

        result = []
        for i in range(total_points):

            point1 = self.first.path[i]
            result.append (point1)

            point2 = self.second.path[i]
            result.append (point2)

        return result

    def append_points(self, to_first, to_second):

        self.first.append_point(to_first)
        self.second.append_point(to_second)

    def to_string(self):
        s = "first: " + self.first.to_string() + "\n"
        s += "second: " + self.second.to_string() + "\n"
        return s

class Trajectory:
    def __init__(self, path):
        self.path = path

    # region Drawing
    def plot_track_point(self, image, point, color):

        x, y = point
        image[x, y] = color

    def draw_straight_line(self, image, one, two, color):

        is_red = color == RED
        # print one, " -> ", two
        x_start, y_start = one
        x_end, y_end = two

        x0, y0 = x_start, y_start

        delta_x = x_end - x_start
        delta_y = y_end - y_start

        if np.abs(delta_x) > np.abs(delta_y):
            step = delta_y/float(delta_x)
            red_shift = 1
            if step < 0:
                red_shift = -1

            for i in range(1, np.abs(delta_x)):
                x = x0 + i
                y = y0 + int(step * i)
                # print (x, y)

                self.plot_track_point(image, (x, y), color)

                if is_red:
                    self.plot_track_point(image, (x, y + red_shift), color)

        else:
            step = delta_x / float(delta_y)

            if step < 0:
                y0 = y_end
                x0 = x_end

            for i in range(1, np.abs(delta_y)):
                x = x0 + int(step * i)
                y = y0 + i
                # print (x, y)

                self.plot_track_point(image, (x, y), color)

                if is_red:
                    self.plot_track_point(image, (x + 1, y), color)

    def draw(self, image, color):

        points_count = len(self.path)

        for i in range(points_count - 1):
            one = self.path[i]
            two = self.path[i + 1]
            self.draw_straight_line (image, one, two, color)
    # endregion

    def append_point(self, point):

        self.path.append(point)

    def get_last_points(self, last):

        total_points = len(self.path)
        temp = last

        if last == -1 or last > total_points:
            temp = total_points

        result = []
        for i in range(total_points - temp, total_points):

            point1 = self.path[i]

            result.append(point1)

        return result

    def to_string(self):

        s = ""
        for point in self.path:
            s += str(point) + " -> "

        return s

# region Image creating and saving
def create_map():

    image = np.empty([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=int)
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            image[i, j] = BLACK
    return image

def save_image(image, suffix):

    str_suffix = str(suffix)
    if len(str_suffix) == 1:
        str_suffix = "0" + str_suffix
    name = IMAGE_NAME + str_suffix + EXTENSION
    cv2.imwrite(name, image)
# endregion

# region Drawing object points
def fill_object_points(image, points_list):

    for point in points_list:
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
def draw_tracks():

    image = create_map()
    count = 0

    hypo_count = len(hypotheses)

    hypotheses.reverse()
    for hypo in hypotheses:
        color = number_to_color.get(count)
        if count == hypo_count - 1:
            color = RED

        hypo.draw(image, color)
        count += 1

    hypotheses.reverse()
    detections = hypotheses[0].get_total_points()
    image = fill_object_points(image, detections)

    return image
# endregion

# region threshold hypotheses
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

def analyse_hypothesis(x_vector, y_vector):

    ## number of points
    N = len(x_vector)

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
    k = (N * S - X * Y) / float(N * Q - X * X)

    ## straight line: y = k * x + b
    ## koefficient b
    b = (Q * Y - S * X) / float(N * Q - X * X)

    rating = calculate_rating(k, b, x_vector, y_vector)
    return rating

def threshold_hypotheses():
    hypoToRating = []

    for hypo in hypotheses:
        x_vector = []
        y_vector = []

        last_points = hypo.first.get_last_points(LAST_POINTS)
        for point in last_points:
            x, y = point
            x_vector.append(x)
            y_vector.append(y)

        rating = analyse_hypothesis (x_vector, y_vector)
        hypoToRating.append((hypo, rating))

    items = sorted(hypoToRating, key=lambda x: x[1])
    count = 0
    result = []
    for item in items:
        if count == GOOD_HYPO_COUNT:
            break

        hypo, rating = item

        # print "hypo", count, ": \n", hypo.to_string(), "rating ", rating
        # print ""
        result.append(hypo)
        count += 1

    return result
# endregion

# region Generate new hypotheses
def generate_hypothesis(point1, point2):

    result = []

    for hypo in hypotheses:
        one_hypo = copy.deepcopy(hypo)
        one_hypo.append_points(point1, point2)
        result.append(one_hypo)

        two_hypo = hypo
        two_hypo.append_points(point2, point1)
        result.append(two_hypo)

    return result
# endregion

# region Source tracks
def get_next_coordinates_1(t):

    return t * (OBJECT_SIZE * 3) + 1, IMAGE_SIZE/2 - t * (OBJECT_SIZE - 1)

def get_next_coordinates_2(t):

    return t * (OBJECT_SIZE * 3) + 1, IMAGE_SIZE/2 + t * (OBJECT_SIZE - 1)
# endregion

for i in range(FRAMES_COUNT):

    print "##### ", i, "starts #####"
    point1 = get_next_coordinates_1(i)
    point2 = get_next_coordinates_2(i)

    if i == 0:
        first_track = Trajectory([point1])
        second_track = Trajectory([point2])

        hypo = Hypothesis(first_track, second_track)
        hypotheses.append(hypo)

        continue

    if i == 1:
        hypo = hypotheses[0]
        hypo.append_points(point1, point2)

        continue

    hypotheses = generate_hypothesis(point1, point2)
    hypotheses = threshold_hypotheses()

    image = draw_tracks()
    save_image(image, i)

    print "#####", i, "ends #####"