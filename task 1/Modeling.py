__author__ = 'User'

import numpy as np
import numpy.random as random
import cv2

p = 0.1

BLACK_COLOR = [0, 0, 0]
WHITE_COLOR = [255, 255, 255]

BLACK_AS_ARRAY = np.asanyarray(BLACK_COLOR)
WHITE_AS_ARRAY = np.asanyarray(WHITE_COLOR)

MAP_SIZE = 50
SHIPS_PER_MAP = 25
SHIP_SIZE = 2

EXPERIMENTS_COUNT = 200
NEED_SAVE_IMAGES = False

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

EXPECTED_TRUE_POSITIVE  = (1 - p) * (1 - p * np.power(1 - p, 3))
EXPECTED_TRUE_NEGATIVE  = (1 - p) * (1 + p * np.power(1 - p, 3))
EXPECTED_FALSE_POSITIVE = p * (1 - np.power(1 - p, 4))
EXPECTED_FALSE_NEGATIVE = p * (1 + np.power(1 - p, 4))

class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to_string(self):
        s = "[x=" + str(self.x) + " y=" + str(self.y)+"]"
        return s

    def __eq__(self, other):
        return isinstance(other, Coordinate) and (self.x == other.x) and self.y == other.y

class Ship:
    def __init__(self, path):
        return self.__init__(path.pop(0), path.pop(1))

    def __init__(self, fstCoord, sndCoord):
        self.fstCoord = fstCoord
        self.sndCoord = sndCoord

    def to_string(self):
        s = self.fstCoord.to_string() + " " + self.sndCoord.to_string()
        return s

def save_image(name, image):

    cv2.imwrite(name + ".png", image)

def is_in_ship_border(center, ships_coords):

    for coord in ships_coords:
        if np.abs (coord.x - center.x) + np.abs (coord.y - center.y) <= 1:
            return True
    return False

def get_fst_ship_coordinate():

    x, y = random.randint(1, MAP_SIZE - 1), random.randint(1, MAP_SIZE - 1)
    return Coordinate(x, y)

def get_snd_ship_coordinate(center, ship_coords):

    candidates = [
                    Coordinate(center.x + 1, center.y),
                    Coordinate(center.x - 1, center.y),
                    Coordinate(center.x, center.y + 1),
                    Coordinate(center.x, center.y - 1),
                 ]
    result = []

    for coord in candidates:
        if  coord.x < 1 or coord.y < 1 or coord.x >= MAP_SIZE - 1 or coord.y >= MAP_SIZE - 1:
            continue

        if is_in_ship_border(coord, ship_coords):
            continue

        result.append(coord)

    return result

def create_ship(ships_coords):

    fstCoord = get_fst_ship_coordinate()

    while is_in_ship_border(fstCoord, ships_coords):
        fstCoord = get_fst_ship_coordinate()

    available = get_snd_ship_coordinate(fstCoord, ships_coords)
    if (len(available) == 0):
        return create_ship(ships_coords)

    else:
        number = random.randint(0, len(available))
        sndCoord = available.pop(number)

        return Ship(fstCoord, sndCoord)

def create_source_map():

    source_map = np.zeros([MAP_SIZE, MAP_SIZE, 3], dtype=np.int)

    created_ships_count = 0
    path = []

    while created_ships_count < SHIPS_PER_MAP:
        ship = create_ship(path)

        path.append(ship.fstCoord)
        path.append(ship.sndCoord)

        source_map[ship.fstCoord.x, ship.fstCoord.y] = WHITE_COLOR
        source_map[ship.sndCoord.x, ship.sndCoord.y] = WHITE_COLOR
        created_ships_count += 1

    return source_map

##### create map with noise #####
def create_noise_map(source_map):

    noise_map = np.copy(source_map)
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            if random.random() < p:
                if np.array_equal(noise_map[i, j], BLACK_AS_ARRAY):
                    noise_map[i, j] = WHITE_COLOR;
                else:
                    noise_map[i, j] = BLACK_COLOR;

    return noise_map

def get_available_neighborhoods(center, path):

    result = []
    coordinate = Coordinate(center.x + 1, center.y)
    if is_available(coordinate, path):
        result.append(coordinate)

    coordinate = Coordinate(center.x - 1, center.y)
    if is_available(coordinate, path):
        result.append(coordinate)

    coordinate = Coordinate(center.x, center.y + 1)
    if is_available(coordinate, path):
        result.append(coordinate)

    coordinate = Coordinate(center.x, center.y - 1)
    if is_available(coordinate, path):
        result.append(coordinate)

    return result

def is_available(coord, path):

    if coord.x < 0 or coord.x == MAP_SIZE or coord.y < 0 or coord.y == MAP_SIZE:
        return False

    for coordinate in path:
        if coordinate == coord:
            return False

    return True

""" search ship coordinates """
def find_neighborhoods(map, path, center):

    path.append(center)
    neighborhoods = get_available_neighborhoods(center, path)

    for n in neighborhoods:
        if np.array_equal(map[n.x, n.y], WHITE_AS_ARRAY):
            path = find_neighborhoods(map, path, n)

    return path

def algorithm(map):

    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            if np.array_equal(map[i, j], WHITE_AS_ARRAY):
                coord = Coordinate(i, j)
                path = find_neighborhoods(map, [], coord)
                if len(path) < SHIP_SIZE:
                    map[i, j] = BLACK_COLOR
    return map

for i in range (EXPERIMENTS_COUNT):

    source_map = create_source_map()
    noise_map = create_noise_map(np.copy(source_map))
    result_map = algorithm(np.copy(noise_map))

    if NEED_SAVE_IMAGES:
        save_image(str(i) + " map", source_map)
        save_image(str(i) + " noise map", noise_map)
        save_image(str(i) + " recovered map", result_map)

    for x in range(MAP_SIZE):
        for y in range(MAP_SIZE):
            if np.array_equal(source_map[x, y], WHITE_AS_ARRAY) and np.array_equal(result_map[x, y], WHITE_AS_ARRAY):
                true_positive += 1
            elif np.array_equal(source_map[x, y], WHITE_AS_ARRAY) and np.array_equal(result_map[x, y], BLACK_AS_ARRAY):
                false_negative += 1
            elif np.array_equal(source_map[x, y], BLACK_AS_ARRAY) and np.array_equal(result_map[x, y], WHITE_AS_ARRAY):
                false_positive += 1
            else: #source_map[x, y] == BLACK and np.array_equal(result_map[x, y], BLACK_AS_ARRAY):
                true_negative += 1

print "true positive results ", true_positive
print "true positive frequency ", true_positive / float(SHIPS_PER_MAP * SHIP_SIZE * EXPERIMENTS_COUNT)
print "true positive probability ", EXPECTED_TRUE_POSITIVE

print ""
print "false negative results ", false_negative
print "false negative frequency ", false_negative / float(SHIPS_PER_MAP * SHIP_SIZE * EXPERIMENTS_COUNT)
print "false negative probability", EXPECTED_FALSE_NEGATIVE
print ""

print "CHECK: true positive results + false negative results = ", SHIPS_PER_MAP * SHIP_SIZE * EXPERIMENTS_COUNT
print "", true_positive, " + ", false_negative, " = ", true_positive + false_negative

print "====="
print "true negative results", true_negative
print "true negative frequency", true_negative / float((MAP_SIZE * MAP_SIZE - SHIPS_PER_MAP * SHIP_SIZE) * EXPERIMENTS_COUNT)
print "true negative probability", EXPECTED_TRUE_NEGATIVE

print ""

print "false positive results", false_positive
print "false positive frequency", false_positive / float ((MAP_SIZE * MAP_SIZE - SHIPS_PER_MAP * SHIP_SIZE) * EXPERIMENTS_COUNT)
print "false positive probability", EXPECTED_FALSE_POSITIVE

print ""

print "CHECK: true negative results + false positive results = ", (MAP_SIZE * MAP_SIZE - SHIPS_PER_MAP * SHIP_SIZE) * EXPERIMENTS_COUNT
print "", true_negative, " + ", false_positive, " = ", true_negative + false_positive
