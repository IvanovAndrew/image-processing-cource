__author__ = 'User'

import numpy as np
import numpy.random as random

p = 0.1
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

size = 10
BLACK = 0
WHITE = 1
PROCESSED = -1

EXPERIMENTS_COUNT = 1000
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

expected_true_positive =  (1 - p) * (1 - p * np.power(1 - p, 3))
expected_true_negative =  (1 - p) * (1 + p * np.power(1 - p, 3))
expected_false_positive = p * (1 - np.power(1 - p, 4))
expected_false_negative = p * (1 + np.power(1 - p, 4))

class Coordinate:
    def __init__(self, coordinate):
        x, y = coordinate
        self.x = x
        self.y = y

    def to_string(self):
        s = "[x=" + str(self.x) + " y=" + str(self.y)+"]"
        return s

    def __eq__(self, other):
        return isinstance(other, Coordinate) and (self.x == other.x) and self.y == other.y

class Ship:
    def __init__(self, path):
        self.coordinates = path

    def to_string(self):
        s = ""
        for coordinate in self.coordinates:
            s = s + coordinate.to_string() + " "
        return s

    def contained_in(self, other):
        if not isinstance(other, Ship):
            return False

        for coord1 in self.coordinates:
            flag = False
            for coord2 in other.coordinates:
                flag = flag or coord1 == coord2

            if flag == True:
                continue
            return False
        return True

def create_source_map(needShip):
    source_map = np.zeros([size, size], dtype=np.int)

    if needShip:
        x, y = random.randint(1, size), random.randint(1, size)
        source_map[x, y] = WHITE

        direction = random.randint(0, 5)

        if direction == LEFT:
            if y != 1:        source_map[x, y - 1] = WHITE
            else:             source_map[x, y + 1] = WHITE

        elif direction == UP:
            if x != size - 1: source_map[x + 1, y] = WHITE
            else:             source_map[x - 1, y] = WHITE

        elif direction == RIGHT:
            if y != size - 1: source_map[x, y + 1] = WHITE
            else:             source_map[x, y - 1] = WHITE

        else:  # direction == down
            if x != 1:        source_map[x - 1, y] = WHITE
            else:             source_map[x + 1, y] = WHITE

    return source_map

##### create map with noise #####
def create_noise_map(source_map):
    noise_map = np.copy(source_map)
    for i in range(size):
        for j in range(size):
            if random.random() < p:
                noise_map[i, j] = (noise_map[i, j] + WHITE) % 2;

    return noise_map

def get_available_neighborhoods(center, path):
    result = []
    coordinate = Coordinate((center.x + 1, center.y))
    if is_available(coordinate, path):
        result.append(coordinate)

    coordinate = Coordinate((center.x - 1, center.y))
    if is_available(coordinate, path):
        result.append(coordinate)

    coordinate = Coordinate((center.x, center.y + 1))
    if is_available(coordinate, path):
        result.append(coordinate)

    coordinate = Coordinate((center.x, center.y - 1))
    if is_available(coordinate, path):
        result.append(coordinate)

    return result

def is_available(coord, path):
    if coord.x < 0 or coord.x == size or coord.y < 0 or coord.y == size:
        return False

    for coordinate in path:
        if coordinate == coord:
            return False

    return True

""" search ship coordinates """
def find_neighborhoods(map, path, center):
    path.append(center)
    map[center.x, center.y] = PROCESSED

    neighborhoods = get_available_neighborhoods(center, path)

    for n in neighborhoods:
        if map[n.x, n.y] == WHITE:
            path = find_neighborhoods(map, path, n)

    return path

def find_ships(map):
    ships_list = []
    for i in range(size):
        for j in range(size):
            if map[i, j] == WHITE:
                coord = Coordinate((i, j))
                path = find_neighborhoods(map, [], coord)
                if len(path) > 1:
                    ship = Ship(path)
                    ships_list.append(ship)
                else:
                    map[i, j] = BLACK
    return ships_list

###ship exists
for i in range (EXPERIMENTS_COUNT):
    source_map = create_source_map(True)

    real_ship = find_ships(np.copy(source_map)).pop(0)

    noise_map = create_noise_map(source_map)

    ships = find_ships(noise_map)

    flag = False
    for ship in ships:
        if real_ship.contained_in (ship):
            flag = True

    if flag:
        true_positive += 1
    else:
        false_negative += 1

###ship doesn't exist
for i in range (EXPERIMENTS_COUNT):
    source_map = create_source_map(False)

    noise_map = create_noise_map(source_map)

    ships = find_ships(noise_map)

    if len(ships) > 0:
        false_positive += 1
    else:
        true_negative += 1

print "true positive results ", true_positive / float (EXPERIMENTS_COUNT)
print "false negative results ", false_negative / float (EXPERIMENTS_COUNT)
print ""

print "expected true_positive ", expected_true_positive
print "expected false_negative ", expected_false_negative

print "====="
print "false positive results ", false_positive / float (EXPERIMENTS_COUNT)
print "true negative results ", true_negative / float (EXPERIMENTS_COUNT)

print ""

print "expected false_positive ", expected_false_positive
print "expected true_negative ", expected_true_negative