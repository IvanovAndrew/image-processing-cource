__author__ = 'Andrew'

import numpy as np
import cv2
import math


VERTICAL_SIZE = 180
HORIZONTAL_SIZE = VERTICAL_SIZE / 2

FRAMES_COUNT = 10
BLACK = [0, 0, 0]
GRAY = [128, 128, 128]
WHITE = [255, 255, 255]
GREEN = [0, 255, 0]

X_START_POSITION = 1
Y_START_POSITION = 1

HORIZONTAL_SPEED = 3
VERTICAL_ACCELERATION = 4

RADIUS = 35

source_map = np.empty([HORIZONTAL_SIZE, VERTICAL_SIZE, 3], dtype=np.uint8)
tracks_map = np.empty([HORIZONTAL_SIZE, VERTICAL_SIZE, 3], dtype=np.uint8)
prediction_map = np.empty([HORIZONTAL_SIZE, VERTICAL_SIZE, 3], dtype=np.uint8)
black_map = np.empty([HORIZONTAL_SIZE, VERTICAL_SIZE, 3], dtype=np.uint8)

for i in range(HORIZONTAL_SIZE):
     for j in range(VERTICAL_SIZE):
         source_map[i, j] = BLACK
         tracks_map[i, j] = BLACK
         prediction_map[i, j] = BLACK
         black_map [i, j] = BLACK

### Create Kalman filter
# first param: Dimensionality of the state
# second param: Dimensionality of the measurement
# third param: Dimensionality of the control vector?
kalman = cv2.cv.CreateKalman(4, 2, 0)
kalman_measurement = cv2.cv.CreateMat(2, 1, cv2.CV_32FC1)

brick_centers = []
predicted_centers = []

def get_next_center(time):

    new_x_center = X_START_POSITION + HORIZONTAL_SPEED * time
    new_y_center = Y_START_POSITION + VERTICAL_ACCELERATION / 2 * time * time
    return new_x_center, new_y_center

def set_kalman_filter(point):

    x, y = point
    # set previous state for prediction
    kalman.state_pre[0,0]  = x
    kalman.state_pre[1,0]  = y
    # kalman.state_pre[2,0]  = 0#HORIZONTAL_SPEED
    # kalman.state_pre[3,0]  = 0 #VERTICAL_ACCELERATION
    print "kalman states"
    print np.asanyarray(kalman.state_pre)

    # set kalman transition matrix
    kalman.transition_matrix[0,0] = 1
    kalman.transition_matrix[0,1] = 0
    kalman.transition_matrix[0,2] = 1
    kalman.transition_matrix[0,3] = 0
    kalman.transition_matrix[1,0] = 0
    kalman.transition_matrix[1,1] = 1
    kalman.transition_matrix[1,2] = 0
    kalman.transition_matrix[1,3] = 1
    kalman.transition_matrix[2,0] = 0
    kalman.transition_matrix[2,1] = 0
    kalman.transition_matrix[2,2] = 1
    kalman.transition_matrix[2,3] = 0
    kalman.transition_matrix[3,0] = 0
    kalman.transition_matrix[3,1] = 0
    kalman.transition_matrix[3,2] = 0
    kalman.transition_matrix[3,3] = 1

    # set Kalman Filter
    cv2.cv.SetIdentity(kalman.measurement_matrix, cv2.cv.RealScalar(1))
    cv2.cv.SetIdentity(kalman.process_noise_cov, cv2.cv.RealScalar(1e-5))
    cv2.cv.SetIdentity(kalman.measurement_noise_cov, cv2.cv.RealScalar(1e-5))
    cv2.cv.SetIdentity(kalman.error_cov_post, cv2.cv.RealScalar(1))

def predict_kalman():

    kalman_prediction = cv2.cv.KalmanPredict(kalman)
    return (kalman_prediction[0,0], kalman_prediction[1,0])

def correct_kalman():

    kalman_estimated = cv2.cv.KalmanCorrect(kalman, kalman_measurement)
    return (kalman_estimated[0,0], kalman_estimated[1,0])

def calculate_error (predicted, real):

    pred_x, pred_y = predicted
    actual_x, actual_y = real

    temp_x = (pred_x - actual_x) * (pred_x - actual_x)
    temp_y = (pred_y - actual_y) * (pred_y - actual_y)

    return math.sqrt (temp_x + temp_y)

def plot_point(image, point, color):

    x, y = point
    image[x, y] = color
    image[x + 1, y] = color
    image[x, y + 1] = color
    image[x + 1, y + 1] = color

    return image

def try_find_brick(image, point, color):

    x, y = point
    xLowerBound = max(0, int(x - RADIUS))
    xUpperBound = min(VERTICAL_SIZE, int(x + RADIUS))

    yLowerBound = max(0, int(y - RADIUS))
    yUpperBound = min(VERTICAL_SIZE, int(y + RADIUS))

    xCoords = []
    yCoords = []

    is_exist = False

    for i in range(xLowerBound, xUpperBound + 1):
        for j in range(yLowerBound, yUpperBound + 1):
            if np.array_equal(image[i, j], color):
                xCoords.append(i)
                yCoords.append(j)
                is_exist = True

    if is_exist:
        x_mass_center = np.sum(xCoords) / len(xCoords)
        y_mass_center = np.sum(yCoords) / len(yCoords)
    else:
        x_mass_center = 0
        y_mass_center = 0
    return is_exist, (x_mass_center, y_mass_center)

def predict_brick(mass_center):

    set_kalman_filter(mass_center)

    x, y = mass_center
    kalman_measurement[0, 0] = x
    kalman_measurement[1, 0] = y

    predict_kalman()
    corrected = correct_kalman()

    return corrected

mass_center = X_START_POSITION, Y_START_POSITION
predicted_center = (0, 0)

errors_sum = 0

def plot_in_result_map(map, real, predicted):

    x, y = real
    real_points = [real, (x + 1, y), (x, y + 1), (x + 1, y + 1)]

    x, y = predicted
    predicted_points = [predicted, (x + 1, y), (x, y + 1), (x + 1, y + 1)]

    for point in real_points:
        p, q = point

        index = -1
        count = 0
        for other in predicted_points:
            s, t = other
            s, t = int(s), int(t)
            if p == s and q == t:
                index = count
                break
            count = count + 1

        if index > -1:
            map[p, q] = GREEN
            predicted_points.pop(index)
        else:
            map[p, q] = WHITE

    for point in predicted_points:
        p, q = point
        map[p, q] = GRAY

for t in range(FRAMES_COUNT):

    print "FRAME", t
    x, y = get_next_center(t)

    plot_point(tracks_map, (x, y), WHITE)
    new_frame = plot_point(np.copy(black_map), (x, y), WHITE)

    is_success, mass_center = try_find_brick(new_frame, predicted_center, WHITE)
    if is_success:
        print "found center", mass_center
        predicted_center = predict_brick(mass_center)

        print "predicted center", predicted_center
        plot_in_result_map(source_map, (x, y), predicted_center)

        errors_sum += calculate_error(predicted_center, mass_center)
        print ""

    else:
        print "brick isn't found. Try increase radius"
        break

def transpose_map(map):

    res = np.empty([VERTICAL_SIZE, HORIZONTAL_SIZE, 3], dtype=np.uint8)

    for i in range(VERTICAL_SIZE):
        for j in range(HORIZONTAL_SIZE):
            res[i, j] = map[j, i]

    return res

source_map = transpose_map(source_map)
tracks_map = transpose_map(tracks_map)
prediction_map = transpose_map(prediction_map)

print "average errors: ", (errors_sum / float(FRAMES_COUNT))
cv2.imwrite("real track.png", tracks_map)
cv2.imwrite("prediction track.png", prediction_map)
cv2.imwrite("result image.png", source_map)