## http://stackoverflow.com/questions/13901997/kalman-2d-filter-in-python
##

__author__ = 'Andrew'

import numpy as np
import cv2
import math

IMAGE_SIZE = 200

FRAMES_COUNT = 9
BLACK = [0, 0, 0]
GRAY = [128, 128, 128]
WHITE = [255, 255, 255]

X_START_POSITION = 1
Y_START_POSITION = 1

HORIZONTAL_SPEED = 3
VERTICAL_ACCELERATION = 4

RADIUS = 36

source_map = np.empty([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8)
tracks_map = np.empty([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8)
prediction_map = np.empty([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8)
black_map = np.empty([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8)

real_mass_centers = []

for i in range(IMAGE_SIZE):
     for j in range(IMAGE_SIZE):
         source_map[i, j] = BLACK
         tracks_map[i, j] = BLACK
         prediction_map[i, j] = BLACK
         black_map [i, j] = BLACK

### Create Kalman filter
# first param: Dimensionality of the state
# second param: Dimensionality of the measurement
# third param: Dimensionality of the control vector?
kalman = cv2.cv.CreateKalman(4, 2, 0)
kalman_state = cv2.cv.CreateMat(4, 1, cv2.cv.CV_32FC1)
kalman_process_noise = cv2.cv.CreateMat(4, 1, cv2.CV_32FC1)
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
    #kalman.state_pre[2,0]  = 0#HORIZONTAL_SPEED
    #kalman.state_pre[3,0]  = 0 #VERTICAL_ACCELERATION
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
    cv2.cv.SetIdentity(kalman.process_noise_cov, cv2.cv.RealScalar(1))
    cv2.cv.SetIdentity(kalman.measurement_noise_cov, cv2.cv.RealScalar(1))
    cv2.cv.SetIdentity(kalman.error_cov_post, cv2.cv.RealScalar(1))

def predict_kalman():

    kalman_prediction = cv2.cv.KalmanPredict(kalman)
    print "prediction"
    print np.asanyarray(kalman_prediction)
    return (kalman_prediction[0,0], kalman_prediction[1,0])

def correct_kalman():

    kalman_estimated = cv2.cv.KalmanCorrect(kalman, kalman_measurement)
    print "correct"
    print np.asanyarray(kalman_estimated)

    return (kalman_estimated[0,0], kalman_estimated[1,0])

def change_measurement(point):

    x, y = point
    kalman_measurement[0, 0] = x
    kalman_measurement[1, 0] = y

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

def find_brick(image, point, color):

    x, y = point
    xLowerBound = max(0, int(x - RADIUS))
    xUpperBound = min(IMAGE_SIZE, int(x + RADIUS))

    yLowerBound = max(0, int(y - RADIUS))
    yUpperBound = min(IMAGE_SIZE, int(y + RADIUS))

    xCoords = []
    yCoords = []

    for i in range(xLowerBound, xUpperBound):
        for j in range(yLowerBound, yUpperBound):
            if np.array_equal(image[i, j], color):
                xCoords.append(i)
                yCoords.append(j)

    x_mass_center = np.sum(xCoords) / len(xCoords)
    y_mass_center = np.sum(yCoords) / len(yCoords)

    return x_mass_center, y_mass_center

def predict_brick(mass_center):
    set_kalman_filter(mass_center)
    change_measurement(mass_center)

    predicted = predict_kalman()
    corrected = correct_kalman()

    return corrected

mass_center = X_START_POSITION, Y_START_POSITION

errors_sum = 0

for t in range(1, FRAMES_COUNT):

    predicted_center = predict_brick(mass_center)
    plot_point(prediction_map, predicted_center, GRAY)
    print "predicted center", predicted_center

    x, y = get_next_center(t)
    real_mass_centers.append((x, y))
    plot_point(tracks_map, (x, y), WHITE)
    new_frame = plot_point(np.copy(black_map), (x, y), WHITE)

    mass_center = find_brick(new_frame, predicted_center, WHITE)
    print "found center", mass_center
    print "real center", (x, y)

    print ""

    errors_sum += calculate_error(predicted_center, mass_center)

source_map = np.transpose(source_map)
#tracks_map = np.transpose(tracks_map)
#prediction_map = np.transpose(prediction_map)

print "average errors: ", (errors_sum / float(FRAMES_COUNT - 1))
cv2.imwrite("real track.png", tracks_map)
cv2.imwrite("prediction track.png", prediction_map)