__author__ = 'Andrew'

import numpy as np
import numpy.random as random
import cv2
import math

IMAGE_SIZE = 100
BRICK_SIZE = 3

FRAMES_COUNT = 5
BLACK = 0
GRAY = 128
WHITE = 255


X_START_POSITION = 1
Y_START_POSITION = 1

HORIZONTAL_SPEED = 5
VERTICAL_ACCELERATION = 10

# test = cv2.imread("cat.jpg")
# print type (test[0, 0])
##print "ok"

source_map = np.empty([IMAGE_SIZE, IMAGE_SIZE], dtype=np.uint8)

for i in range(IMAGE_SIZE):
     for j in range(IMAGE_SIZE):
         source_map[i, j] = BLACK

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

def plot_bricks(points, color):
    for coord in points:
        x, y = coord

        source_map[x - 1, y - 1] = color
        source_map[x - 1, y]     = color
        source_map[x - 1, y + 1] = color

        source_map[x, y - 1] = color
        source_map[x, y]     = color
        source_map[x, y + 1] = color

        source_map[x + 1, y - 1] = color
        source_map[x + 1, y]     = color
        source_map[x + 1, y + 1] = color

def set_kalman_filter(x, y):
    # set previous state for prediction
    kalman.state_pre[0,0]  = x
    kalman.state_pre[1,0]  = y
    kalman.state_pre[2,0]  = 0
    kalman.state_pre[3,0]  = 0

    # set kalman transition matrix
    kalman.transition_matrix[0,0] = 1
    kalman.transition_matrix[0,1] = 0
    kalman.transition_matrix[0,2] = HORIZONTAL_SPEED
    kalman.transition_matrix[0,3] = 0
    kalman.transition_matrix[1,0] = 0
    kalman.transition_matrix[1,1] = 1
    kalman.transition_matrix[1,2] = 0
    kalman.transition_matrix[1,3] = VERTICAL_ACCELERATION ## need some value
    kalman.transition_matrix[2,0] = 0
    kalman.transition_matrix[2,1] = 0
    kalman.transition_matrix[2,2] = 1
    kalman.transition_matrix[2,3] = 0
    kalman.transition_matrix[3,0] = 0
    kalman.transition_matrix[3,1] = 0
    kalman.transition_matrix[3,2] = 0
    kalman.transition_matrix[3,3] = 1

    # set Kalman Filter
    cv2.cv.SetIdentity(kalman.measurement_matrix, cv2.cv.RealScalar(0.56))
    cv2.cv.SetIdentity(kalman.process_noise_cov, cv2.cv.RealScalar(0.1))
    cv2.cv.SetIdentity(kalman.measurement_noise_cov, cv2.cv.RealScalar(1e-1))
    cv2.cv.SetIdentity(kalman.error_cov_post, cv2.cv.RealScalar(0.1))

def predict_kalman ():
    kalman_prediction = cv2.cv.KalmanPredict(kalman)
    return (kalman_prediction[0,0], kalman_prediction[1,0])

def correct_kalman():
    kalman_estimated = cv2.cv.KalmanCorrect(kalman, kalman_measurement)
    return (kalman_estimated[0,0], kalman_estimated[1,0])

def change_measurement(x, y):
    kalman_measurement[0, 0] = x
    kalman_measurement[1, 0] = y

def calculate_error (predicted, real):
    pred_x, pred_y = predicted
    actual_x, actual_y = real

    temp_x = (pred_x - actual_x) * (pred_x - actual_x)
    temp_y = (pred_y - actual_y) * (pred_y - actual_y)

    return math.sqrt (temp_x + temp_y)


x, y = X_START_POSITION, Y_START_POSITION
brick_centers.append((x, y))
errors_sum = 0

for t in range(1, FRAMES_COUNT):
    set_kalman_filter(x, y)
    change_measurement(x, y)

    predicted = predict_kalman()
    corrected = correct_kalman()

    predicted_centers.append(corrected)

    x, y = get_next_center(t)
    brick_centers.append((x, y))

    print "corrected (", corrected, ")"

    print "real (", (x, y), ")"
    error = calculate_error(corrected, (x, y))
    print "error ", error

    errors_sum += error
    print ""

plot_bricks(predicted_centers, GRAY)
plot_bricks(brick_centers, WHITE)

source_map = np.transpose(source_map)

print "errors: ", errors_sum
print "average errors: ", (errors_sum / float(len(predicted_centers)))
black_image = cv2.cvtColor(source_map, cv2.COLOR_GRAY2BGR)
cv2.imwrite("kalman.png", black_image)


print "ok"