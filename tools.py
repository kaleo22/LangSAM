import cv2
import numpy as np

def kalman_init():
    #Initialize Kalman filter
    kalman = cv2.KalmanFilter(2, 2)
    
    kalman.measurementMatrix = np.array([[1, 0], [0, 1]], np.float32)

    kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)

    kalman.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.1

    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.01

    kalman.errorCovPost = np.array([[1, 0], [0, 1]], np.float32)

    kalman.statePost = np.array([[0], [0]], np.float32)
    
    return kalman


def kalman_predict(kalman, ROI):
    # Predict the next state
    prediction= kalman.predict()
    
    # Update the Kalman filter with the new measurement
    kalman.correct(ROI)
    
    return prediction