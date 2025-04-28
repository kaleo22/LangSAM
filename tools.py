import cv2
import numpy as np
import json

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

# Callback when the client connects to the broker
def on_connect(client, userdata, flags, rc, topic):
    if rc == 0:
        print("Connected to MQTT Broker!")
        # Subscribe to a topic
        client.subscribe(topic)
    else:
        print(f"Failed to connect, return code {rc}")

# Callback when a message is received
def on_message(client, userdata, msg):
    payload = msg.payload()

    if payload[:2] == b'\xff\xd8' and payload[-2:] == b'\xff\xd9':
        # Decode the payload
        nparr = np.frombuffer(payload, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image_np
    else:
        data =  json.loads(payload.decode())
        return data
    print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
