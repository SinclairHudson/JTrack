from filterpy.kalman import KalmanFilter
import numpy as np
import cv2

dim_x = 5
dim_z = 2


class Track:
    def __init__(self, initialx, initialy, temporal_frame=50):
        self.id = id(self)
        self.prev_states = []
        self.temporal_frame = 50
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.kf.x = np.expand_dims(
            np.array([initialx, initialy, 0, 0, -9.8]), axis=1)
        self.mia = 0
        self.age = 0
        self.kf.F = np.array([
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
        ], dtype=float)
        self.kf.H = np.array([
            [1, 0],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 0],
        ], dtype=float).transpose()
        self.kf.R = np.eye(dim_z, dim_z)

        self.kf.P *= 1000

    def update(self, measurement):
        self.mia -= 1
        self.kf.update(measurement)

    def predict(self):
        self.mia += 1
        self.age += 1
        self.prev_states.append(self.kf.x)
        if(len(self.prev_states) > self.temporal_frame):
            self.prev_states = self.prev_states[1:]
        self.kf.predict()

    def draw(self, image, min_age=5):
        if self.age >= min_age:
            image = cv2.circle(
                image, (int(self.kf.x[0]), int(self.kf.x[1])), 20, (0, 0, self.id % 256), -1)
        return image

    def drawTemporal(self, image, min_age=5):
        if self.age >= min_age:
            image = cv2.circle(
                image, (int(self.kf.x[0]), int(self.kf.x[1])), 20, (0, 0, self.id % 256), -1)
        return image

    def drawTemporalLines(self, image, colour=(0, 0, 255), min_age=5):
        if self.age >= min_age:
            for x in range(len(self.prev_states)-1):
                image = cv2.line(
                    image, (int(self.prev_states[x][0]), int(
                        self.prev_states[x][1])),
                    (int(self.prev_states[x+1][0]),
                     int(self.prev_states[x+1][1])),
                    colour, 4)
        return image

    def drawSpeedLines(self, image, min_age=5):
        if self.age >= min_age:
            for x in range(len(self.prev_states)-1):
                value = (int(self.prev_states[x][3]) + 127) % 256
                hsv = np.uint8([[[value,230,255]]])
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).tolist()[0][0]
                image = cv2.line(
                    image, (int(self.prev_states[x][0]), int(
                        self.prev_states[x][1])),
                    (int(self.prev_states[x+1][0]),
                     int(self.prev_states[x+1][1])),
                    bgr, 20)
        return image


def cleanTracks(tracks, mia_cutoff=4):
    cleaned = []
    for track in tracks:
        if track.mia < mia_cutoff:
            cleaned.append(track)
    return cleaned
