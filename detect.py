import cv2
import time
import numpy as np


def detectBalls(image, hsv_low, hsv_high, min_area=200):
    lowerOrange = np.array([0, 170, 117])
    upperOrange = np.array([36, 256, 256])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    mask = 255 - mask  # invert

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 256

    params.filterByArea = True
    params.minArea = min_area

    params.filterByCircularity = False
    params.minCircularity = 0.1

    params.filterByConvexity = False
    params.minConvexity = 0.87

    params.filterByInertia = False
    params.minInertiaRatio = 0.01
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(mask)

    return keypoints
