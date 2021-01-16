import cv2
import numpy as np
from detect import detectBalls
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

dim_x = 5
dim_z = 2
kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
kf.x = np.expand_dims(np.array([500, 500, 0, 0, -9.8]), axis=1)
assert kf.x.shape == (dim_x,1)
kf.F = np.array([
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1],
    ], dtype=float)
assert kf.F.shape == (dim_x, dim_x)
kf.H = np.array([
    [1, 0],
    [0, 1],
    [0, 0],
    [0, 0],
    [0, 0],
    ], dtype=float).transpose()
assert kf.H.shape == (dim_z, dim_x)
kf.R = np.eye(dim_z,dim_z)

kf.P *= 1000

vidcap = cv2.VideoCapture('./source_videos/mono3.mp4')  # open the video file
count = 0
success, image = vidcap.read()
while success:
    keypoints = detectBalls(image)
    print(len(keypoints))
    if(len(keypoints) > 0):
        measurement = keypoints[0].pt
        z = np.array(measurement)
    kf.predict()
    pred = (int(kf.x[0]), int(kf.x[1]))

    kf.update(z)

    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]),
                                      (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints = cv2.circle(im_with_keypoints, pred, 20, (0,0,255), -1)
    height, width = image.shape[:2]
    dim = (width // 3, height // 3)
    cv2.imshow("tracks", cv2.resize(im_with_keypoints, dim, interpolation=cv2.INTER_AREA))
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    count += 1
    success, image = vidcap.read()

