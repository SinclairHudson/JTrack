import cv2
import math
import numpy as np
from detect import detectBalls
from track import Track, cleanTracks
from scipy import optimize


def euclideanDistance(track, detection):
    return math.sqrt((track.kf.x[0] - detection.pt[0])**2 + (track.kf.x[1] - detection.pt[1])**2)


vidcap = cv2.VideoCapture('./source_videos/showingOff.mp4')  # open the video file
count = 0
success, image = vidcap.read()
tracks = []
while success:

    for track in tracks:
        track.predict()  # advance the kalman filters

    lowerOrange = np.array([0, 190, 117])
    upperOrange = np.array([30, 256, 256])

    detections = detectBalls(image, lowerOrange, upperOrange, min_area=250)
    num_detections = len(detections)
    num_tracks = len(tracks)

    if num_tracks == 0:
        for detection in detections:
            tracks.append(Track(detection.pt[0], detection.pt[1]))
        count += 1
        success, image = vidcap.read()
        continue

    if num_detections == 0:
        count += 1
        success, image = vidcap.read()
        continue  # pretty easy here

    cost_matrix = np.zeros((len(tracks), len(detections)))
    for x, track in enumerate(tracks):  # height
        for y, detection in enumerate(detections):  # width
            cost_matrix[x][y] = euclideanDistance(track, detection)

    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)

    matching = np.stack((row_ind, col_ind), axis=1)

    filtered_matching = []
    for track_index, detection_index in matching:
        if(cost_matrix[track_index][detection_index] < 200):
            filtered_matching.append([track_index, detection_index])

    # we have a matching now
    filtered_matching = np.array(filtered_matching)
    associated_detections = set(filtered_matching[:, 1])
    associated_tracks = set(filtered_matching[:, 0])

    for track_index, detection_index in filtered_matching:
        tracks[track_index].update(np.array(detections[detection_index].pt))


    unassociated_detections = list(
        set(range(len(detections))) - set(associated_detections))

    unassociated_tracks = list(
        set(range(len(detections))) - set(associated_detections))

    for s in unassociated_detections:
        tracks.append(Track(detections[s].pt[0], detections[s].pt[1]))

    tracks = cleanTracks(tracks)

    im_with_keypoints = cv2.drawKeypoints(image, detections, np.array([]),
                                          (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    for track in tracks:
        track.drawSpeedLines(im_with_keypoints)

    height, width = image.shape[:2]
    dim = (width // 3, height // 3)
    cv2.imshow("tracks", cv2.resize(im_with_keypoints,
                                    dim, interpolation=cv2.INTER_AREA))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count += 1
    success, image = vidcap.read()
