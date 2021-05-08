import cv2
import math
import numpy as np
from detect import detectBalls, detectClubs
from track import Track, cleanTracks
from scipy import optimize
import os


def euclideanDistance(track, detection):
    return math.sqrt((track.kf.x[0] - detection.pt[0])**2 + (track.kf.x[1] - detection.pt[1])**2)

def filter_matching(matching, cost_matrix, max_speed=200):
    filtered_matching = []
    for track_index, detection_index in matching:
        if(cost_matrix[track_index][detection_index] < max_speed):
            filtered_matching.append([track_index, detection_index])

    return filtered_matching

def TrackingJugglingVideo(vidpath, lowHSV, highHSV, min_area=250, framerate=60,
                          show=False, outfile="output.mp4"):
    vidcap = cv2.VideoCapture(vidpath)  # open the video file
    count = 0
    success, image = vidcap.read()
    tracks = []
    print("tracking and drawing the frames... this may take some time.")
    os.system("rm -rf output")  # clean up
    os.system("mkdir output")  # create directory
    peaks = []
    while success:

        for track in tracks:
            track.predict()  # advance the kalman filters

        detections = detectBalls(
            image, lowHSV, highHSV)
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

        filtered_matching = filter_matching(matching, cost_matrix)

        if len(matching) == 0:
            # if there are no reasonable matches, just ignore and go next
            count += 1
            success, image = vidcap.read()
            continue  # pretty easy here


        # we have a matching now
        filtered_matching = np.array(filtered_matching)
        try:
            associated_detections = set(filtered_matching[:, 1])
        except:
            associated_detections = set()

        try:
            associated_tracks = set(filtered_matching[:, 0])
        except:
            associated_tracks = set()

        for track_index, detection_index in filtered_matching:
            tracks[track_index].update(
                np.array(detections[detection_index].pt))

        unassociated_detections = list(
            set(range(len(detections))) - set(associated_detections))

        unassociated_tracks = list(
            set(range(len(detections))) - set(associated_detections))

        for s in unassociated_detections:
            tracks.append(Track(detections[s].pt[0], detections[s].pt[1]))

        tracks = cleanTracks(tracks)

        # image = cv2.drawKeypoints(image, detections, np.array([]),
        # (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        for track in tracks:
            # image = track.drawSpeedLines(image)
            image = track.drawContrail(image)

            # peak = track.peak(max_y=800)
            # if not peak is None:
                # peaks.append(peak)

        # for peak in peaks:
            # image = cv2.circle(image, (int(peak[0]), int(peak[1])), 20, (255, 0, 0), -1)


        cv2.imwrite(f"./output/frame{count:08}.jpg", image)

        if show:
            height, width = image.shape[:2]
            dim = (width // 3, height // 3)
            cv2.imshow("tracks", cv2.resize(image,
                                            dim, interpolation=cv2.INTER_AREA))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        count += 1
        success, image = vidcap.read()

    os.system(f"ffmpeg -framerate {framerate} -i ./output/frame%08d.jpg {outfile}")
    os.system("rm -rf output")  # clean up


lowerOrange = np.array([0, 192, 117])
upperOrange = np.array([24, 256, 256])

deeplowOrange = np.array([0, 121, 69])
deephighOrange = np.array([4, 256, 256])

lowdeepblue = np.array([97, 118, 64])
highdeepblue = np.array([115, 255, 255])

lowaqua = np.array([46, 76, 20])
highaqua = np.array([98, 255, 255])

TrackingJugglingVideo("./source_videos/7orange.mp4", deeplowOrange,
                      deephighOrange, min_area=50, show=True)
