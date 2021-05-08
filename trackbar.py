import cv2
import numpy as np


def callback(x):
    pass


def hsv_mask(img_path):

    frame = cv2.imread(img_path)
    cv2.namedWindow('trackbars')

    # inital values
    ilowH = 0
    ihighH = 4
    ilowS = 121
    ihighS = 255
    ilowV = 69
    ihighV = 255
    areaL = 191
    areaH = 938
    kernel = 5

    # create trackbars for color change
    cv2.createTrackbar('lowH', 'trackbars', ilowH, 180, callback)
    cv2.createTrackbar('highH', 'trackbars', ihighH, 180, callback)

    cv2.createTrackbar('lowS', 'trackbars', ilowS, 255, callback)
    cv2.createTrackbar('highS', 'trackbars', ihighS, 255, callback)

    cv2.createTrackbar('lowV', 'trackbars', ilowV, 255, callback)
    cv2.createTrackbar('highV', 'trackbars', ihighV, 255, callback)

    cv2.createTrackbar('areaL', 'trackbars', areaL, 10_000, callback)
    cv2.createTrackbar('areaH', 'trackbars', areaH, 10_000, callback)
    cv2.createTrackbar("gKernel", 'trackbars', kernel, 50, callback)

    while True:
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ilowH = cv2.getTrackbarPos('lowH', 'trackbars')
        ihighH = cv2.getTrackbarPos('highH', 'trackbars')
        ilowS = cv2.getTrackbarPos('lowS', 'trackbars')
        ihighS = cv2.getTrackbarPos('highS', 'trackbars')
        ilowV = cv2.getTrackbarPos('lowV', 'trackbars')
        ihighV = cv2.getTrackbarPos('highV', 'trackbars')
        ilowV = cv2.getTrackbarPos('lowV', 'trackbars')
        ihighV = cv2.getTrackbarPos('highV', 'trackbars')
        areaL = cv2.getTrackbarPos('areaL', 'trackbars')
        areaH = cv2.getTrackbarPos('areaH', 'trackbars')
        kernel = cv2.getTrackbarPos('kernel', 'trackbars')
        lower_hsv = np.array([ilowH, ilowS, ilowV])
        higher_hsv = np.array([ihighH, ihighS, ihighV])
        mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
        mask = 255 - mask  # invert

        if kernel % 2 == 0:  # if even, make odd
            kernel +=1

        mask = cv2.GaussianBlur(mask,(5,5), 0)


        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 0;
        params.maxThreshold = 256;

        params.filterByArea = True
        params.minArea = areaL
        params.maxArea = areaH

        params.filterByCircularity = False
        params.minCircularity = 0.0
        params.maxCircularity = 0.5

        params.filterByConvexity = False
        params.minConvexity = 0.87

        params.filterByInertia = False
        params.minInertiaRatio = 0.01

        detector = cv2.SimpleBlobDetector_create(params)
        # mask = cv2.GaussianBlur(mask, (11, 11), 0)
        keypoints = detector.detect(mask)
        print(keypoints)

        detectframe = cv2.drawKeypoints(frame, keypoints, np.array([]),
        (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        dim = (width, height)
        cv2.imshow('trackbars', cv2.resize(mask, dim, interpolation=cv2.INTER_AREA))
        cv2.imshow('original image', cv2.resize(detectframe, dim, interpolation=cv2.INTER_AREA))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return lower_hsv, higher_hsv


if __name__ == '__main__':
    hsv_mask("frames/frame00000400.jpg")
