import cv2
vidcap = cv2.VideoCapture('./source_videos/showingOff.mp4')  # open the video file
count = 0
success, image = vidcap.read()
while success:
    print("writing frame")
    cv2.imwrite(f"./frames/frame{count:08}.jpg", image)  # save in colour if colour
    count += 1
    success, image = vidcap.read()
