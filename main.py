import datetime
import cv2
import numpy as np
import hand_detector

detection_graph, sess = hand_detector.load_inference_graph()

#Args:
score_thresh = 0.2
video_source = 0 #device camera
width = 320
height = 180
display = 1
num_workers = 4
queue_size = 5

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

start_time = datetime.datetime.now()
num_frames = 0
im_width, im_height = (vid.get(3), vid.get(4))

# max number of hands we want to detect/track
num_hands_detect = 4

if (vid.isOpened()== False):
     print("Error opening video stream or file")

while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    print(frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()