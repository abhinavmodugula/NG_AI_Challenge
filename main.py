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
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    ret, image_np = vid.read()
    # image_np = cv2.flip(image_np, 1)
    try:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")

    # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
    # while scores contains the confidence for each of these boxes.
    # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

    boxes, scores = detector_utils.detect_objects(image_np,
                                                  detection_graph, sess)

    # draw bounding boxes on frame
    hand_detector.draw_box_on_image(num_hands_detect, score_thresh,
                                     scores, boxes, im_width, im_height,
                                     image_np)

    # Calculate Frames per second (FPS)
    num_frames += 1
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time

    if (args.display > 0):
        # Display FPS on frame
        hand_detector.draw_fps_on_image("FPS : " + str(int(fps)),
                                             image_np)

        cv2.imshow('Single-Threaded Detection',
                   cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    else:
        print("frames processed: ", num_frames, "elapsed time: ",
              elapsed_time, "fps: ", str(int(fps)))
vid.release()
cv2.destroyAllWindows()
