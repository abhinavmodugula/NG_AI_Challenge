import datetime
import cv2
import numpy as np
import hand_detector

def select_roi(img):
    """
    Allows the user to select which areas of the image to
    monitor
    """
    r = cv2.selectROI("Select region", img, False, False)
    
    region = [r[0], r[1], r[0]+r[2], r[1]+r[3]]
    
    #cv2.rectangle(img, (r[0], r[1]), (r[2]+r[0], r[3]+r[1]), (255, 0, 0), 2)
    
    
    cv2.waitKey(0)
    
    return region

def get_center(box, im_width, im_height):
    #takes the output of the hand detection model and returns the center
    (left, right, top, bottom) = (box[1] * im_width, box[3] * im_width,
                                          box[0] * im_height, box[2] * im_height)
    p1 = (int(left), int(top))
    p2 = (int(right), int(bottom))
    c_x = (p2[1] - p1[0]) / 2
    c_y = (p2[0] - p1[1]) / 2
    center = (int(c_y), int(c_x))
    return center
    

detection_graph, sess = hand_detector.load_inference_graph()

#Args:
fps = 1
score_thresh = 0.2
video_source = 0 #device camera
width = 1520
height = 580
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
first = True #if first frame
region = None

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
    
    #if first frame, let user select region to monitor
    if first:
        first = False
        region = select_roi(image_np)
    

    # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
    # while scores contains the confidence for each of these boxes.
    # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

    boxes, scores = hand_detector.detect_objects(image_np,
                                                  detection_graph, sess)
    
    # See if any hands are in the region of interest
    for i in range(num_hands_detect):
        if scores[i] > score_thresh:
            box = boxes[i]
            #draw center to test
            center = get_center(box, im_width, im_height)
            print("Hand center")
            print(center)
            cv2.circle(image_np, center, radius=2, color=(0, 0, 255), thickness=4)

    # draw bounding boxes on frame
    hand_detector.draw_box_on_image(num_hands_detect, score_thresh,
                                     scores, boxes, im_width, im_height,
                                     image_np)
    
    cv2.rectangle(image_np, (region[0], region[1]), (region[2], region[3]), (255, 0, 0), 2)

    # Calculate Frames per second (FPS)
    num_frames += 1
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time

    if (fps > 0):
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
