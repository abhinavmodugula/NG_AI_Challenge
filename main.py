import datetime
import cv2
import numpy as np
import hand_detector
import time

global regions, touch_map, touches
regions = []
touch_map = {}

W_NAME = "Single-Threaded Detection"

def start_select_roi(img):
    """
    Allows the user to select which areas of the image to
    monitor
    """
    global touches
    touches = []

    def mouse_fn(event, x, y, flags, param):
        global regions, touch_map

        if event != 1:
            return

        if len(touches) == 0 or len(touches[-1]) == 4:
            touches.append([(x, y)])
        else:
            touches[-1].append((x, y))

        if len(touches[-1]) == 4:
            regions.append(touches[-1])
            touch_map[len(regions) - 1] = THRESHOLD + 1

    cv2.setMouseCallback(W_NAME, mouse_fn)

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

def point_in_roi(point, roi):
    """
    Returns whether point (x, y) is in a roi
    
    """
    if (point[0] < roi[2] and point[0] > roi[0]):
        if (point[1] < roi[3] and point[1] > roi[1]):
            return True
    return False

def draw_region(image, region):
    cv2.line(image, region[0], region[1], (0, 0, 255), 2)
    cv2.line(image, region[1], region[2], (0, 0, 255), 2)
    cv2.line(image, region[2], region[3], (0, 0, 255), 2)
    cv2.line(image, region[3], region[0], (0, 0, 255), 2)

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

# cv2.namedWindow("app")
vid = cv2.VideoCapture(0)
# vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

start_time = datetime.datetime.now()
num_frames = 0
im_width, im_height = (vid.get(3), vid.get(4))

# max number of hands we want to detect/track
num_hands_detect = 4
first = True #if first frame
# regions = None

camera_adjust_frames = 200

if (vid.isOpened()== False):
     print("Error opening video stream or file")

"""
DOES NOT WORK AT THE MOMENT
while True:
    camera_adjust_frames -= 1
    ret, image_np = vid.read()
    cv2.imshow('Adjust Camera',
                   cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    if camera_adjust_frames == 0:
        cv2.destroyAllWindows()
        break
"""

cv2.namedWindow(W_NAME)

THRESHOLD = 5  # number of seconds after which touches don't count anymore

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
        start_select_roi(image_np)
    #     regions = select_roi(image_np)
    #     for i in range(len(regions)):
    #         touch_map[i] = THRESHOLD + 1
    

    # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
    # while scores contains the confidence for each of these boxes.
    # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

    boxes, scores = hand_detector.detect_objects(image_np,
                                                  detection_graph, sess)
    

    # draw bounding boxes on frame
    centers = hand_detector.draw_box_on_image(num_hands_detect, score_thresh,
                                     scores, boxes, im_width, im_height,
                                     image_np)

    for i in touches:
        if len(i) != 4:
            for (a, b) in zip(i, i[1:]):
                cv2.line(image_np, a, b, (255, 0, 0), 2)

    n = time.time()
    overlay = image_np.copy()
    for i, region in enumerate(regions):
        # for center in centers:
        #     cv2.circle(image_np, center, 20, (0, 0, 255), 10)
        #     if (point_in_roi(center, region)):
        #         print(f"Hand in region {i}!!!")
                # touch_map[i] = n

        t = int(((n - touch_map[i]) / THRESHOLD) * 255)
        if t > 255:
            t = 255
        r = 50  # min(region[2] - region[0], region[3] - region[1]) // 4  # circle only fills half of the region

        # cv2.rectangle(image_np, (region[0], region[1]), (region[2], region[3]), (255 - t, t, 0), -1)
        # cv2.rectangle(image_np, (region[0], region[1]), (region[2], region[3]), (0, 0, 255), 2)
        # cv2.circle(overlay, ((region[0] + region[2]) // 2, (region[1] + region[3]) // 2), r, (255 - t, t, 0), -1)
        draw_region(image_np, region)

    alpha = 0.4
    image_np = cv2.addWeighted(overlay, alpha, image_np, 1 - alpha, 0)

    # Calculate Frames per second (FPS)
    num_frames += 1
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time

    if (fps > 0):
        # Display FPS on frame
        hand_detector.draw_fps_on_image("FPS : " + str(int(fps)),
                                        image_np)

        cv2.imshow(W_NAME,
                   cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    else:
        print("frames processed: ", num_frames, "elapsed time: ",
              elapsed_time, "fps: ", str(int(fps)))

vid.release()
cv2.destroyAllWindows()
