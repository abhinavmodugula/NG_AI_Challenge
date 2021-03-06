import datetime
import cv2
import mediapipe as mp
import time

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

"""
Code created for the Northrup Grumman AI Challenge
Team 6
Simon C., Abhinav M., Sameer H.

This code runs the live simulator where the webcam
input from any device can be used. This script will
first allow a user to select the regions in the first frame.
Then, we run a live detection of any hands that touch a surface.
When a hand touches a region of interest, that region will
chnage color from green to red. It will transition back after a set 
amount of time that the user can define.

The hand detection model is through the mediapipe ML API
"""

global regions, touch_map, touches, intregions
regions = []
intregions = []
touch_map = {}

mp_drawing = mp.solutions.drawing_utils #load in the detection API
mp_hands = mp.solutions.hands

W_NAME = "Single-Threaded Detection" #name for the displayed window

def start_select_roi(img):
    """
    Allows the user to select which areas of the image to
    monitor. Multiple regions can be selected
    """
    global touches
    touches = []

    def mouse_fn(event, x, y, flags, param):
        """
        Function to process the user's
        mouse inputs

        """
        global regions, touch_map

        if event != 1:  # left mouse button
            return

        # no previous touches or last touch is full? add a new touch
        if len(touches) == 0 or len(touches[-1]) == 4:
            touches.append([Point(x, y)])
        else:  # continue last touch
            touches[-1].append(Point(x, y))

        if len(touches[-1]) == 4:  # add touch to regions
            regions.append(Polygon(touches[-1]))
            intregions.append([(int(i.x), int(i.y)) for i in touches[-1]])
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
    return roi.contains(point)
    # if (point[0] < roi[2] and point[0] > roi[0]):
    #     if (point[1] < roi[3] and point[1] > roi[1]):
    #         return True
    # return False

def draw_region(image, region):  # draws a region on the screen
    r = region
    cv2.line(image, r[0], r[1], (0, 0, 255), 2)
    cv2.line(image, r[1], r[2], (0, 0, 255), 2)
    cv2.line(image, r[2], r[3], (0, 0, 255), 2)
    cv2.line(image, r[3], r[0], (0, 0, 255), 2)


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
#
cv2.namedWindow(W_NAME)

THRESHOLD = 25  # number of seconds after which touches don't count anymore

with open("points2.txt", "w") as file: #log file
    with mp_hands.Hands(
          static_image_mode=True,
          max_num_hands=2,
          min_detection_confidence=0.5) as hands:
        while vid.isOpened():
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            ret, image_np = vid.read()
            # image_np = cv2.flip(image_np, 1)
            try:
                image_np = cv2.cvtColor(cv2.flip(image_np, 1), cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            if not ret:
                print("tijoo")
                continue

            #if first frame, let user select region to monitor
            if first:
                first = False
                start_select_roi(image_np)
                # regions = select_roi(image_np)
                # for i in range(len(regions)):
                #     touch_map[i] = THRESHOLD + 1


            # Old detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

            # boxes, scores = hand_detector.detect_objects(image_np,
            #                                               detection_graph, sess)
            #
            #
            # # draw bounding boxes on frame
            # centers = hand_detector.draw_box_on_image(num_hands_detect, score_thresh,
            #                                  scores, boxes, im_width, im_height,
            #                                  image_np)
            # print(touches)
            # print(regions)
            image_np.flags.writeable = False  # makes detection more efficient
            result = hands.process(image_np)
            image_np.flags.writeable = True

            for i in touches:
                if len(i) != 4:  # draw the regions that are currently being created in red
                    for (a, b) in zip(i, i[1:]):
                        cv2.line(image_np, (int(a.x), int(a.y)), (int(b.x), int(b.y)), (255, 0, 0), 2)

            if result.multi_hand_landmarks:  # draw dots & lines on detected hands
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image_np, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            n = time.time()
            overlay = image_np.copy()
            for i, (ir, region) in enumerate(zip(intregions, regions)):
                if result.multi_hand_landmarks: # for each region, for each hand, check if it's in the region
                    for r in result.multi_hand_landmarks:
                        c = r.landmark[9]  # 9th landmark is the center of the hand
                        center = (int(c.x * im_width), int(c.y * im_height))
                        print(center)
                        cv2.circle(image_np, center, 20, (0, 0, 255), 10)
                        if (point_in_roi(Point(center), region)):
                            print(f"Hand in region {i}!!!")
                            touch_map[i] = n
                            file.write(f"{center}\n")

                t = int(((n - touch_map[i]) / THRESHOLD) * 255)  # color of the circle: linearly related to the given threshold
                if t > 255:
                    t = 255
                r = int(region.length // 12)  # min(region[2] - region[0], region[3] - region[1]) // 4  # circle only fills half of the region

                # cv2.rectangle(image_np, (region[0], region[1]), (region[2], region[3]), (255 - t, t, 0), -1)
                # cv2.rectangle(image_np, (region[0], region[1]), (region[2], region[3]), (0, 0, 255), 2)
                p = region.centroid
                cv2.circle(overlay, (int(p.x), int(p.y)), r, (255 - t, t, 0), -1)
                draw_region(image_np, ir)

            alpha = 0.4
            image_np = cv2.addWeighted(overlay, alpha, image_np, 1 - alpha, 0)  # overlay dots so it looks cooler :)
            #
            # # Calculate Frames per second (FPS)
            # num_frames += 1
            # elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            # fps = num_frames / elapsed_time

            # if (fps > 0):
                # Display FPS on frame
                # hand_detector.draw_fps_on_image("FPS : " + str(int(fps)),
                #                                 image_np)

            cv2.imshow(W_NAME,
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            # else:
                # print("frames processed: ", num_frames, "elapsed time: ",
                #       elapsed_time, "fps: ", str(int(fps)))

        vid.release()
        cv2.destroyAllWindows()
