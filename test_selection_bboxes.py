# -*- coding: utf-8 -*-
"""
Script to test the UI component only
of the code that lets a user select multiple
regions from a frame.

This code has been migrated to main.py
"""


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
    
    cv2.rectangle(img, (region[0], region[1]), (region[2], region[3]), (255, 0, 0), 2)
    cv2.imshow("image", img)
    
    
    cv2.waitKey(0)
    
    return region

def select_multiple_roi(img):
    """
    DOES NOT WORK

    """
    regions = cv2.selectROIs("Select regions", img, False, False)
    for r in regions:
        cv2.rectangle(img, (r[0], r[1]), (r[2]+r[0], r[3]+r[1]), (255, 0, 0), 2)
    cv2.imshow("image", img)
    cv2.waitKey(0)


#Args:
fps = 0
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

while True:
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    ret, image_np = vid.read()
    # image_np = cv2.flip(image_np, 1)
    try:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")
    
    region = select_roi(image_np)
    print(region)
    break

    # Calculate Frames per second (FPS)

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
        print("frames processed: ")
vid.release()
cv2.destroyAllWindows()