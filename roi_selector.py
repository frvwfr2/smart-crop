# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import json
import datetime
import imutils
import time
import datetime
import cv2
from numpy.core.fromnumeric import size

# ROI means "Region of Interest"

def parse_roi_file(roi_filepath):
    with open(roi_filepath) as f:
        data = json.load(f)
        roi_include = data["include"]
        roi_exclude = data["exclude"]
        roi_cones = data["conelist"]
    return roi_include, roi_exclude, roi_cones

def save_coord(event, x, y, flags, param):
    global whitelist, blacklist, conelist, mode_flag
    # print(mode_flag)

    # First we are getting Cones
    if mode_flag == "cones" and event == cv2.EVENT_LBUTTONDOWN:
        conelist.append( (x, y) )
        cv2.rectangle(f, (x-4, y-4), (x+4, y+4), (0, 165, 255), -1)
    elif mode_flag == "cones" and event == cv2.EVENT_MBUTTONDOWN:
        mode_flag = "inclusion"
        # Refill in the rectangle, so we can put fresh text on top
        cv2.rectangle(f, (0, 0), (1500, 50), (0,0,0), -1)
        cv2.putText(f, "Left-click each corner of the inclusion zone. Look out for tall people and shadows", (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
    # If l_click pressed
    elif mode_flag == "inclusion" and event == cv2.EVENT_LBUTTONDOWN:
        # Append a tuple with the coordinates
        whitelist.append((x,y))
        # Probably want to draw on the image 
        if len(whitelist) > 1:
            cv2.line(f, whitelist[-1], whitelist[-2], (0,0,255), 2)
        # print(x, y)
    elif mode_flag == "inclusion" and event == cv2.EVENT_MBUTTONDOWN:
        mode_flag = "exclusion"
        # Refill in the rectangle, so we can put fresh text on top
        cv2.rectangle(f, (0, 0), (1500, 50), (0,0,0), -1)
        cv2.putText(f, "Left-click each corner of exclusion zones. Middle-click to 'complete' a zone. Press Q when finished.", (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
    # If r_click pressed
    elif mode_flag == "exclusion" and event == cv2.EVENT_LBUTTONDOWN:
        # Append our coordinates to the most recently created list entry
        # We do this so that we can create MULTIPLE "deadzones"
        blacklist[-1].append((x,y))
        if len(blacklist[-1]) > 1:
            cv2.line(f, blacklist[-1][-1], blacklist[-1][-2], (0,0,0), 1)
    # If middle button pressed, we want to "finish" the current exclude-zone and start a new one
    elif mode_flag == "exclusion" and event == cv2.EVENT_MBUTTONDOWN:
        # Draw a line from our current exclude-zone's most recent value, to the first value, to finish off the box.
        cv2.line(f, blacklist[-1][-1], blacklist[-1][0], (0,0,0), 1)
        # Fill in the box we just drew
        cv2.fillPoly(f, np.array([ blacklist[-1] ]), (0,0,0))
        # Make a new list as the starting point
        blacklist.append(list())
        pass

def create_roi_from_video(video):
    vs = cv2.VideoCapture(video)
    frame = vs.read()
    frame = frame[1]
    # print(frame)
    # We need to get the frame from the video
    whitelist, blacklist, conelist = roi_selection(frame)
    values = dict()
    values["include"] = whitelist
    values["exclude"] = blacklist
    values["conelist"] = conelist
    print(f"Cones list: {conelist}")
    print(f"ROI coordinates: {whitelist}")
    print(f"Exclude list: {blacklist}")
    filename = f"{video}_roi.json"
    with open(filename, 'w') as f:
        json.dump(values, f)
    return filename
        # f.writelines(f"INCLUDE\t{whitelist}\n")
        # f.writelines(f"EXCLUDE\t{blacklist}")

# We need to receive the frame to use to draw on. Do we need anything else? Should this piece convert to 4k or just return the variables as-is? Probably as-is
def roi_selection(frame):
    # Should we just go in order - Left-click cones. Left-click ROI. Left-click exclusion zones.
    global whitelist, f, blacklist, mode_flag, conelist
    mode_flag = "inclusion"
    whitelist = list()
    blacklist = list()
    conelist = list()
    blacklist.append(list())
    f = frame
    cv2.namedWindow("ROI Selector")
    cv2.setMouseCallback("ROI Selector", save_coord)
    # Draw a rectangle for us to put instruction text on
    f = imutils.resize(f, width=1920)
    # cv2.rectangle(f, (0, 0), (1500, 50), (0,0,0), -1)
    # cv2.putText(f, "Left-click each sideline, starting from the same endzone. Middle-click when complete.", (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
    cv2.rectangle(f, (0, 0), (1500, 50), (0,0,0), -1)
    cv2.putText(f, "Left-click each corner of the inclusion zone. Look out for tall people and shadows", (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
    while True:
        f = imutils.resize(f, width=1920)
        w = cv2.imshow("ROI Selector", f)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            cv2.destroyWindow(w)
            break
            # Mark that there is a cone at this location
    return whitelist, blacklist, conelist
        # pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file", required=True)
    ap.add_argument("--roi_file", help="File path to existing ROI file to add to. Not implemented.")
    args = vars(ap.parse_args())

    filename = create_roi_from_video(args["video"])
    print(f"ROI file written to {filename}")