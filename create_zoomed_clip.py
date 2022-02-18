# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import datetime
import imutils
import time
import datetime
import cv2
import os
import subprocess
import sys
import json
from numpy.core.fromnumeric import size
from numpy.lib.function_base import append
from roi_selector import create_roi_from_video, roi_selection, parse_roi_file
from parse_teams import Team
import logging



# # This takes in a "target center" and outputs the "actual center"
# def find_center_within_bounds(center_x, center_y, target_width, target_height, max_width, max_height):
#     # Ensure we aren't off the screen to the left
#     center_x *= ANALYZE_SHRINK_FACTOR
#     print()
#     print(center_x, center_y, target_width, target_height, max_width, max_height)
#     cropped_x1 = max(center_x - target_width // 2, 0)
#     # Ensure we aren't off the screen to the right
#     cropped_x1 = min((cropped_x1, max_width - target_width))
#     # cropped_x2 = min((self.ZOOM_FACTOR * ANALYZE_SHRINK_FACTOR * new_center[0]) + self.OUTPUT_SIZE[0] // 2, 3840)
#     cropped_x2 = cropped_x1 + target_width

#     # Ensure we aren't off the screen to the top
#     cropped_y1 = max(center_y - target_height // 2, 0)
#     # Ensure we aren't off the screen to the bottom
#     cropped_y1 = min((cropped_y1, max_height - target_height))
#     cropped_y2 = cropped_y1 + target_height
#     return (cropped_x1, cropped_y1), (cropped_x2, cropped_y2)

def get_new_momentum(old_velocity, new_distance):
    acceleration = 1
    # print(old_velocity, new_distance)
    if new_distance > old_velocity:
        # print(old_velocity + acceleration)
        return old_velocity - acceleration
    elif new_distance < old_velocity:
        return old_velocity + acceleration
    else:
        return max(old_velocity - acceleration, 0)

def find_dilated_canny_contours(canny, input_w, input_h, args):
    thresh = cv2.dilate(canny, None, iterations=1)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    min_x, max_x = input_w, 0
    min_y, max_y = input_h, 0
    for c in cnts:
        # print(cv2.contourArea(c))
        # If the area we found was too small, ignore it
        # thresh = cv2.bitwise_not(thresh, c)
        if cv2.contourArea(c) < args["min_area"]:
            continue
            # print("FOUND AREA TOO SMALL")
        (x,y,w,h) = cv2.boundingRect(c)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
    if min_x == input_w and max_x == 0 and min_y == input_h and max_y == 0:
        return None, None, None, None, thresh
    return min_x, max_x, min_y, max_y, thresh

def draw_debug_text(frame):
    text_y = 15
    frame = cv2.putText(frame, 'Edge Detection bounds in Green', (2, text_y), cv2.FONT_HERSHEY_DUPLEX, .5, (0,0,0), 2)
    frame = cv2.putText(frame, 'Edge Detection bounds in Green', (2, text_y), cv2.FONT_HERSHEY_DUPLEX, .5, (0,255,0), 1)
    text_y += 20
    frame = cv2.putText(frame, 'Green dot is "Center of target"', (2, text_y), cv2.FONT_HERSHEY_DUPLEX, .5, (0,0,0), 2)
    frame = cv2.putText(frame, 'Green dot is "Center of target"', (2, text_y), cv2.FONT_HERSHEY_DUPLEX, .5, (0,255,0), 1)
    text_y += 20
    frame = cv2.putText(frame, 'White dot is "Current center"', (2, text_y), cv2.FONT_HERSHEY_DUPLEX, .5, (0,0,0), 2)
    frame = cv2.putText(frame, 'White dot is "Current center"', (2, text_y), cv2.FONT_HERSHEY_DUPLEX, .5, (255,255,255), 1)
    text_y += 20
    frame = cv2.putText(frame, 'If green dot is within "Movement Deadzone", don\'t move. ', (2, text_y), cv2.FONT_HERSHEY_DUPLEX, .5, (0,0,0), 2)
    frame = cv2.putText(frame, 'If green dot is within "Movement Deadzone", don\'t move. ', (2, text_y), cv2.FONT_HERSHEY_DUPLEX, .5, (255,255,255), 1)
    text_y += 20
    frame = cv2.putText(frame, 'If green box is within "Outer buffer box", don\'t move.', (2, text_y), cv2.FONT_HERSHEY_DUPLEX, .5, (0,0,0), 2)
    frame = cv2.putText(frame, 'If green box is within "Outer buffer box", don\'t move.', (2, text_y), cv2.FONT_HERSHEY_DUPLEX, .5, (255,255,255), 1)
    text_y += 20
    frame = cv2.putText(frame, 'Else, shift the cropped frame from the white dot towards the green dot.', (2, text_y), cv2.FONT_HERSHEY_DUPLEX, .5, (0,0,0), 2)
    frame = cv2.putText(frame, 'Else, shift the cropped frame from the white dot towards the green dot.', (2, text_y), cv2.FONT_HERSHEY_DUPLEX, .5, (255,255,255), 1)

    return frame


# def main(args):

class ZoomedClipCreator:

    # Returns the mask and the ROI corners
    def generate_roi_mask(self, roi_filepath, frame):
        frame = frame.copy()
        canny_mask = np.zeros(frame.shape, dtype=np.uint8)

        roi_include, roi_exclude, roi_cones = parse_roi_file(roi_filepath=roi_filepath)
        # We need to generate the INCLUDE mask first, and then exclude onto it.

        roi_corners = np.array([ roi_include ])

        roi_corners = roi_corners.astype(int)

        # fill the ROI so it doesn't get wiped out when the mask is applied
        channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your image
        # we are making a mask of all 1's, and then doing an AND of base_image and mask to limit our selection to ONLY the mask
        # We will then need to do a mask of all 0's

        # Draw the include box of all 1's
        include_mask_color = (255,)*channel_count
        # 
        cv2.fillPoly(canny_mask, roi_corners, include_mask_color)

        # Draw the exclude box of all 0's
        for exclude_box in roi_exclude:
            if not exclude_box:
                continue
            exclude_corners = np.array([ exclude_box ])
            exclude_mask_color = (0,)*channel_count
            cv2.fillPoly(canny_mask, exclude_corners, exclude_mask_color)
        # from Masterfool: use cv2.fillConvexPoly if you know it's convex
        logging.debug(f"Applying mask to frame")
        # apply the mask
        # canny_mask = cv2.bitwise_and(frame, canny_mask)
        return canny_mask, roi_corners

    def find_new_center_from_old(self, canny_coords):
        # This is the locations of the edges we detected
        x_min, y_max, x_max, y_min = canny_coords

        # If we are not currently trying to get centered... check if we need to move that direction
        if not self.find_inner_deadzone:
            if abs(self.old_center[0] - self.new_center[0]) < self.DEADZONE_X:
                self.new_center[0] = self.old_center[0]
            else:
                self.find_inner_deadzone = True
                # print("DEADZONE FLAG SET TO TRUE")
            # compare old Y to new Y
            if abs(self.old_center[1] - self.new_center[1]) < self.DEADZONE_Y:
                self.new_center[1] = self.old_center[1]
        # This is NOT an else statement so that if we trigger it above, then we immediately begin to shift to fix it
        if self.find_inner_deadzone:
            # print("DEADZONE FLAG CURRENTLY TRUE")
            if abs(self.old_center[0] - self.new_center[0]) < self.INNER_DEADZONE_X:
                self.new_center[0] = self.old_center[0]
                self.find_inner_deadzone = False
                # print("DEADZONE FLAG SET TO FALSE")
            # compare old Y to new Y
            if abs(self.old_center[1] - self.new_center[1]) < self.DEADZONE_Y:
                self.new_center[1] = self.old_center[1]
        # If all of our edges are well-within the buffers, don't move the camera at all. Here we generate the buffers based on where the old_center was.
        # If new_center is still inside this, don't shift

        # 2) Are we well inside the frame? If yes, stay there
        self.left_buffer = (self.old_center[0] - self.INPUT_SIZE[0]//self.ANALYZE_SHRINK_FACTOR//self.ZOOM_FACTOR2 + self.OUTER_BOUND_BUFFER_X)
        self.right_buffer = (self.old_center[0] + self.INPUT_SIZE[0]//self.ANALYZE_SHRINK_FACTOR//self.ZOOM_FACTOR2 - self.OUTER_BOUND_BUFFER_X)
        self.top_buffer = (self.old_center[1] - self.INPUT_SIZE[1]//self.ANALYZE_SHRINK_FACTOR//self.ZOOM_FACTOR2 + self.OUTER_BOUND_BUFFER_Y)
        self.bottom_buffer = (self.old_center[1] + self.INPUT_SIZE[1]//self.ANALYZE_SHRINK_FACTOR//self.ZOOM_FACTOR2 - self.OUTER_BOUND_BUFFER_Y)
        # print(new_center, self.OUTPUT_SIZE, OUTER_BOUND_BUFFER_X, OUTER_BOUND_BUFFER_Y)
        # print(left_buffer, right_buffer, top_buffer, bottom_buffer)
        # Check if we are already staying in place
        # Currently DISABLED for testing.
        if True:
            if self.new_center != self.old_center:
                # If x_values are between the buffers, then don't move
                if x_min > self.left_buffer and x_max < self.right_buffer:
                    # print("X between buffers, not moving")
                    self.new_center[0] = self.old_center[0]
        # To calculate Zoom, we want to get x_max - x_min. If that value is larger than our Crop target, then we need to zoom out some.

        # cropped = cv2.putText(cropped, self.FILENAME, (5, self.OUTPUT_SIZE[1]//40), cv2.FONT_HERSHEY_DUPLEX, .5, (0, 0, 0), 3)
        # cropped = cv2.putText(cropped, self.FILENAME, (5, self.OUTPUT_SIZE[1]//40), cv2.FONT_HERSHEY_DUPLEX, .5, (255, 255, 255), 1)
        
        # Limit movement to MAX_PIXEL_MOVEMENT_X/Y pixels

        # 3) Are we moving? Okay, only move _this_ quickly
        # compare old X to new X

        # print(old_center, new_center)
        # velocity = get_new_momentum(velocity, old_center[0] - new_center[0])
        # if velocity > MAX_PIXEL_MOVEMENT_X:
        #     velocity = MAX_PIXEL_MOVEMENT_X
        # print(velocity)

        # compare old X to new X
        # If we are moving further than our max_allowed_movement...
        if abs(self.old_center[0] - self.new_center[0]) > self.MAX_PIXEL_MOVEMENT_X:
            # If we are moving to the left, set the new center LEFT by velocity
            if self.old_center[0] > self.new_center[0]:
                self.new_center[0] = self.old_center[0] - self.MAX_PIXEL_MOVEMENT_X
            # Otherwise, set the new center RIGHT by velocity
            else:
                self.new_center[0] = self.old_center[0] + self.MAX_PIXEL_MOVEMENT_X

        # compare old Y to new Y
        # If we are moving further than our max_allowed_movement...
        if abs(self.old_center[1] - self.new_center[1]) > self.MAX_PIXEL_MOVEMENT_Y:
            # If we're moving DOWN, adjust our movement upwards
            if self.old_center[1] > self.new_center[1]:
                self.new_center[1] = self.old_center[1] - self.MAX_PIXEL_MOVEMENT_Y
            # Otherwise, we must be moving UP. So adjust ourselves upwards
            else:
                self.new_center[1] = self.old_center[1] + self.MAX_PIXEL_MOVEMENT_Y

        smooth_top_x = 1
        # smooth_top_x = max(new_center[0] - (FORCED_WIDTH // self.ZOOM_FACTOR2), 0)
        # smooth_top_y = max(new_center[1] - (FORCED_HEIGHT // self.ZOOM_FACTOR2), 0)
        # print("NEW CENTER", new_center)

        # If the bounds are still within our frame, don't move at all
        if self.PREVENT_PAN_WHILE_BOUNDED:
            if x_max < smooth_top_x + self.input_w//self.ZOOM_FACTOR and x_min > smooth_top_x:
                # print("STILL WITHIN THE PREVIOUS BOX X")
                self.new_center[0] = self.old_center[0]
            if y_max < self.smooth_top_y + self.input_h//self.ZOOM_FACTOR and y_min > self.smooth_top_y:
                # print("STILL WITHIN THE PREVIOUS BOX Y")
                self.new_center[1] = self.old_center[1]
            if x_max < self.top_x + self.input_w//self.ZOOM_FACTOR and x_min > self.top_x:
                # print("STILL WITHIN THE PREVIOUS BOX X")
                self.new_center[0] = self.old_center[0]
            if y_max < self.top_y + self.input_h//self.ZOOM_FACTOR and y_min > self.top_y:
                # print("STILL WITHIN THE PREVIOUS BOX Y")
                self.new_center[1] = self.old_center[1]
            # If all the bounds are still inside our box, don't move at all

        # TODO Add code for "If something is out on BOTH sides, we need to zoom out"

    # This will persist for the entirety of this clip. The Init() should just prep the clip
    # Another function should do the actual step through of each frame. Each frame should be processed in another function.
    def __init__(self, args):
        # Things in self. should only be ones that carry across frames
        self.DEBUG = args["debug"]
        self.QUIET = True
        if self.DEBUG:
            self.SHOW_DEBUG = True
        else:
            self.SHOW_DEBUG = args["show_debug"]

        if self.DEBUG:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

        logging.info("Beginning clip")

        # if the video argument is None, then we are reading from webcam
        if args.get("video", None) is None:
            raise NotImplemented
            self.vs = VideoStream(src=0).start()
            time.sleep(2.0)
        # otherwise, we are reading from a video file
        else:
            self.vs = cv2.VideoCapture(args["video"])
        # initialize the first frame in the video stream
        self.firstFrame = None
        self.next = None

        self.team_one = None
        self.team_one_score = None
        self.team_two = None
        self.team_two_score = None

        if args.get("team_one_file", None) is not None:
            self.team_one = Team(args.get("team_one_file"))
        if args.get("team_one_score", None) is not None:
            self.team_one_score = args.get("team_one_score", None)
        if args.get("team_two_file", None) is not None:
            self.team_two = Team(args.get("team_two_file"))
        if args.get("team_two_score", None) is not None:
            self.team_two_score = args.get("team_two_score", None)
        # Initialize vars
        self.count = 0
        self.total_count = 0

        self.center_x = None
        self.center_y = None
        self.top_x = None
        self.top_y = None

        self.smooth_top_x = None
        self.smooth_top_y = None

        self.new_center = None
        self.old_center = None

        # We don't care about these in the Y-direction, not enough movement to care.
        # With these added, we probably want to bump our default max_movement up a little bit from 3.
        self.velocity = 0
        self.acceleration = 0
        self.jerk = 0

        # 1 means the most we change per frame is 1 ? is that sufficient?
        self.max_acceleration = 1
        self.max_jerk = 1

        # Maximum movement velocity of the cropped zone per frame.
        self.MAX_PIXEL_MOVEMENT_X = args["max_camera_movement_x"]
        # MAX_PIXEL_MOVEMENT_X = 10
        self.MAX_PIXEL_MOVEMENT_Y = args["max_camera_movement_y"]

        # Transparency level
        self.TRANSPARENCY = .3

        # Deadzone where the camera doesn't move at all, as long as the center is still within that many pixels.
        self.DEADZONE_X = args["deadzone_x"]
        self.DEADZONE_Y = args["deadzone_y"]
        self.INNER_DEADZONE_X = self.DEADZONE_X // 2
        self.INNER_DEADZONE_Y = self.DEADZONE_Y // 2
        self.OUTER_BOUND_BUFFER_X = args["outer_bound_buffer_x"]
        self.OUTER_BOUND_BUFFER_Y = args["outer_bound_buffer_y"]

        # This value defines if the crop should attempt to keep the video centered on the boxes, or simply stop moving if all the boxes are still inside the crop.
        self.PREVENT_PAN_WHILE_BOUNDED = False
        # Flag to disable writing video
        self.WRITE_VIDEO = args["write"]
        # Not implemented. How much time would it save us if we only re-calculated the "new-center" every 5 frames, rather than every single frame?
        self.FRAMES_BETWEEN_RE_COMPARE = 1
        self.MAX_PIXEL_MOVEMENT_X *= self.FRAMES_BETWEEN_RE_COMPARE
        self.MAX_PIXEL_MOVEMENT_Y *= self.FRAMES_BETWEEN_RE_COMPARE
        # This flag defines if we are still in the "slow-zoom" phase of a clip.
        # Does this need to be a counter? Let's do that for now, for flexibility
        self.ZOOM_IN_DURATION_FRAMES = args["zoomin_duration_frames"]

        self.FILENAME = os.path.basename(os.path.splitext(args["video"])[0])
        logging.info(f"File: {self.FILENAME}")

        self.input_w = int(self.vs.get(cv2.CAP_PROP_FRAME_WIDTH ))
        self.input_h = int(self.vs.get(cv2.CAP_PROP_FRAME_HEIGHT ))
        self.frames_count = int(self.vs.get(cv2.CAP_PROP_FRAME_COUNT))
        # This value is what the video CLAIMS it is. It is not necessarily the truth.
        self.fps = self.vs.get(cv2.CAP_PROP_FPS)
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", 
                                args["video"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        self.duration = "DISABLED"
        try:
            self.duration = float(result.stdout.splitlines()[-1])
            # This causes major issues with concating videos. Need to figure out how to merge video and audio
            self.fps = self.frames_count / self.duration
        except ValueError:
            print("Assigned default fps of 30")
            self.fps = 30

        logging.info(f"Frames: {self.frames_count} FPS: {self.fps} Duration: {self.duration}s")
        logging.info(f"Inputs: {self.input_w}x{self.input_h} @ {self.fps} fps")

        self.mask = None



        # How much to shrink the raw input for analyzing contours
        self.ANALYZE_SHRINK_FACTOR = 2

        self.INPUT_SIZE = (self.input_w, self.input_h)
        self.OUTPUT_SIZE = (1920, 1080)

        self.ZOOM_FACTOR = self.input_w // self.OUTPUT_SIZE[0]
        # The self.Zoom_factor squared
        self.ZOOM_FACTOR2 = self.ZOOM_FACTOR*self.ZOOM_FACTOR

        # FORCED_WIDTH = 1920
        # FORCED_HEIGHT = 1080

        # TODO
        # Have the zoom-pieces use the shrunken frame (raw_input // 4), but the output video use raw_input // 2 for the zoom factor. Make these factors codeable?
        if self.WRITE_VIDEO:
            # If the file already exists, and we are NOT overwriting existing
            if os.path.exists(args["output_filename"]) and not args["overwrite_existing"]:
                print("File already exists, and command line arg '--overwrite_existing' is not specified.")
                # raise FileExistsError
                return 1
            if args.get("output_filename", None):
                self.cropped_recording = cv2.VideoWriter(args["output_filename"], cv2.VideoWriter_fourcc(*'MP4V'), self.fps, (self.OUTPUT_SIZE[0], self.OUTPUT_SIZE[1]))
            else:
                self.cropped_recording = cv2.VideoWriter(args["video"] + "_canny.mp4", cv2.VideoWriter_fourcc(*'MP4V'), self.fps, (self.OUTPUT_SIZE[0], self.OUTPUT_SIZE[1]))
        if self.DEBUG:
            self.live_recording = cv2.VideoWriter(args["video"] + "_debug.mp4", cv2.VideoWriter_fourcc(*'MP4V'), self.fps, (self.input_w // self.ANALYZE_SHRINK_FACTOR, self.input_h // self.ANALYZE_SHRINK_FACTOR))

        # print(f"Beginning looping @ {datetime.datetime.now()}")
        logging.info("Beginning looping")

        # This should be set to true if we are moving while outside of the deadzone. 
        # If the flag is true, we want to continue to move towards the center, until we are within INNER_DEADZONE distance. Then the flag can be reset to False
        self.find_inner_deadzone = False

        self.process_video()

    def process_video(self):
        self.process_start_time = datetime.datetime.now()
        # loop over the frames of the video
        while True:
            # print("LINE")
            logging.debug(f"Starting New Frame")
                # print(f"\r{total_count:08d} frames / {int(total_count // fps):04d} seconds", end='')
            args["skip_starting_frames"] = 0
            # grab the current frame and initialize the occupied/unoccupied
            self.frame = self.vs.read()
            self.frame = self.frame if args.get("video", None) is None else self.frame[1]
            logging.debug(f"Making copies and resizing")
            # if the frame could not be grabbed, then we have reached the end
            # of the video
            if self.frame is None:
                break

            
            # Copy the frame to be used by the cropped segment, to prevent losing any detail
            self.cropped = self.frame.copy()
            
            # resize the frame, convert it to grayscale, and blur it
            # 960 is 1/4 of 4k
            # frame = imutils.resize(frame, width=960)
            # 1920 is 1/2 of 4k
            # Halves the size of the analyzed_frames
            # print(FORCED_WIDTH, ANALYZE_SHRINK_FACTOR)

            logging.debug(f"Resizing frame")
            self.frame = imutils.resize(self.frame, width=self.input_w // self.ANALYZE_SHRINK_FACTOR)
            
            logging.debug(f"Done resizing frame")







            # Create the mask, this should only be done once
            # TODO this should be a separate function
            if self.mask is None:
                # self.mask = self.create_roi_mask(roi_filepath)
                logging.debug("Creating mask")
                # If we were passed an ROI filepath, use that
                if args.get("roi_filepath", None):
                    self.mask, self.roi_corners = self.generate_roi_mask(args["roi_filepath"], self.frame)
                # Otherwise, we need to create an ROI file 
                else:
                    roi_file = create_roi_from_video(args["video"])
                    self.mask, self.roi_corners = self.generate_roi_mask(roi_file, self.frame)
                bw_mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
                # Create a different mask to avoid breaking the one that is re-used on each frame
                m2 = np.array(self.mask, dtype=np.float) # .017 secs
                # Convert to BGR values
                m2 /= 255.0 # .008 secs
                # Divide by transparency
                m2 *= self.TRANSPARENCY # .008 secs
                # Create our pure green mask
                self.green = np.ones(self.frame.shape, dtype=np.float)*(0,1,0) # .04 secs
                logging.debug("Done creating mask")



            logging.debug("Running Canny detection")
            self.canny = self.frame.copy()
            self.canny = cv2.Canny(self.canny, 240, 250)
            # print(frame)
            # print(canny.shape)
            # print(mask.shape)
            self.canny = cv2.bitwise_and(self.canny, bw_mask)
            logging.debug("Find Dilated Canny contours")
            # Dilate the image to fill in gaps, then find contours
            # if self.total_count % self.FRAMES_BETWEEN_RE_COMPARE != 0:
            canny_min_x, canny_max_x, canny_min_y, canny_max_y, thresh = find_dilated_canny_contours(self.canny, self.input_w, self.input_h, args)

            # Draw a line from roi_cones[0] to roi_cones[2]
            # Draws a line on each sideline.
            # cv2.line(frame, roi_cones[0], roi_cones[1], (255,255,255), 1)
            # cv2.line(frame, roi_cones[2], roi_cones[3], (255,255,255), 1)
            # cv2.line(frame, roi_cones[0], roi_cones[2], (255,255,255), 1)
            # cv2.line(frame, roi_cones[1], roi_cones[3], (255,255,255), 1)
            # cv2.line(frame, )
            # Draw the intermediate lines too
            # Fisheye effect kills this :(
            # for i in range(0, 7+1):
            #     cone_top_x = int(roi_cones[0][0] + (roi_cones[1][0] - roi_cones[0][0]) * i/7)
            #     cone_top_y = int(roi_cones[0][1] + (roi_cones[1][1] - roi_cones[0][1]) * i/7)
            #     cone_bot_x = int(roi_cones[2][0] + (roi_cones[3][0] - roi_cones[2][0]) * i/7)
            #     cone_bot_y = int(roi_cones[2][1] + (roi_cones[3][1] - roi_cones[2][1]) * i/7)
            #     print(cone_top_x, cone_top_y, cone_bot_x, cone_bot_y)
            #     cv2.line(frame,  (cone_top_x, cone_top_y), (cone_bot_x, cone_bot_y), (255,255,255), 2)

            # sys.exit()
            # preview = frame.copy()

            # canny_coords = np.where(canny != [0])
            # # print(type(canny_coords))
            # canny_min_x = np.amin(canny_coords[1])
            # canny_max_x = np.amax(canny_coords[1])
            # canny_min_y = np.amin(canny_coords[0])
            # canny_max_y = np.amax(canny_coords[0])

            # # Reassign these values to use the "blurred" versions from above
            # canny_min_x = min_x
            # canny_max_x = max_x
            # canny_min_y = min_y
            # canny_max_y = max_y

            # print(f"({canny_min_x}, {canny_min_y}) / ({canny_max_x}, {canny_max_y})")
            # frame = cv2.putText(frame, 'Edge Detection bounds in Green', (canny_min_x, canny_min_y-5), cv2.FONT_HERSHEY_DUPLEX, .5, (255,255,255), 2)
            # frame = cv2.putText(frame, 'Edge Detection bounds in Green', (canny_min_x, canny_min_y-5), cv2.FONT_HERSHEY_DUPLEX, .5, (0,255,0), 1)
            if self.DEBUG:
                logging.debug(f"Draw debug text")
                self.frame = draw_debug_text(self.frame)
            # If green dot is within "Movement Deadzone", don't move. 
            # If green box is within "Outer buffer box", don't move.
            # Else, shift the cropped frame from the white dot towards the green dot.

            # if the first frame is None, initialize it. Also, every 50 frames, reset the comparison image
            # We need to do a tick-tock - instead of starting fresh each time, use a frame from 1s prior

            # Reset the min/max values IF we detected a new location
            if canny_min_x is not None:
                x_min = None
                y_min = None
                x_max = None
                y_max = None

            logging.debug(f"Starting to locate camera-center processing")
            # If we didn't find any Canny Edges, OR we are skipping frames between recompares, re-use values from prior
            if not canny_min_x or self.total_count % self.FRAMES_BETWEEN_RE_COMPARE != 0:
                # print("REUSING OLD CENTER")
                # print("NO Canny Edges found")
                # If we had previous values to use, then produce an object from those
                if self.DEBUG and center_x and center_y:
                    # Draw a green center-dot where the center of the edges are. This matches the Green Box that defines this point
                    cv2.rectangle(self.frame, (center_x-1, center_y-1), (center_x+1, center_y+1), (0, 255, 0), 2)
                    # Draw a blue box showing the cropped section
                    cv2.rectangle(self.frame, (top_x, top_y), (top_x + self.input_w//self.ZOOM_FACTOR, top_y + self.input_h//self.ZOOM_FACTOR), (255, 0, 0), 2)
                    self.frame = cv2.putText(self.frame, 'Cropped frame', (top_x, top_y-5), cv2.FONT_HERSHEY_DUPLEX, .5, (0,0,0), 2)
                    self.frame = cv2.putText(self.frame, 'Cropped frame', (top_x, top_y-5), cv2.FONT_HERSHEY_DUPLEX, .5, (255,0,0), 1)
                # If we did not have previous values to use, write out the full frame. This should only ever occur on the very first frame(s).
                if not (center_x and center_y):
                    logging.warning("WRITING BAD OUTPUT")
                    # raise ValueError("No canny edges detected in frame, and no center yet produced")
                    # Write the current image to the video file
                    if self.WRITE_VIDEO:
                        self.frame = cv2.resize(self.frame, (self.OUTPUT_SIZE[0], self.OUTPUT_SIZE[1]))
                        # Write the clip name on the frame, before sending to file
                        self.frame = cv2.putText(self.frame, self.FILENAME, (5, self.OUTPUT_SIZE[1]//40), cv2.FONT_HERSHEY_DUPLEX, .5, (0, 0, 0), 3)
                        self.frame = cv2.putText(self.frame, self.FILENAME, (5, self.OUTPUT_SIZE[1]//40), cv2.FONT_HERSHEY_DUPLEX, .5, (255, 255, 255), 1)
                        self.cropped_recording.write(self.frame)
                    continue
            else:
                # Get the corners of the contours that were located. x+w, y+h
                x_min = canny_min_x
                x_max = canny_max_x
                y_min = canny_min_y
                y_max = canny_max_y
                # print(x_min, x_max, y_min, y_max)
                # Calculate the center and top_corner values for drawing boxes
                center_x = (x_max + x_min)//2
                top_x = max(center_x - (self.input_w // self.ZOOM_FACTOR), 0)
                center_y = (y_max + y_min)//2
                top_y = max(center_y - (self.input_h // self.ZOOM_FACTOR), 0)
                self.new_center = [center_x, center_y]
                # This only triggers on first frame
                if not self.old_center:
                    self.old_center = self.new_center

                # compare old X to new X
                # If New is close to Old, don't move at all
                # IF WE ARE OUTSIDE OF THE BIG DEADZONE... MOVE TO GET INSIDE THE SMALL DEADZONE
                # find_inner_deadzone is set to True if we are seeking the inner_deadzone
                # this could (should?) be a function
                # Inputs: find_inner_deadzone flag, old_center, new_center
                # Outputs: find_inner_deadzone flag, new_center

                # 1) Are we near the center? If yes, stay there




                canny_coords = (x_min, y_max, x_max, y_min)
                # TODO this should all be 1 function 


                self.find_new_center_from_old(canny_coords=canny_coords)

                # TODO until here, should be 1 function. Take in Old_Center, calculate New_Center based on internal parameters





            # Once we have determined a target point, begin to show the preview and write to a file
            if True: # smooth_top_x is not None:

                # Draw a smoothed-pan center point
                # THIS VALUE IS CORRECT

                # print(center_x, center_y, top_x, top_y)
                # Draw a 1920x1080 box with the center as the center of the contours, with edge-safety
                # cv2.rectangle(frame, (top_x, top_y), (top_x + FORCED_WIDTH//(self.ZOOM_FACTOR * ANALYZE_SHRINK_FACTOR), top_y + FORCED_HEIGHT//(self.ZOOM_FACTOR * ANALYZE_SHRINK_FACTOR)), (255, 0, 0), 2)

                # Cropped stuff goes very-very last. Everything prior to this should use local-scaling
                
                ######
                # new_center is ACCURATE, for the preview image.
                ######

                # This code is an attempt to "zoom out" to capture all objects - but it doesn't work well (...or at all)
                crop_width = 2*(x_max - x_min)
                crop_height = 2*(y_max - y_min)
                if True: # or crop_width < self.OUTPUT_SIZE[0]:
                    crop_width = self.OUTPUT_SIZE[0]
                    crop_height = self.OUTPUT_SIZE[1]
                else:
                    crop_height = crop_width * 9 // 16

                # 4) Adjust based on the offset_x and offset_y arguments
                self.new_center[0] += args["offset_x"]
                self.new_center[1] += args["offset_y"]
                # self.OUTPUT_SIZE 0 and 1 should be replaced with the new value of "zoom_target width and height"
                # tl, br = find_center_within_bounds(new_center[0], new_center[1], max(x_max - x_min, self.OUTPUT_SIZE[0]), max(y_max - y_min, self.OUTPUT_SIZE[1]), INPUT_SIZE[0], INPUT_SIZE[1])
                # 5) Ensure we aren't off the screen to the left
                cropped_x1 = max((self.ANALYZE_SHRINK_FACTOR * self.new_center[0]) - crop_width // 2, 0)
                # Ensure we aren't off the screen to the right
                cropped_x1 = min((cropped_x1, self.input_w - crop_width))
                # cropped_x2 = min((self.ZOOM_FACTOR * ANALYZE_SHRINK_FACTOR * new_center[0]) + self.OUTPUT_SIZE[0] // 2, 3840)
                cropped_x2 = cropped_x1 + crop_width

                # Ensure we aren't off the screen to the top
                cropped_y1 = max((self.ANALYZE_SHRINK_FACTOR * self.new_center[1]) - crop_height // 2, 0)
                # Ensure we aren't off the screen to the bottom
                cropped_y1 = min((cropped_y1, self.input_h - crop_height))
                cropped_y2 = cropped_y1 + crop_height
                
                # print(cropped_x1, cropped_y1, cropped_x2, cropped_y2)
                # cropped_y2 = min((self.ZOOM_FACTOR * ANALYZE_SHRINK_FACTOR * new_center[1]) + self.OUTPUT_SIZE[1] // 2, 2160)
                # cropped_x = int(smooth_top_x * self.ZOOM_FACTOR * ANALYZE_SHRINK_FACTOR)
                # cropped_y = int(smooth_top_y * self.ZOOM_FACTOR * ANALYZE_SHRINK_FACTOR)
                # print(smooth_top_x, smooth_top_y, self.ZOOM_FACTOR, ANALYZE_SHRINK_FACTOR)


                # Basically, we define a zoom_time. This will give time for the pull to occur etc.
                # While the frame_count is less than ... 300 frames? (10 seconds)
                # We want to do a progressive zoom. We will need to specify the size of the box to be cropped (last line before writing to the video)
                # This should use a similar formula to now, just the values need to be modified IF zoomed_frame_count < 300
                # cropped_x1 needs to halve the distance from 0 -> cropped_x1 ->>> Formula could be... 
                # Calc the distance from the boundary. distance = abs(cropped_XY - {0 OR INPUT_WIDTH[01]})
                # current_bound_x1 = (total_count / self.ZOOM_IN_DURATION_FRAMES) * distance_x1. We are shrinking $distance by the percent of self.ZOOM_IN_DURATION_FRAMES we have
                # current_bound_x2 = INPUT_WIDTH - ((total_count / self.ZOOM_IN_DURATION_FRAMES) * distance_x2)
                # cropped_x1 = cropped_x1 - current_bound_x1
                # cropped_y1 needs to halve the distance from 0 -> cropped_y1
                # cropped_x2 needs to be larger, towards the 4k bound (3840?) INPUT_WIDTH[0]
                # cropped_y2 needs to be larger, towards the 4k bound (2160?) INPUT_WIDTH[1]
                # And then resize it to self.OUTPUT_SIZE[0]
                # A linear scaling. We might want to make this do a quicker zoom initially, and then slow down as it approaches. Log-scale?
                if self.ZOOM_IN_DURATION_FRAMES == -1 or (self.total_count > 0 and self.total_count < self.ZOOM_IN_DURATION_FRAMES):
                    percent_from_edge = 1 - (self.total_count / self.ZOOM_IN_DURATION_FRAMES)
                    if self.ZOOM_IN_DURATION_FRAMES == -1:
                        percent_from_edge = 1
                    # print(percent_to_edge)
                    cropped_x1 = int(cropped_x1 - (percent_from_edge * cropped_x1))
                    cropped_y1 = int(cropped_y1 - (percent_from_edge * cropped_y1))

                    cropped_x2 = int(cropped_x2 + (percent_from_edge * abs(self.input_w - cropped_x2)))
                    cropped_y2 = int(cropped_y2 + (percent_from_edge * abs(self.input_h - cropped_y2)))
                    pass
                
                # Draws the crop-preview box
                if self.SHOW_DEBUG:
                    # print(cropped_x1, cropped_y1, cropped_x2, cropped_y2)
                    # Draw the cropping box in blue
                    cv2.rectangle(self.frame, (cropped_x1 // self.ZOOM_FACTOR, cropped_y1 // self.ZOOM_FACTOR), (cropped_x2 // self.ZOOM_FACTOR, cropped_y2 // self.ZOOM_FACTOR), (255, 0, 0), 1)
                    self.frame = cv2.putText(self.frame, 'Cropped frame', (cropped_x1 // self.ZOOM_FACTOR, cropped_y1 // self.ZOOM_FACTOR - 5), cv2.FONT_HERSHEY_DUPLEX, .5, (0,0,0), 2)
                    self.frame = cv2.putText(self.frame, 'Cropped frame', (cropped_x1 // self.ZOOM_FACTOR, cropped_y1 // self.ZOOM_FACTOR - 5), cv2.FONT_HERSHEY_DUPLEX, .5, (255,0,0), 1)

                # Cut down the size of the cropped box to what we actually want, rather than the full frame
                self.cropped = self.cropped[cropped_y1:cropped_y2, cropped_x1:cropped_x2]
                self.cropped = cv2.resize(self.cropped, (self.OUTPUT_SIZE[0], self.OUTPUT_SIZE[1]))

                # Write the name of the clip at the top left
                # Write a black outline, followed by the same text but in white and thinner
                self.cropped = cv2.putText(self.cropped, self.FILENAME, (5, self.OUTPUT_SIZE[1]//40), cv2.FONT_HERSHEY_DUPLEX, .5, (0, 0, 0), 3)
                self.cropped = cv2.putText(self.cropped, self.FILENAME, (5, self.OUTPUT_SIZE[1]//40), cv2.FONT_HERSHEY_DUPLEX, .5, (255, 255, 255), 1)
                # if DRAW_SCOREBOARD
                # TODO we should put some format in the Timestamps file to place score 
                SCOREBOARD_WIDTH = 600
                HALF_SCOREBOARD_WIDTH = SCOREBOARD_WIDTH // 2
                SCORE_SECTION_WIDTH = HALF_SCOREBOARD_WIDTH // 6
                SCOREBOARD_HEIGHT = 30
                # Draw team one side
                logging.debug("Draw scoreboard")
                if self.team_one is not None:
                    # Team one is left-aligned, so we don't need any extra work
                    # Draw black rectangle for Dark score
                    # This is the inner-box, filled in
                    cv2.rectangle(self.cropped, (self.OUTPUT_SIZE[0]//2 - HALF_SCOREBOARD_WIDTH - 1, 0), (self.OUTPUT_SIZE[0]//2 - 1, SCOREBOARD_HEIGHT-2), self.team_one.bg_color, -1)
                    # This is the outer-box, just a thin line
                    cv2.rectangle(self.cropped, (self.OUTPUT_SIZE[0]//2 - HALF_SCOREBOARD_WIDTH - 1, 0), (self.OUTPUT_SIZE[0]//2 - 1, SCOREBOARD_HEIGHT-2), self.team_one.font_color, 1)
                    cv2.putText(self.cropped, self.team_one.name, (self.OUTPUT_SIZE[0]//2 - HALF_SCOREBOARD_WIDTH + 5, SCOREBOARD_HEIGHT-4), cv2.FONT_HERSHEY_DUPLEX, 1, self.team_one.font_color, 1)
                    if self.team_one_score is not None:
                        # Draw a box to put the Score into
                        cv2.rectangle(self.cropped, (self.OUTPUT_SIZE[0]//2 - SCORE_SECTION_WIDTH - 1, 0), (self.OUTPUT_SIZE[0]//2 - 1, SCOREBOARD_HEIGHT-2), self.team_one.font_color, 1)
                        cv2.putText(self.cropped, str(self.team_one_score), (self.OUTPUT_SIZE[0]//2 - SCORE_SECTION_WIDTH + 5, SCOREBOARD_HEIGHT-4), cv2.FONT_HERSHEY_DUPLEX, 1, self.team_one.font_color, 1)
                # Draw team two side
                if self.team_two is not None:
                    # Team two needs to be right-aligned
                    self.team_two_text_size = cv2.getTextSize(self.team_two.name, cv2.FONT_HERSHEY_DUPLEX, 1, 1)[0][0]
                    # print(self.team_two_text_size)
                    # Draw white rectangle for White score
                    # This is the inner-box, filled in
                    cv2.rectangle(self.cropped,(self.OUTPUT_SIZE[0]//2 + HALF_SCOREBOARD_WIDTH, 0), (self.OUTPUT_SIZE[0]//2 + 1, SCOREBOARD_HEIGHT-2), self.team_two.bg_color, -1)
                    # This is the outer-box, just a thin line
                    cv2.rectangle(self.cropped,(self.OUTPUT_SIZE[0]//2 + HALF_SCOREBOARD_WIDTH, 0), (self.OUTPUT_SIZE[0]//2 + 1, SCOREBOARD_HEIGHT-2), self.team_two.font_color, 1)
                    cv2.putText(self.cropped, self.team_two.name, (self.OUTPUT_SIZE[0]//2 + HALF_SCOREBOARD_WIDTH - self.team_two_text_size, SCOREBOARD_HEIGHT-4), cv2.FONT_HERSHEY_DUPLEX, 1, self.team_two.font_color, 1)
                    if self.team_two_score is not None:
                        # Draw a box to put the Score into
                        cv2.rectangle(self.cropped,(self.OUTPUT_SIZE[0]//2 + SCORE_SECTION_WIDTH, 0), (self.OUTPUT_SIZE[0]//2 + 1, SCOREBOARD_HEIGHT-2), self.team_two.font_color, 1)
                        cv2.putText(self.cropped, str(self.team_two_score), (self.OUTPUT_SIZE[0]//2 + 5, SCOREBOARD_HEIGHT-4), cv2.FONT_HERSHEY_DUPLEX, 1, self.team_two.font_color, 1)
                # Write the current image to the video file
                logging.debug(f"Write frame to disk")
                if self.WRITE_VIDEO:
                    if self.total_count < self.ZOOM_IN_DURATION_FRAMES:
                        pass
                        # cropped = imutils.resize(cropped, width=self.OUTPUT_SIZE[0])
                        # cropped = cv2.resize(cropped, (self.OUTPUT_SIZE[0], self.OUTPUT_SIZE[1]))
                    self.cropped_recording.write(self.cropped)
                    # print("FRAME WRITTEN")
                
                self.total_count += 1

                self.old_center = self.new_center
        
                
                # frame = np.bitwise_or(frame, canny[:, :, np.newaxis])

                
                # canny = cv2.Canny(masked_image, 100, 200)
                # cv2.imshow("Canny", canny)

                # if self.SHOW_DEBUG:
                # Green overlay disabled for now
                if False: # DEBUG:
                    print(f"Generate green overlay")
                    # Draw the threshold over top of the live frame
                    frame = np.bitwise_or(frame, thresh[:, :, np.newaxis])
                    # We'd rather this be AFTER the overlay, but it's more complex now that the frame is decimal instead of int
                    # Convert our live frame to Float from int. 
                    frame = np.array(frame, dtype=np.float) # .015 secs
                    frame /= 255.0 # .013 secs

                    # Multiply our green mask by our transparency mask, and then apply it on top of the Live frame
                    frame = green*m2 + frame*(1.0-m2) # .09 secs
                    # Convert the frame back to allow file-writing. If we aren't writing the debug file, don't bother
                    frame *= 255.0
                    frame = np.array(frame, dtype=np.uint8)
                else:  
                    # Draw polygon of the Masked-Area. This is NOT accurate, it does not include the Exclusion zones
                    cv2.polylines(self.frame, self.roi_corners, True, (0, 0, 255), 2)


                if self.SHOW_DEBUG:
                    logging.debug(f"Draw debug rectangles over everything")
                    # Draw the outer buffer box
                    cv2.rectangle(self.frame, (self.left_buffer, self.top_buffer), (self.right_buffer, self.bottom_buffer), (255, 255, 255), 1)
                    self.frame = cv2.putText(self.frame, 'Outer buffer box', (self.left_buffer, self.top_buffer-5), cv2.FONT_HERSHEY_DUPLEX, .5, (0,0,0), 2)
                    self.frame = cv2.putText(self.frame, 'Outer buffer box', (self.left_buffer, self.top_buffer-5), cv2.FONT_HERSHEY_DUPLEX, .5, (255,255,255), 1)

                    # Draw bounds of what the Canny is detecting
                    cv2.rectangle(self.frame, (canny_min_x, canny_min_y), (canny_max_x, canny_max_y), (0, 255, 0), 1)
                    # Draw the center of our zoom
                    cv2.rectangle(self.frame, (self.new_center[0]-1, self.new_center[1]-1), (self.new_center[0]+1, self.new_center[1]+1), (255, 255, 255), 2)
                    # Inner deadzone
                    cv2.rectangle(self.frame, (self.new_center[0] - self.INNER_DEADZONE_X, self.new_center[1] - self.INNER_DEADZONE_Y), (self.new_center[0] + self.INNER_DEADZONE_X, self.new_center[1] + self.INNER_DEADZONE_Y), (255,255,255), 1)
                    # Outer deadzone
                    cv2.rectangle(self.frame, (self.new_center[0] - self.DEADZONE_X, self.new_center[1] - self.DEADZONE_Y), (self.new_center[0] + self.DEADZONE_X, self.new_center[1] + self.DEADZONE_Y), (255,255,255), 1)
                    self.frame = cv2.putText(self.frame, "Movement deadzone", (self.new_center[0] - self.DEADZONE_X, self.new_center[1] - self.DEADZONE_Y - 5), cv2.FONT_HERSHEY_DUPLEX, .5, (0, 0, 0), 2)
                    self.frame = cv2.putText(self.frame, "Movement deadzone", (self.new_center[0] - self.DEADZONE_X, self.new_center[1] - self.DEADZONE_Y - 5), cv2.FONT_HERSHEY_DUPLEX, .5, (255, 255, 255), 1)
                    # This draws a box showing what the Cropped Image will contain
                    # cv2.rectangle(frame, (smooth_top_x * ANALYZE_SHRINK_FACTOR * self.ZOOM_FACTOR, smooth_top_y * ANALYZE_SHRINK_FACTOR * self.ZOOM_FACTOR), (smooth_top_x + FORCED_WIDTH//(self.ZOOM_FACTOR * ANALYZE_SHRINK_FACTOR), smooth_top_y + FORCED_HEIGHT//(self.ZOOM_FACTOR * ANALYZE_SHRINK_FACTOR)), (255, 255, 255), 2)
                    # Draw a pixel at the center of the in-the-moment edges
                    cv2.rectangle(self.frame, (center_x-1, center_y-1), (center_x+1, center_y+1), (0, 0, 255), 1)

                
                # 6) De-Adjust for the next loop?
                self.new_center[0] -= args["offset_x"]
                self.new_center[1] -= args["offset_y"]

                # If we want to keep our debug output, write the "live" frame to a debug file
                if self.DEBUG:
                    logging.debug(f"Write debug view to disk")
                    # Resize the frame to our Analyze Shrink factor
                    live_frame = cv2.resize(self.frame, (self.input_w // self.ANALYZE_SHRINK_FACTOR, self.input_h // self.ANALYZE_SHRINK_FACTOR))
                    self.live_recording.write(live_frame)

                # This is the "debug" view
                if self.SHOW_DEBUG:
                    cv2.imshow("Debug", self.frame)
                # This is our final output, being written to our "good" file
                if not self.QUIET:
                    cv2.imshow("Cropped", self.cropped)
                # cv2.imshow("ROI Mask", mask)
                # cv2.imshow("Threshold", thresh)
                # cv2.imshow("Preview", preview)
            
                # cv2.imshow("masked_image", masked_image)
                # cv2.imshow("Thresh", thresh)
                # cv2.imshow("Frame Delta", frameDelta)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key is pressed, break from the lop
            if key == ord("q"):
                break
            self.count += 1
            # total_count += 1
            # input("ENTER TO CONTINUE")
        # cleanup the camera and close any open windows
        if self.WRITE_VIDEO:
            self.cropped_recording.release()
        if self.DEBUG:
            self.live_recording.release()
        self.vs.stop() if args.get("video", None) is None else self.vs.release()

        cv2.destroyAllWindows()
        self.process_end_time = datetime.datetime.now()
        # self.process_start_time = datetime.datetime.now()
        difference = (self.process_end_time - self.process_start_time).total_seconds()
        # FPS of video / FPS of process = Multiplier of our time taken
        # 1 / self.fps will get us what we "need" to process to hit 1x time
        processing_fps = None
        time_multiplier = None
        if difference != 0:
            processing_fps = self.count / difference
            time_multiplier = self.fps / processing_fps
        # 
        logging.info("Clip completed")
        if processing_fps is not None:
            logging.info(f"Processed {self.count} frames. Total seconds: {difference}. Time per frame: {difference/self.count}. {time_multiplier:2.4f}x")
        return 0 

def run_args(zoom_args):
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=50, help="minimum area size")
    ap.add_argument("-d_x", "--deadzone_x", type=int, default=80, help="Deadzone where the camera does not pan in x-axis")
    ap.add_argument("-d_y", "--deadzone_y", type=int, default=60, help="Deadzone where the camera does not pan in y-axis")
    ap.add_argument("-m_x", "--max_camera_movement_x", type=int, default=3, help="Max pixels per frame the camera pans")
    ap.add_argument("-m_y", "--max_camera_movement_y", type=int, default=1, help="Max pixels per frame the camera tilts")
    ap.add_argument("-ob_x", "--outer_bound_buffer_x", type=int, default=100, help="Number of pixels from the edge -> crop_outer at which point the camera needs to shift.")
    ap.add_argument("-ob_y", "--outer_bound_buffer_y", type=int, default=100, help="Number of pixels from the edge -> crop_outer at which point the camera needs to shift.")
    ap.add_argument("--offset_x", type=int, default=0, help="Offset to shift the camera left or right. Negative values for left. Default 0.")
    ap.add_argument("--offset_y", type=int, default=0, help="Offset to shift the camera up or down. Negative values for up. Default 0.")
    ap.add_argument("--zoomin_duration_frames", type=int, default=600, help="Number of frames to spend zooming in at the beginning of a clip")
    ap.add_argument("--seconds_between_comparisons", type=float, default=.5, help="Number of seconds between each comparison frame")
    # ap.add_argument("--skip_starting_frames", type=int, default=0, help="Skip this many frames at the start of the video. Generally used for testing events further into a clip.")
    ap.add_argument("--disable_filename_overlay", default=False, action='store_true', help="Disable writing the filename in the upper-left corner of a clip")
    ap.add_argument("-r", "--roi_filepath", default=None, help="File containing ROI info. Generated by roi_selector.py")
    ap.add_argument("-o", "--output_filename", default=None, help="Output video file name")
    ap.add_argument("-w", dest="write", default=False, action='store_true', help="Write output to file")
    ap.add_argument("-d", "--debug", dest="debug", default=False, action='store_true', help="Debug output. Also will write the Debug frame to a file")
    ap.add_argument("-o_e", "--overwrite_existing", default=False, action='store_true', help="Flag to enable overwriting existing files. Default is False")
    ap.add_argument("--show_debug", dest="show_debug", default=False, action='store_true', help="Show live debug output")
    ap.add_argument("--team_one_file", help="Path to file consisting of info for Team 1, on the left", type=str, default=None)
    ap.add_argument("--team_two_file", help="Path to file consisting of info for Team 2, on the right", type=str, default=None)
    ap.add_argument("--team_one_score", "--t1s", help="Current score for Team 1", type=int, default=None)
    ap.add_argument("--team_two_score", "--t2s", help="Current score for Team 2", type=int, default=None)
    args = vars(ap.parse_args(zoom_args))
    print(args)
    z = ZoomedClipCreator(args)
    return z.process_video()
    # return main(args)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=50, help="minimum area size. Default=50")
    ap.add_argument("-d_x", "--deadzone_x", type=int, default=80, help="Deadzone where the camera does not pan in x-axis")
    ap.add_argument("-d_y", "--deadzone_y", type=int, default=60, help="Deadzone where the camera does not pan in y-axis")
    ap.add_argument("-m_x", "--max_camera_movement_x", type=int, default=3, help="Max pixels per frame the camera pans")
    ap.add_argument("-m_y", "--max_camera_movement_y", type=int, default=1, help="Max pixels per frame the camera tilts")
    ap.add_argument("-ob_x", "--outer_bound_buffer_x", type=int, default=100, help="Number of pixels from the edge -> crop_outer at which point the camera needs to shift.")
    ap.add_argument("-ob_y", "--outer_bound_buffer_y", type=int, default=100, help="Number of pixels from the edge -> crop_outer at which point the camera needs to shift.")
    ap.add_argument("--offset_x", type=int, default=0, help="Offset to shift the camera left or right. Negative values for left. Default 0.")
    ap.add_argument("--offset_y", type=int, default=0, help="Offset to shift the camera up or down. Negative values for up. Default 0.")
    ap.add_argument("--zoomin_duration_frames", type=int, default=600, help="Number of frames to spend zooming in at the beginning of a clip")
    ap.add_argument("--seconds_between_comparisons", type=float, default=.5, help="Number of seconds between each comparison frame")
    # ap.add_argument("--skip_starting_frames", type=int, default=0, help="Skip this many frames at the start of the video. Generally used for testing events further into a clip.")
    ap.add_argument("--disable_filename_overlay", default=False, action='store_true', help="Disable writing the filename in the upper-left corner of a clip")
    ap.add_argument("-r", "--roi_filepath", default=None, help="File containing ROI info. Generated by roi_selector.py")
    ap.add_argument("-o", "--output_filename", default=None, help="Output video file name")
    ap.add_argument("-w", dest="write", default=False, action='store_true', help="Write output to file")
    ap.add_argument("-d", "--debug", dest="debug", default=False, action='store_true', help="Debug output. Also will write the Debug frame to a file")
    ap.add_argument("-o_e", "--overwrite_existing", default=False, action='store_true', help="Flag to enable overwriting existing files. Default is False")
    ap.add_argument("--show_debug", dest="show_debug", default=False, action='store_true', help="Show live debug output")
    ap.add_argument("--team_one_file", help="Path to file consisting of info for Team 1, on the left", type=str, default=None)
    ap.add_argument("--team_two_file", help="Path to file consisting of info for Team 2, on the right", type=str, default=None)
    ap.add_argument("--team_one_score", "--t1s", help="Current score for Team 1", type=str, default=None)
    ap.add_argument("--team_two_score", "--t2s", help="Current score for Team 2", type=str, default=None)
    args = vars(ap.parse_args())

    z = ZoomedClipCreator(args)
    z.process_video()
    # main(args)