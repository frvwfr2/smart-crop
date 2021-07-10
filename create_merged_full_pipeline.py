# Create Merged w Audio.py
# Assume we already have a merged video file

# We need to...
# 0) Merge all clips together into one video
    # 0a) Get timestamps of "Point 1 started at 0:25, ended at 0:50." Repeat for whole game
# 1) Get the video file path
# 2) Get the timestamps file path
# 3) Cut each clip into 001 -> 0XX. Place these into a _workingdir ?
# 4) Generate ROI from the first clip and trust it for all others. 
#   5) Do the Tracking on each clip, using the ROI generated.
#   6) fix_audio.py on each clip
# 7) Re-merge all the clips

# Future TODO
# Add an outer-bound smoothness
# Add a way to put text onto clips, on demand. Should be provided by the Timestamps file. This file can pass the Text through via command line arguments.
# This should be a final step before writing to file.
# Update the ROI selector with Description of what to do
# Add "Cone ID" to ROI Selector, for use below
# Add yardage markers? Point to the 4 inner-cones, then draw lines down the field
# Yardage markers - Take 4 points. Move from point 1 to point 2. Subset this into 7 pieces. Do the same for 3->4. Connect (1,2[i]) to (3,4[i]). Repeat for i=1->7
# Team names and scoreboard

# TODO
# When objects are past the frame, move the frame. This will be a 3rd condition for movement
# "Zoom out" when objects are outside the frame, to keep all objects in frame
# Implement the Edge + Motion Detection for lined fields
# Implement "only parse every X frame"

import os
import subprocess
import argparse
from mass_clipper import generate_clips
import glob
from roi_selector import create_roi_from_video
from fix_audio import fix_audio
import sys


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the merged video file", required=True)
ap.add_argument("-t", "--timestamps", help="Path to the timestamps file")
ap.add_argument("-r", "--roi_file", help="File path to existing ROI file to use", default=None)
ap.add_argument("-d", "--directory", help="Full path to the directory to write files into.")
args = vars(ap.parse_args())

print(args)

# This will cut each clip from the longer video
clip_count = int(generate_clips(args["video"], args["timestamps"]))
# clip_count = 11
if args["roi_file"]:
    roi_filename = args["roi_file"]
    print(f"ROI File located, proceding. {roi_filename}")
else:
    print("No ROI file provided, please input values in the window")
    roi_filename = create_roi_from_video(args["video"])
    print(f"ROI File was written to {roi_filename}")

if args["directory"]:
    directory = args["directory"]
else:
    directory = os.path.dirname(args["video"])

# File to write our final mergelist to
with open(f"{directory}\post_file_mergelist.txt", 'w') as f:
    for i in range(1, clip_count + 1):
        print(f"WORKING ON CLIP #{i}")
        # command = f'motion_detector_test.py -v "{directory}\clip_{i:03d}.MP4" --roi_filepath "{roi_filename}" -o "{directory}\clip_{i:03d}_pipeline.mp4"'
        command = f'motion_detector_test.py -v "{directory}\clip_{i:03d}.MP4" --roi_filepath "{roi_filename}" -o "{directory}\clip_{i:03d}_pipeline.mp4" -w'
        print(command)
        subprocess.call(command, shell=True)
        print("fixing audio")
        fix_audio(video_file=f"clip_{i:03d}_pipeline.mp4", audio_file=f'clip_{i:03d}.MP4', directory=directory)
        # Write the filename to the mergelist
        f.write(f"file '{directory}\clip_{i:03d}_pipeline.mp4_audio.mp4'")
        # sys.exit()
        # pass
