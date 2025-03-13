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
# DONE - Add an outer-bound smoothness
# Add a way to put text onto clips, on demand. Should be provided by the Timestamps file. This file can pass the Text through via command line arguments.
# This should be a final step before writing to file, to avoid being written over top of by lines, cropped, etc
# DONE-ISH - Update the ROI selector with Description of what to do
# DE-FISHEYE THE FOOTAGE. Need a chessboard? 
#   https://www.theeminentcodfish.com/gopro-calibration/
#   https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
# Add "Cone ID" to ROI Selector
#   Add yardage markers - Point to the 4 inner-cones, then draw lines down the field
#   Yardage markers - Take 4 points. Move from point 1 to point 2. Subset this into 7 pieces. Do the same for 3->4. Connect (1,2[i]) to (3,4[i]). Repeat for i=1->7
# Team names and scoreboard
# When objects are past the frame, move the frame. This will be a 3rd condition for movement
# "Zoom out" when objects are outside the frame, to keep all objects in frame
# Implement the Edge + Motion Detection for lined fields
#   https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
# DONE - Implement "only parse every X frame"
#   Not very useful. Not much speed up. Possibly more to be done for this - a thread ONLY for collecting new frames from the video
# DONE - Implement "x_offset" and "y_offset" args for create_zoomed_clip. Allow manual adjustment for when clips are not keeping up with action.
#   A bit messy to do this for files that need left- and right- shifts done. Edge case though. Unlikely to need it.
# Implement ability to parse args in the Timestamps file. Then clips could be generated by reading those values, and if a file needs to manually "shift left", that can be done.
# Implement "force_redo" argument. Change default to "If the clip already exists, don't do it again." This would enable updating things after the fact, like the x_offset value above.
#   This arg should be parsed in the create_merged_full_pipeline file itself
#   Purpose is to be able to have the timestamps file be a "pipeline" of steps to do - and we can be efficient and not re-do steps that don't need to be re-done upon review.
# Output working-pieces into a sub-folder, but the "final output" into the main folder?
# Delete the mid-step parts - only keep the RawClip and ZoomedWithAudio pieces. Delete the "zoomed clip, but no audio"
# Speed up extraction of clips! Get the timestamp, subtract all but 5 minutes off of it, and modify the command as such

import os
import subprocess
import argparse
from typing import final
from mass_clipper import generate_clips
import create_zoomed_clip
import glob
from roi_selector import create_roi_from_video
from fix_audio import fix_audio
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the merged video file", required=True)
ap.add_argument("-t", "--timestamps", help="Path to the timestamps file")
ap.add_argument("-r", "--roi_file", help="File path to existing ROI file to use", default=None)
ap.add_argument("-d", "--directory", help="Full path to the directory to write files into. Optional. Defaults to the input file directory")
ap.add_argument("--no_zoom", action='store_true', help="Don't perform zooms, just chop up the clips")
ap.add_argument("--base_output_prefix", help="Prefix to use on the in-between clips that are generated", default="clip")
ap.add_argument("--debug", help="Enable debug mode in the zoomed_clip_maker", action="store_true")
ap.add_argument("--team_one_file", help="Path to the file describing team_one", default=None)
ap.add_argument("--team_two_file", help="Path to the file describing team_two", default=None)
args = vars(ap.parse_args())

print(args)

if args["directory"]:
    directory = args["directory"]
else:
    directory = os.path.dirname(args["video"])

# This will cut each clip from the longer video
# clip_info needs to get info on what CLI arguments need to exist
# needs to be a list of tuples? List is in order of each clip. 
# We should just make an object for each Clip, and store the metadata about it

clips = generate_clips(args["video"], args["timestamps"], base_output_prefix=args["base_output_prefix"], output_dir=directory)
# print(clips)
clip_count = len(clips)
print(f"Found {clip_count} clips")
# clip_count = 12


if args["no_zoom"]:
    print("Not performing zoom steps - only cutting up the points")
    pass
elif args["roi_file"]:
    roi_filename = args["roi_file"]
    print(f"ROI File located, proceding. {roi_filename}")
else:
    print("No ROI file provided, please input values in the window")
    # roi_filename = create_roi_from_video(args["video"])
    # If we have more than one clip, use the 2nd to create the ROI in case we were moving the camera at the start of Clip_01
    if len(clips) > 1:
        roi_filename = create_roi_from_video(f"{directory}\{clips[1].name}")
    else:
        roi_filename = create_roi_from_video(f"{directory}\{clips[0].name}")
    print(f"ROI File was written to {roi_filename}")

# File to write our final mergelist to
with open(f"{directory}\post_{args['base_output_prefix']}_files.txt", 'w') as f:
    # For each Clip we produced...
    for clip in clips:
        if args["no_zoom"]:
            f.write(f"file '{directory}\\{clip.name}'\n")
        else:
            f.write(f"file '{directory}\\zoomed_{clip.name}_audio.mp4_reencode.mp4'\n")
        print("CLIP:", clip.name, clip.start, clip.end, clip.arguments)
        if args["no_zoom"]:
            continue
    # for i in range(1, clip_count + 1):
        print(f"WORKING ON {clip.name}")
        # zoom_parser = argparse.ArgumentParser()
        # Create the arguments to pass to create_zoomed_clip.py
        zoom_args = ['-v', f"{directory}\{clip.name}", '--roi_filepath', roi_filename, "-o", f"{directory}\zoomed_{clip.name}", "-w"]
        if args["team_one_file"]:
            zoom_args.extend(["--team_one_file", args["team_one_file"]])
        if args["team_two_file"]:
            zoom_args.extend(["--team_two_file", args["team_two_file"]])
        if clip.arguments:
            zoom_args.extend(clip.arguments.split())
        if args["debug"]:
            zoom_args.append("-d")
            zoom_args.append("--show_debug")
        zoom_status = create_zoomed_clip.run_args(zoom_args)
        if zoom_status != 0:
            print("Zoom step was not performed")
            continue
        print("fixing audio")
        # Get the final file out. We have to re-encode it before being usable however.
        fixed_audio_file = fix_audio(video_file=f"{directory}\zoomed_{clip.name}", audio_file=f'{directory}\{clip.name}')

        # Re-encode the file
        command = f'ffmpeg -i "{fixed_audio_file}" -loglevel fatal -hide_banner -s hd1080 -r 30000/1001 -video_track_timescale 30k -c:a copy -y "{fixed_audio_file}_reencode.mp4"'
        print(command)
        subprocess.call(command)

        # delete the two component files for the audio + video, that have since been re-encoded
        os.remove(f"{directory}\zoomed_{clip.name}")
        # os.remove(f'{directory}\{clip.name}')
        os.remove(f"{fixed_audio_file}")
        # Write the filename to the mergelist
        # sys.exit()
        # pass

print(f'To concatenate, run something like\n')
print(f'ffmpeg -loglevel fatal -hide_banner -f concat -safe 0 -i "{directory}\post_{args["base_output_prefix"]}_files.txt" -c copy "{directory}\\final_{args["base_output_prefix"]}.mp4"')
