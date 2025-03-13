# Reads a file with times, and makes clips

import argparse
import subprocess
import re
import os
import json

from imutils import video
from numpy.core.numeric import full

class Clip:
    def __init__(self, start, end, name, arguments=""):
        self.start = start
        self.end = end
        self.name = name
        self.arguments = arguments
        self.team_one_score = None
        self.team_two_score = None

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def generate_clips(video_path, input_file, output_dir=None, base_output_prefix="clip"):
    clips = list()
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    print(video_path, output_dir, input_file)

    start_string = None
    args = ""

    # Parse the timestamps file and produce a list of timestamps we care about. That list is timecodes
    with open(input_file) as f:
        count = 0
        for line in f.readlines():
            # If a line starts with whitespace, move on. This is probably a comment line.
            if re.match(r'\s', line):
                continue
            if line.startswith("#"):
                continue
            # print(line)
            # We found a relevant timecode
            start_string = line.split()[0]
            if isinstance(start_string, int):
                pass
            else:
                start = start_string.split(":")
                if len(start) == 3:
                    h = int(start[0])
                    m = int(start[1])
                    s = int(start[2])
                    start_secs = s + m*60 + h*60*60
                elif len(start) == 2:
                    m = int(start[0])
                    s = int(start[1])
                    start_secs = s + m*60
            # print(start_secs, end_secs)
            # Get 15 seconds in front of the turnover, and 10 seconds afterwards
            start_secs -= 10
            end_secs = start_secs + 15
            c = Clip(start_secs, end_secs, f"{base_output_prefix}_{count:03d}.MP4", args)
            clips.append(c)
            count += 1
            # Reset all the variables
            start_string = None
            args = ""
    if start_string is not None:
        raise ValueError("Entries in timestamps file don't match. There must be an even number of entries.")
    
    for clip in clips:
        print(clip.start, clip.end)
        # Do some mumbo-jumbo to cut the clips faster. 
        # Basically, if we want something 45 minutes in, do a fast-forward to 44m50s, and then do a slow-forward for the next 10 secs. Saves a ton of time
        if clip.start > 10:
            skip_secs = clip.start - 10
            clip_start = 10
            clip_end = clip.end - skip_secs
        else:
            skip_secs = 0
            clip_start = clip.start
            clip_end = clip.end
        # print(clip.arguments)
        if "-o_e" in clip.arguments or "--overwrite_existing" in clip.arguments:
            print("Regenerating clip")
            overwrite_val = "-y"
        else:
            overwrite_val = "-n"
        print(f"Generating clip '{clip.name}' from {clip.start} to {clip.end} ({clip.end - clip.start} seconds)")
        full_command = f'ffmpeg -ss {skip_secs} -i "{video_path}" -ss {clip_start} -to {clip_end} -loglevel fatal -hide_banner {overwrite_val} -c:v copy -c:a copy "{output_dir}\\{clip.name}"'
        print(full_command)
        subprocess.run(full_command)

    return clips

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the input video file")
    ap.add_argument("-t", "--times", help="Path to the file with Timestamps")
    ap.add_argument("--output_dir", help="Output directory. Default is same folder as source", default=None)
    ap.add_argument("--base_output_prefix", help="Output to pre-pend to the output videos. Clips are numbered from 001 -> X", default="turnover")
    args = vars(ap.parse_args())

    print(args)
    input_file = args["times"]
    video_path = args["video"]
    if args["output_dir"] is not None:
        output_folder = args["output_dir"]
    else:
        output_folder = os.path.dirname(video_path)
    base_output_prefix = args["base_output_prefix"]

    generate_clips(video_path, input_file, output_folder, base_output_prefix)

    print("RUN SOMETHING LIKE THIS")
    print(rf"""ffmpeg -loglevel fatal -hide_banner -f concat -safe 0 -i "{output_folder}\turnover_filelist.txt" -c copy "{output_folder}\combined_turnovers.mp4" """)
    

if __name__ == "__main__":
    main()