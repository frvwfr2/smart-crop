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

    timecodes = list()
    start_string = None
    end_string = None
    args = ""

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
            timecode = line.split()[0]
            # If we just found the start of a clip...
            # We need to check for arguments. Do a split only once, to get rid of timecode.
            # Then do a split on the previous split[1], targeting # to split on, and use [0] index. Then trim white space from the ends. This should be valid command line args
            # Do we only want to do it if it starts with "-" ? Any args should be hyphens
            if start_string == None:
                count += 1
                start_string = timecode
                # Split once, to get rid of timecode. p for post-timecode
                if len(line.split()) > 1:
                    p = line.split(maxsplit=1)[1]
                    # a for args
                    # split the post-timecode on #, only once. Use the stuff prior to #. 
                    a = p.split("#", 1)[0].strip()
                    # If it starts with a hyphen, we want to use it
                    if a.startswith("-"):
                        args = a
                    # Otherwise, we don't want to use it, and just pass back Blank
                    else:
                        args = ""
                else:
                    args = ""
            # If we just found the end of a clip... We use th
            else:
                end_string = timecode

                # Parse the start_string and end_string into integer seconds
                try:
                    start_string = int(start_string)
                except ValueError:
                    pass
                try:
                    end_string = int(end_string)
                except ValueError:
                    pass

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
                if isinstance(end_string, int):
                    end_secs = end_string
                else:
                    end = end_string.split(":")
                    if len(end) == 3:
                        h = int(end[0])
                        m = int(end[1])
                        s = int(end[2])
                        end_secs = s + m*60 + h*60*60
                    elif len(end) == 2:
                        m = int(end[0])
                        s = int(end[1])
                        end_secs = s + m*60
                # print(start_secs, end_secs)
                c = Clip(start_secs, end_secs, f"{base_output_prefix}_{count:03d}.MP4", args)
                clips.append(c)
                # Reset all the variables
                start_string = None
                end_string = None
                args = None
            # print(count, timecode)
            timecodes.append(timecode)
    if start_string is not None:
        raise ValueError("Entries in timestamps file don't match. There must be an even number of entries.")
    
    # Split the list of timecodes into pairs
    # pairs = list()
    # for entry in chunks(timecodes, 2):
    #     pairs.append(entry)

    # If the timestamps 
    # if len(pairs[-1]) != 2:
    #     raise ValueError("Entries in timestamps file don't match. There must be an even number of entries.")

    # for i, clip in enumerate(pairs):
    #     print(f"clip_{i+1:03d} {clip[0]:>10} to {clip[1]:>10}")
    #     # print(clip)

    for clip in clips:
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

    # count = 0
    # for pair in pairs:
    #     count += 1
    #     full_command = f'ffmpeg -i "{video_path}" -ss {pair[0]} -to {pair[1]} -loglevel fatal -hide_banner -n -c:v copy -c:a copy "{output_dir}\\{base_output_prefix}_{count:03d}.MP4"'
    #     print(full_command)
    #     # Add this Clip, and its metadata, to the list we will return
    #     clips.append(Clip(pair[0], pair[1], f"{base_output_prefix}_{count:03d}.MP4"))
    #     subprocess.run(full_command)
    #     # break
    # return count

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the input video file")
    ap.add_argument("-t", "--times", help="Path to the file with Timestamps")
    ap.add_argument("--output_dir", help="Output directory. Default is same folder as source", default=None)
    ap.add_argument("--base_output_prefix", help="Output to pre-pend to the output videos. Clips are numbered from 001 -> X", default="clip")
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
    

if __name__ == "__main__":
    main()