# Reads a file with times, and makes clips

import argparse
import subprocess
import re
import os
import json

from imutils import video

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def generate_clips(video_path, input_file, output_dir=None, base_output_prefix="clip"):
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    print(video_path, output_dir, input_file)

    timecodes = list()
    with open(input_file) as f:
        count = 0
        for line in f.readlines():
            # If a line starts with whitespace, move on. This is probably a comment line.
            if re.match(r'\s', line):
                continue
            # print(line)
            timecode = line.split()[0]
            
            count += 1
            # print(count, timecode)
            timecodes.append(timecode)
    
    pairs = list()
    for entry in chunks(timecodes, 2):
        pairs.append(entry)
    print(pairs)

    count = 0
    for pair in pairs:
        count += 1
        full_command = rf'ffmpeg -i "{video_path}" -ss {pair[0]} -to {pair[1]} -y -c:v copy -c:a copy "{output_dir}\\{base_output_prefix}_{count:03d}.MP4"'
        print(full_command)
        subprocess.run(full_command)
        # break
    return count

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