#Fix audio

import subprocess
import argparse

def fix_audio(video_file, audio_file):
    # video_id_num = 1
    # video_count = 9
    # directory = r"F:\Media\RawClips\2021-06-19 Practice"
    # audio_file = f"scrimmage_{i:03d}.MP4"
    # video_file = f"scrimmage_{i:03d}.MP4_cropped.mp4"
    command = f'ffmpeg -i "{audio_file}" -i "{video_file}" -loglevel fatal -hide_banner -c copy -map 1:v:0 -map 0:a:0 -shortest "{video_file}_audio.mp4" -y'

    print(command)
    subprocess.call(command)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video_file", help="path to the video file", required=True)
    ap.add_argument("-a", "--audio_file", help="path to the audio file", required=True)
    args = vars(ap.parse_args())
    fix_audio(args["video_file"], args["audio_file"])

# TODO
# Combine all steps into one python script

# Merge 1-2-3-4 gopro videos (CREATE CODE TO DO THIS, low priority)
# MANUAL: Create timestamps for each point, stick into file. 
# Split the full video into each point (mass_clipper.py)
# MANUAL AND CODE: UI for drawing the "whitelist area". -> roi_selector.py is the helper function for this. Not sure we can combine it all into one piece with this included.
# FOR EACH POINT:
    # Run the zoom/crop (motion_detector_test.py)
        # Currently, the ROI manually needs to be added in for each piece
    # Merge the audio/video (fix_audio.py) (no CLI support yet)
    # Merge each cropped_w_audio video back into one big video

# TODO
# Output a file containing info used for this clip.
# roi_selector data for example. Add a way to delete an entry if coordinates are too close 
# Add a way for Canny detector to ignore pieces - such as Cones. Similar to ROI Selection, but inversed.
# Could use roi_selector and right-click to designate "bad-coordinates"