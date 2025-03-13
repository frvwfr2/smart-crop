This utility zooms in on provided video, and attempts to auto-track movement within a specified area.

A sample "timestamps.txt" file is provided.

Future: Add additional files for Team Names, team colors, team members. Could have args for "--left_team_info team_one.txt --right_team_info team_two.txt"

## Future todos
Have the config be a separate file - not stuffed into a billion CLI arguments
Add Times to the videos - can be done based on the "start of clip 1".. until halftime, somehow keeping track of each "clip" as we go through? Def doable but need to think about it
Add some way to have a "pre-game message" - Big text box showing "20 minute halves" type of thing. Better way to show rosters maybe too.

## Basic Usage
### 1. Merge separate GoPro videos into "merged" clips of each video segment.  
This is needed when you have a pile of 8 minute videos from a GoPro, that you need to combine in order to watch

generate_merge_list.py --directory "samples"

### 2. Create a timestamps.txt file
This file contains info about when each point occurred. A sample is located in "samples/timestamps.txt"

Each entry can have custom parameters applied.

### 3. Run on a full video. This completes the steps.

create_merged_full_pipeline.py -v "samples\merged.mp4" -t "samples\timestamps.txt" -r "samples\roi.json" --team_one_file "samples\team_one.txt" --team_two_file "samples\team_two.txt"

-r is the path to the ROI (region of interest) file. If you do not have one, remove this argument, and the tool will prompt for one.

## Separate pieces. These are not needed for basic usage
### Break up the long-file into each point.
This file cuts up a full video into each clip. This produces one new video for each point specified in the timestamps file.

mass_clipper.py -v "samples\full_video.mp4" -t "samples\timestamps.txt"

### Run on a single clip. This will not include the audio
For testing an ROI, or various settings. This is generally run from the create_merged_full_pipeline

create_zoomed_clip.py -v "clip_028.MP4" --team_one_file "team_one.txt" --team_two_file "team_two.txt" --t1s 8 --t2s 10
