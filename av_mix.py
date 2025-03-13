import os
import subprocess
from datetime import datetime, timedelta
import sys


def get_audio_duration(file_path):
    """Gets the duration of an audio file using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip()) if result.stdout.strip() else 0


# Function to extract the creation timestamp of a media file
def get_creation_time(file_path):
    command = ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
         "-of", "default=noprint_wrappers=1:nokey=1", file_path]
    # print(f"{command=}")
    result = subprocess.run(command,
        capture_output=True, text=True
    )
    timestamp = result.stdout.strip()
    
    if timestamp:
        # Convert to datetime object
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")  # Adjust format if needed
    else:
        # print(f"No timestamp found for {file_path}. Using system modified time.")
        return datetime.fromtimestamp(os.path.getmtime(file_path))  # Use file modified time as fallback
    
def create_temp_audio_file1():
    pass
    

# Paths
video_folder = "F:\\Media\\RawClips\\AV Mixing"  # Use raw string or double backslashes
audio_file1 = "F:\\Media\\RawClips\\AV Mixing\\00004_Wireless_PRO.WAV"  # Path to the long audio file
audio_file2 = "F:\\Media\\RawClips\\AV Mixing\\00004_Wireless_PRO.WAV"
output_folder = "F:\\Media\\RawClips\\AV Mixing\\output_videos"

audio_shift_seconds = +3

os.makedirs(output_folder, exist_ok=True)

# Get all video files sorted by timestamp
video_files = sorted(
    [f for f in os.listdir(video_folder) if f.endswith(".MP4")],
    key=lambda x: os.path.getmtime(os.path.join(video_folder, x))
)

print(f"{len(video_files)=}")

# Get audio file's start timestamp
audio_end_time = get_creation_time(audio_file1) + timedelta(hours=1) # Adjust for daylight savings?...

# Get the duration of the audio file
audio_duration = get_audio_duration(audio_file1)
print(f"Audio Duration: {audio_duration} seconds")
audio_start_time = audio_end_time - timedelta(seconds=audio_duration) + timedelta(seconds=audio_shift_seconds)
print(f"{audio_start_time=}")

for video in video_files:
    # Hard code to only get video 4, where we know some chatter is
    # It should be telling us the audio_offset is 325 seconds (5m25s)
    # video = video_files[3]

    audio1 = create_temp_audio_file1(audio_file1)
    audio2 = create_temp_audio_file1()


    print(f"Getting start+duration for {video}")
    video_path = os.path.join(video_folder, video)
    output_path = os.path.join(output_folder, video)

    # Get video duration
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    # print(f"{' '.join(command)}")
    duration_result = subprocess.run(
        command,
        capture_output=True, text=True)
    video_duration = float(duration_result.stdout.strip()) if duration_result.stdout.strip() else 0

    # Get video creation time
    video_start_time = get_creation_time(video_path) - timedelta(seconds=video_duration)

    # Calculate how far into the audio file this video starts
    audio_offset = (video_start_time - audio_start_time).total_seconds()
    if audio_offset < 0:
        audio_offset = 0  # Prevent negative values
    
    print(f"{video_duration} {video} ({video_start_time} - {audio_start_time} = {audio_offset})")

    # Temporary audio file for the extracted section
    temp_audio_path = os.path.join(output_folder, f"temp_{video}.wav")
    print(f"Cutting audio for '{video}'")
    # print(f"{video_path=} {output_path=} {audio_offset=} {duration=} {temp_audio_path=}")

    # Extract the correct portion of audio using ffmpeg
    command = [
        "ffmpeg", "-y", "-i", audio_file1, "-ss", str(audio_offset), "-t", str(video_duration),
        "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2", temp_audio_path
    ]
    # print(f"{' '.join(command)}")
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL

    # Merge video with extracted audio
    command = [
        "ffmpeg", "-y", "-i", video_path, "-i", temp_audio_path, "-i", 
        "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", output_path
    ]
    # print(f"{' '.join(command)}")
    print(f"Merging audio '{temp_audio_path}' with video '{video}'")
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Remove temporary audio file
    os.remove(temp_audio_path)
    # break

print("Processing complete. Check the output_videos folder.")
