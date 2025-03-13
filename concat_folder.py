import os
import subprocess

# Folder containing video files
video_folder = "F:\\Media\\RawClips\\AV Mixing\\output_videos"  # Change this to your folder
output_file = os.path.join(video_folder, "concatenated_output.mp4")  # Output file name
list_file = os.path.join(video_folder, "video_list.txt")  # Temporary file list for ffmpeg

# Get all MP4 files in the folder, sorted by modification time
video_files = sorted(
    [f for f in os.listdir(video_folder) if f.endswith(".MP4")],
    key=lambda x: os.path.getmtime(os.path.join(video_folder, x))
)

# Check if we have enough files
if len(video_files) < 2:
    print("Not enough videos to concatenate. Need at least 2.")
    exit()

# Create a file list for ffmpeg
with open(list_file, "w") as f:
    for video in video_files:
        f.write(f"file '{os.path.join(video_folder, video).replace('\\', '/')}'\n")

# Run ffmpeg to concatenate videos
subprocess.run([
    "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file,
    "-c", "copy", output_file
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Cleanup
os.remove(list_file)

print(f"Concatenation complete. Output saved as: {output_file}")
