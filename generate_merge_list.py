from glob import glob
from os import DirEntry, path
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--directory", required=True)
args = parser.parse_args()

def get_group_and_index_from_path(filepath):
    shortpath = path.basename(filepath)
    index = shortpath[2:4]
    group = shortpath[4:-4]
    # print(group, index)
    return group, index
# F:\\Media\\RawClips\\2021 Hodown\\
groups = set()
for filepath in glob(f'{args.directory}\\*.MP4'):

    # print(file)
    group, index = get_group_and_index_from_path(filepath)
    # print(group, index)
    with open(f"{args.directory}\\{group}_merge.txt", 'a') as f:
        print(f"file '{filepath}'", file=f)
        groups.add(f"{group}_merge.txt")

print(f"{len(groups)} files to concat")
for entry in groups:
    command = f'ffmpeg -loglevel panic -f concat -safe 0 -i "{args.directory}\\{entry}" -y -c copy "{args.directory}\\{entry}.mp4"'
    print(command)
    subprocess.run(command)
