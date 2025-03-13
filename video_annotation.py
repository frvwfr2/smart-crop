import cv2
import numpy as np

# Load the video
# video_path = 'F:\\Media\\RawClips\\Triumph\\2023-08 Hodown\\vs CC2\\0178_merge.txt.mp4'
video_path = 'F:\\Media\\RawClips\\2022-11-22 Grand Finals - Clegg vs Kenline.mp4'

# 2022-11-22 Grand Finals - Clegg vs Kenline.mp4
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video frame rate
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Window setup
cv2.namedWindow('Video Player', cv2.WINDOW_NORMAL)

# Variables
play = True
frame = None
drawing = False
start_point = None
overlay = None
reverse = False
color = (0, 255, 0)  # Default to neon-green
temp_overlay = None
thickness = 2
jump_frames = 10 * fps
overlay_history = []

def draw(event, x, y, flags, param):
    global drawing, start_point, overlay, color, last_saved_overlay

    if event == cv2.EVENT_LBUTTONDOWN:
        overlay_history.append(overlay.copy())
        drawing = True
        start_point = (x, y)
        color = (0, 255, 0)  # Neon-green
    elif event == cv2.EVENT_RBUTTONDOWN:
        overlay_history.append(overlay.copy())
        drawing = True
        start_point = (x, y)
        color = (0, 0, 255)  # Red
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(overlay, start_point, (x, y), color, thickness)
            start_point = (x, y)  # Update the start point for the next segment
    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
        drawing = False

cv2.setMouseCallback('Video Player', draw)

while True:
    if play and not reverse:
        ret, frame = cap.read()
        if not ret:
            break
    elif play and reverse:
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 2)
        ret, frame = cap.read()
        if not ret:
            break

    if overlay is None:
        overlay = np.zeros_like(frame)

    if temp_overlay is not None:
        combined = cv2.addWeighted(frame, 1, temp_overlay, 0.9, 0)
    else:
        combined = cv2.addWeighted(frame, 1, overlay, 0.9, 0)

    cv2.imshow('Video Player', combined)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('p') or key == ord(' '):
        play = not play
    elif key == ord('r'):
        reverse = not reverse
    elif key == ord('c'):
        overlay = np.zeros_like(frame)
    elif key == 27:  # ESC key
        break
    elif key == ord('='):  # Use '[' as UP arrow key alternative
        thickness += 1
    elif key == ord('-'):  # Use ']' as DOWN arrow key alternative
        thickness = max(1, thickness - 1)
    elif key == ord('['):  # Use ',' as LEFT arrow key alternative
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - jump_frames)
    elif key == ord(']'):  # Use '.' as RIGHT arrow key alternative
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + jump_frames*3)
    elif key == ord('z'):  # Undo the last drawn lines since mouse was pressed
        if overlay_history:
            overlay = overlay_history.pop()


cap.release()
cv2.destroyAllWindows()
