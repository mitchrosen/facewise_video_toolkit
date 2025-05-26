import cv2
import numpy as np
import os

# Parameters
output_path = "tests/data/short_video.mp4"
fps = 30
duration_sec = 3
frame_width = 640
frame_height = 360
total_frames = fps * duration_sec

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

for i in range(total_frames):
    # Scene 1: red background, Scene 2: green background
    color = (0, 0, 255) if i < total_frames // 2 else (0, 255, 0)
    frame = np.full((frame_height, frame_width, 3), color, dtype=np.uint8)

    # Draw simulated face (white circle) at scene center
    face_center = (int(frame_width / 2), int(frame_height / 2))
    cv2.circle(frame, face_center, 40, (255, 255, 255), -1)

    out.write(frame)

out.release()
print(f"✅ Test video saved to: {output_path}")
