import cv2
import json
import argparse
from pathlib import Path

def draw_face_boxes(frame, face_details, frame_width, frame_height, box_color=None):
    """
    Draws green rectangles on the frame for each face in the provided list.

    Args:
        frame (np.ndarray): The video frame to draw on.
        face_details (list): List of dicts with face bounding boxes (center_x, center_y, face_width, face_height).
        frame_width (int): Width in pixels of frame
        frame_height (int): Height in pixels of frame
    """
    for face in face_details:
        x_center = (face["center_x"] / 100.0) * frame_width
        y_center = (face["center_y"] / 100.0) * frame_height
        box_width = (face["face_width"] / 100.0) * frame_width
        box_height = (face["face_height"] / 100.0) * frame_height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        if not box_color:
            box_color = (0, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color=box_color, thickness=2)

def label_frame(frame, shot_number, frame_number, first_last_pos):
    """
    Overlays text indicating the shot and frame number onto the frame.

    Args:
        frame (np.ndarray): The video frame to label.
        shot_number (int): Shot index from metadata.
        frame_number (int): Frame number in the video.
        first_last_pos (int): 1 indicates first frame in shot; -1 indicates last frame in shot
    """
    frame_prefix = "First" if first_last_pos == 1 else "Last"
    label = f"Shot {shot_number}, {frame_prefix} shot frame {frame_number}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2, cv2.LINE_AA)

def visualize_shots_on_video(input_video_path, 
                             input_json_path, 
                             output_video_path,
                             box_color=None):
    """
    Reads a video and a corresponding JSON shot feature file, and writes a new video
    where face bounding boxes are drawn across all frames of each shot and labeled
    at the first and last frame of each shot.

    Args:
        input_video_path (str or Path): Path to the input video file.
        input_json_path (str or Path): Path to the input JSON file.
        output_video_path (str or Path): Where to write the annotated video.
        box_color (tuple): BGR spec for color of bounding box
    """
    input_video_path = Path(input_video_path)
    input_json_path = Path(input_json_path)
    output_video_path = Path(output_video_path)

    # Load shot features from JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)
        shots = data.get("shots", [])

    # Open the input video
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"❌ Failed to open video: {input_video_path}")

    # Extract video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

    # Build map of frame -> face boxes and label info
    frame_annotations = {}
    for shot in shots:
        shot_num = shot["shot_number"]
        detected_faces = shot.get("detected_faces", {})
        face_details = detected_faces.get("face_details", [])

        first = shot["first_frame"]
        last = shot["last_frame"]

        for i in range(first, last + 1):
            frame_annotations[i] = {
                "faces": face_details,
                "shot_num": shot_num,
                # first shot frame has 1 in first pos, last shot frame has -1 and all others have 0
                "first_last_pos": 1 if i == first else -1 if i == last else 0
            }

    # Iterate through video and annotate
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotation = frame_annotations.get(current_frame)
        if annotation:
            draw_face_boxes(frame, annotation["faces"], frame_width, frame_height, box_color)
            first_last_pos = annotation["first_last_pos"]
            if first_last_pos:
                shot_num = annotation["shot_num"]
                label_frame(frame, shot_num, current_frame, first_last_pos)

        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    print(f"✅ Output video saved to: {output_video_path}")

def parse_bgr(bgr_str):
    try:
        parts = [int(x) for x in bgr_str.split(',')]
        if len(parts) != 3 or not all(0 <= c <= 255 for c in parts):
            raise ValueError
        return tuple(parts)
    except ValueError:
        raise argparse.ArgumentTypeError("BGR must be three integers 0–255, e.g. '0,255,0' for green.")


def main():
    """
    CLI entry point. Parses arguments and calls visualization logic.
    """
    parser = argparse.ArgumentParser(description="Visualize shots and face boxes on video.")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("input_json", help="Path to JSON file with shot features")
    parser.add_argument("--output_video", help="Optional path to save annotated video")
    parser.add_argument(
        "--bgr",
        type=parse_bgr,
        default=None,
        help="Face box color in BGR format, e.g. '255,0,0' for blue"
    )
    args = parser.parse_args()

    input_video = args.input_video
    input_json = args.input_json
    output_video = args.output_video or str(Path(input_video).with_name(
        Path(input_video).stem + "_with_shots.mp4"))
    box_color = args.bgr

    visualize_shots_on_video(input_video, input_json, output_video, box_color)

if __name__ == "__main__":
    main()
