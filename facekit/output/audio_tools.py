import subprocess
import os

def restore_audio_from_source(video_with_audio: str, video_no_audio: str) -> None:
    """
    Restores the original audio track from the source video and merges it into the processed (target) video.

    This function uses `ffmpeg` to extract the audio from the source and combine it with the target video, 
    preserving the original video stream of the target and the audio stream of the source.

    Args:
        video_with_audio (str): Path to the video file that contains the original audio.
        video_no_audio (str): Path to the processed video file (typically silent) that will receive the audio.

    Returns:
        None
    """
    temp_path = os.path.join(os.path.dirname(video_no_audio), "temp_output.mp4")
    cmd = [
        "ffmpeg",
        "-i", video_no_audio,
        "-i", video_with_audio,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-y",
        "-hide_banner",
        "-loglevel", "quiet",
        temp_path  # output filename must go last
    ]
    subprocess.run(cmd, check=True)
    os.replace(temp_path, video_no_audio)
