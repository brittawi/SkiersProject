import cv2
import imageio
import subprocess
import os
import uuid

# function to extract frame from video
def extract_frame(video_path, frame_idx):
    """Extracts and returns a specific frame from a video file."""
    # minus 1 because frames start at 0 but image id starts at 1
    #print("extracting frame: ", frame_idx)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        #return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    else:
        print(f"Frame {frame_idx} could not be extracted from this path: {video_path}")
        return None
    
def extract_frame_second(video_path, frame_idx):
    """Extracts a specific frame by reading the video sequentially."""
    #print("extracting frame: ", frame_idx)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return None

    current_frame = 0
    while current_frame <= frame_idx:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Frame {frame_idx} could not be read.")
            cap.release()
            return None
        current_frame += 1

    cap.release()
    return frame  # Returns the frame (BGR format)

def extract_frame_imageio(video_path, frame_idx):
    """Extracts a frame using imageio (alternative to OpenCV)."""
    vid = imageio.get_reader(video_path, "ffmpeg")
    
    try:
        frame = vid.get_data(frame_idx)  # Get exact frame
        return frame
    except IndexError:
        print(f"Error: Frame {frame_idx} out of range.")
        return None
    
def extract_frame_ffmpeg(video_path, frame_idx):
    """Extracts a specific frame using ffmpeg and loads it with OpenCV."""
    
    output_file = "temp_frame.jpg"
    
    # FFmpeg command to extract a single frame
    command = [
        "ffmpeg", "-i", video_path,  # Input video
        "-vf", f"select=eq(n\,{frame_idx})",  # Select frame by index
        "-vsync", "vfr", output_file,  # Avoid duplicated frames
        "-y"  # Overwrite existing file
    ]
    
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Read the image using OpenCV
    frame = cv2.imread(output_file)
    
    return frame if frame is not None else None

def get_image_by_id(images, target_id):
    return next((image for image in images if image['id'] == target_id), None)

def reencode_to_all_keyframes_temp(input_path, temp_dir="./temp_videos"):
    """Re-encodes a video to intra-frame (all keyframes) format and returns temp file path."""
    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = f"{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(temp_dir, temp_filename)

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-g", "1",
        "-an",  # No audio
        output_path
    ]

    # print(f"Re-encoding {input_path} → {output_path}")
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path