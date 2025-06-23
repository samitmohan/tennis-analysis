import cv2
def read_vid(vid_path):
    cap = cv2.VideoCapture(vid_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    return frames

def save_vid(output_vid_frames, output_vid_path):
    if not output_vid_frames:
        raise ValueError("No frames provided to save video")
    
    # Check frame consistency
    frame_height, frame_width = output_vid_frames[0].shape[:2]
    if any(frame.shape[:2] != (frame_height, frame_width) for frame in output_vid_frames):
        raise ValueError("Inconsistent frame sizes in video")
    
    # Use more reliable codec and allow specifying FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # More widely supported codec
    out = cv2.VideoWriter(output_vid_path, fourcc, 30, (frame_width, frame_height))
    
    try:
        for frame in output_vid_frames:
            out.write(frame)
    except Exception as e:
        print(f"Error writing video: {e}")
    finally:
        out.release()