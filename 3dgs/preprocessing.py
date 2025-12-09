import cv2
import os
from pathlib import Path


def extract_frames_from_video(video_path, output_dir, frame_rate=1):
    """
    Extract frames from a video file for COLMAP processing.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted frames
        frame_rate (int): Extract every nth frame (1 = every frame, 2 = every other frame, etc.)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Extracting every {frame_rate} frame(s)")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        success, frame = video.read()
        
        if not success:
            break
        
        # Save frame based on frame_rate
        if frame_count % frame_rate == 0:
            # Create filename with zero-padded frame number
            frame_filename = f"frame_{saved_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            # Save the frame
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    video.release()
    print(f"  Extracted {saved_count} frames to {output_dir}\n")


def process_data_folder(data_folder, frame_rate=1):
    """
    Process all MOV files in the data folder and extract frames.
    
    Args:
        data_folder (str): Path to the data folder containing MOV files
        frame_rate (int): Extract every nth frame
    """
    data_path = Path(data_folder)
    
    if not data_path.exists():
        print(f"Error: Data folder {data_folder} does not exist")
        return
    
    # Find all MOV files
    mov_files = list(data_path.glob("*.MOV")) + list(data_path.glob("*.mov"))
    
    if not mov_files:
        print(f"No MOV files found in {data_folder}")
        return
    
    print(f"Found {len(mov_files)} MOV file(s)\n")
    
    # Process each video file
    for video_path in mov_files:
        # Create output directory based on video name
        video_name = video_path.stem  # Get filename without extension
        output_dir = data_path / f"{video_name}_frames"
        
        extract_frames_from_video(str(video_path), str(output_dir), frame_rate)
    
    print("Frame extraction complete!")


if __name__ == "__main__":
    # Configuration
    DATA_FOLDER = "../data"  # Relative path from the 3dgs folder
    FRAME_RATE = 1  # Extract every 10th frame (adjust as needed for COLMAP)
    
    # Get absolute path
    script_dir = Path(__file__).parent
    data_folder_path = (script_dir / DATA_FOLDER).resolve()
    
    print("=" * 60)
    print("Frame Extraction for COLMAP")
    print("=" * 60)
    print(f"Data folder: {data_folder_path}")
    print(f"Frame rate: Every {FRAME_RATE} frame(s)")
    print("=" * 60 + "\n")
    
    # Process all MOV files
    process_data_folder(str(data_folder_path), frame_rate=FRAME_RATE)
