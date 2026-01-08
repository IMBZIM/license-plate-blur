import cv2
import sys
import os
import subprocess

def blur_license_plate(video_path, output_path):
    """
    Main function to blur a license plate in a video using manual selection and tracking.
    
    Args:
        video_path: Path to the input video file
        output_path: Path where the final processed video will be saved
    """
    
    # ============================================================================
    # STEP 1: VIDEO FILE VALIDATION
    # ============================================================================
    # Check if the input video file actually exists on disk
    if not os.path.exists(video_path):
        print(f"\nFATAL ERROR: The file '{video_path}' was not found.")
        return
    
    # ============================================================================
    # STEP 2: OPEN VIDEO AND READ FIRST FRAME
    # ============================================================================
    # Create a VideoCapture object to read the video
    cap = cv2.VideoCapture(video_path)
    
    # Verify that the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit()
    
    # Read the very first frame - this will be used for ROI selection
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video.")
        sys.exit()
    
    # ============================================================================
    # STEP 3: CREATE ZOOMED-OUT CANVAS FOR EASIER SELECTION
    # ============================================================================
    # Get the original frame dimensions
    h, w = frame.shape[:2]
    
    # Scale the frame down to 300px height for easier viewing
    display_h = 300
    scale = display_h / h
    nw, nh = int(w * scale), int(h * scale)  # New width and height after scaling
    
    # Create a large black canvas (700x700) to center the scaled frame
    canvas_size = 700
    canvas = cv2.zeros((canvas_size, canvas_size, 3), dtype="uint8")
    
    # Calculate offsets to center the scaled frame on the canvas
    x_off = (canvas_size - nw) // 2
    y_off = (canvas_size - nh) // 2
    
    # Resize the frame and place it on the canvas
    resized_frame = cv2.resize(frame, (nw, nh))
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized_frame
    
    # ============================================================================
    # STEP 4: USER SELECTS LICENSE PLATE REGION
    # ============================================================================
    # Display instructions to the user
    print("\n--- ACTION REQUIRED ---")
    print("1. Draw a box over the plate.")
    print("2. Press ENTER.")
    
    # Create a named window and set its size
    win_name = "Digital Rift - Select Plate"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, canvas_size, canvas_size)
    
    # Let the user draw a bounding box on the canvas
    bbox_canvas = cv2.selectROI(win_name, canvas, False)
    cv2.destroyWindow(win_name)
    
    # Check if user cancelled the selection (returns all zeros)
    if bbox_canvas == (0,0,0,0):
        print("Selection cancelled.")
        return
    
    # ============================================================================
    # STEP 5: CONVERT CANVAS COORDINATES BACK TO ORIGINAL VIDEO COORDINATES
    # ============================================================================
    # The bbox was drawn on the scaled/centered canvas, so we need to:
    # 1. Subtract the offset to get coordinates relative to the scaled frame
    # 2. Divide by scale to get coordinates in the original frame
    real_x = int((bbox_canvas[0] - x_off) / scale)
    real_y = int((bbox_canvas[1] - y_off) / scale)
    real_w = int(bbox_canvas[2] / scale)
    real_h = int(bbox_canvas[3] / scale)
    
    # Store the bounding box in original video coordinates
    bbox_original = (real_x, real_y, real_w, real_h)
    
    # ============================================================================
    # STEP 6: INITIALIZE OBJECT TRACKER
    # ============================================================================
    # Try to use CSRT tracker (more accurate but slower)
    # If not available, fall back to KCF tracker (faster but less accurate)
    try:
        tracker = cv2.TrackerCSRT_create()
    except:
        tracker = cv2.legacy.TrackerKCF_create()
    
    # Initialize the tracker with the first frame and the selected bounding box
    tracker.init(frame, bbox_original)
    
    # ============================================================================
    # STEP 7: SETUP VIDEO WRITER FOR OUTPUT
    # ============================================================================
    # Get the frames per second from the original video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create a temporary output file (without audio for now)
    temp_output = "temp_no_audio.mp4"
    
    # Initialize VideoWriter to save the processed frames
    # Uses mp4v codec, same fps as original, same dimensions as original
    out = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    # ============================================================================
    # STEP 8: PROCESS ALL FRAMES - TRACK AND BLUR LICENSE PLATE
    # ============================================================================
    print("\nProcessing frames...")
    
    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:  # No more frames, video is finished
            break
        
        # Update the tracker to get the new position of the license plate
        success, box = tracker.update(frame)
        
        if success:  # Tracker successfully found the object
            # Convert tracker coordinates to integers
            tx, ty, tw, th = [int(v) for v in box]
            
            # Ensure coordinates don't go negative (clamp to frame boundaries)
            ty, tx = max(0, ty), max(0, tx)
            
            # Extract the region of interest (the license plate area)
            roi = frame[ty:ty+th, tx:tx+tw]
            
            # Only blur if ROI is valid (not empty)
            if roi.size > 0:
                # Apply Gaussian blur with large kernel (99x99) and high sigma (30)
                # This creates a heavy blur effect to obscure the license plate
                frame[ty:ty+th, tx:tx+tw] = cv2.GaussianBlur(roi, (99, 99), 30)
        
        # Write the processed frame to the output video
        out.write(frame)
        
        # Display a preview of the processing (scaled down for viewing)
        cv2.imshow("Processing...", cv2.resize(frame, (nw, nh)))
        
        # Allow user to quit early by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # ============================================================================
    # STEP 9: CLEANUP VIDEO PROCESSING
    # ============================================================================
    # Release the video capture and writer objects
    cap.release()
    out.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    
    # ============================================================================
    # STEP 10: MERGE AUDIO FROM ORIGINAL VIDEO USING FFMPEG
    # ============================================================================
    print("\nAttempting to merge audio with FFmpeg...")
    
    # Build the FFmpeg command:
    # -i temp_output: Input the silent processed video
    # -i video_path: Input the original video (for audio)
    # -c:v copy: Copy video stream without re-encoding (faster)
    # -c:a aac: Encode audio as AAC
    # -map 0:v:0: Use video from first input (processed video)
    # -map 1:a:0: Use audio from second input (original video)
    # -shortest: End output when shortest input ends
    # -y: Overwrite output file if it exists
    cmd = [
        'ffmpeg', '-i', temp_output, '-i', video_path,
        '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
        '-shortest', '-y', output_path
    ]
    
    try:
        # Execute the FFmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:  # FFmpeg succeeded
            # Remove the temporary file (no longer needed)
            if os.path.exists(temp_output):
                os.remove(temp_output)
            print(f"SUCCESS! Video with audio saved: {output_path}")
        else:  # FFmpeg ran but encountered an error
            print("\n--- AUDIO MERGE FAILED ---")
            print("FFmpeg is installed but ran into an error.")
            print(f"Error details: {result.stderr}")
            print(f"Silent video kept as: {temp_output}")
            
    except FileNotFoundError:  # FFmpeg is not installed
        print("\n--- AUDIO MERGE FAILED ---")
        print("FFmpeg was NOT found on your system.")
        print("1. Download FFmpeg from https://ffmpeg.org/download.html")
        print("2. Add it to your System PATH.")
        print(f"3. Manual Fix: Run this command in your terminal later:")
        print(f"ffmpeg -i {temp_output} -i {video_path} -c:v copy -c:a aac -shortest {output_path}")


# ============================================================================
# PROGRAM ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Execute the blur function with specified input and output paths
    blur_license_plate("4BB98A5B-DB93-4F24-BB43-9D3C09B80B64.mov", "blurred_with_audio.mp4")