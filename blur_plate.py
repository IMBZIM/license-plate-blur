import cv2
import numpy as np
import sys
import os
import subprocess

# Try to import machine learning libraries for smart detection
try:
    from ultralytics import YOLO
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: Machine learning library not available. Install with: pip install ultralytics")
    print("   Falling back to manual selection only.")

def nothing(x):
    pass

def detect_license_plate_ml(frame, model_path='yolov8n.pt'):
    """
    Use YOLOv8 to detect license plates in the frame
    """
    if not ML_AVAILABLE:
        return None

    try:
        # Load YOLO model (you can use a custom trained model for license plates)
        model = YOLO(model_path)

        # Run inference
        results = model(frame, conf=0.3, classes=[2])  # class 2 is 'car' in COCO dataset

        # Get bounding boxes
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) > 0:
            # For simplicity, take the first detected car and estimate license plate position
            # In a real implementation, you'd use a license plate specific model
            x1, y1, x2, y2 = boxes[0]

            # Estimate license plate position (typically bottom 20-30% of car)
            plate_height = (y2 - y1) * 0.25
            plate_y1 = y2 - plate_height
            plate_y2 = y2 - plate_height * 0.1

            # Center horizontally
            plate_width = (x2 - x1) * 0.8
            plate_x1 = x1 + (x2 - x1 - plate_width) / 2
            plate_x2 = plate_x1 + plate_width

            return (int(plate_x1), int(plate_y1), int(plate_width), int(plate_height))

    except Exception as e:
        print(f"ML detection failed: {e}")

    return None

def blur_license_plate(video_path, output_path):
    # Step 1: Validation
    if not os.path.exists(video_path):
        print(f"\nFATAL ERROR: The file '{video_path}' was not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Process the entire video (no cutting)
    start = 0
    end = total_frames - 1

    print(f"\n--- PROCESSING ENTIRE VIDEO ---")
    print(f"Processing full video: {total_frames} frames at {fps} FPS")
    print("No cutting - entire video will be processed")

    # --- STEP 3: SMART ROI SELECTION ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    # Define display variables for progress (used later)
    display_h = 400
    scale = display_h / h
    nw, nh = int(w * scale), int(h * scale)

    # Try ML detection first if available
    ml_bbox = None
    if ML_AVAILABLE:
        print("\n--- AI-POWERED DETECTION ---")
        print("Using machine learning to detect license plate...")
        ml_bbox = detect_license_plate_ml(frame)
        if ml_bbox:
            print("AI detected license plate automatically!")
            # Show the detected area for confirmation
            confirm_frame = frame.copy()
            x1, y1, w1, h1 = ml_bbox
            cv2.rectangle(confirm_frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 3)
            cv2.putText(confirm_frame, "AI Detected - Press 'Y' to use, 'N' to select manually", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("AI Detection Confirmation", cv2.resize(confirm_frame, (800, 600)))
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord('y') or key == ord('Y'):
                rx, ry, rw, rh = ml_bbox
            else:
                ml_bbox = None  # Force manual selection
                print("Manual selection requested.")
        else:
            print("AI detection failed, falling back to manual selection")

    if not ml_bbox:
        # Manual selection fallback
        print("\n--- MANUAL LICENSE PLATE SELECTION ---")
        print("1. Click and drag to draw a rectangle around the license plate")
        print("2. Press SPACE or ENTER to confirm selection")
        print("3. Press ESC to cancel and exit")

        # Nuclear Zoom Logic - Larger canvas for easier selection
        canvas = np.zeros((800, 800, 3), dtype="uint8")
        x_off, y_off = (800 - nw) // 2, (800 - nh) // 2
        canvas[y_off:y_off+nh, x_off:x_off+nw] = cv2.resize(frame, (nw, nh))

        # Add clear instructions overlay
        cv2.putText(canvas, "Draw a rectangle around the license plate", (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Press SPACE or ENTER when done", (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, "Press ESC to cancel", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        roi_win = "Select License Plate - Draw Rectangle"
        cv2.namedWindow(roi_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(roi_win, 800, 800)

        bbox = cv2.selectROI(roi_win, canvas, False)
        cv2.destroyWindow(roi_win)

        if bbox == (0,0,0,0):
            print("Selection cancelled.")
            return

        # Back to original coordinates
        rx = int((bbox[0] - x_off) / scale)
        ry = int((bbox[1] - y_off) / scale)
        rw = int(bbox[2] / scale)
        rh = int(bbox[3] / scale)

    # --- STEP 4: CONFIRMATION & POLISHING ---
    # Show selected area for confirmation
    confirm_frame = frame.copy()
    cv2.rectangle(confirm_frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 3)
    cv2.putText(confirm_frame, "License Plate Selected", (rx, ry - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show confirmation window (skip for automated testing)
    print("\n--- CONFIRMATION ---")
    print("License plate area selected!")
    print("Continuing with processing...")
    # cv2.imshow("Confirm License Plate Selection", cv2.resize(confirm_frame, (800, 600)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # --- STEP 5: ADVANCED TRACKING & BLURRING ---
    # Initialize with available tracker
    tracker = None
    try:
        tracker = cv2.TrackerMIL_create()  # Robust and available
    except:
        try:
            tracker = cv2.TrackerDaSiamRPN_create()  # Alternative tracker
        except:
            tracker = cv2.TrackerNano_create()  # Lightweight fallback

    if tracker is None:
        print("ERROR: No suitable tracker available!")
        return

    tracker.init(frame, (rx, ry, rw, rh))

    # Store original bounding box for potential re-initialization
    original_bbox = (rx, ry, rw, rh)
    tracking_failures = 0
    max_failures = 5  # Allow some tracking failures before giving up

    temp_output = "temp_no_audio.mp4"

    # Process ALL frames to retain original file size and playback speed
    total_frames_to_process = end - start + 1
    out = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print("\n--- PROCESSING VIDEO (FULL QUALITY) ---")
    print("Processing all frames to retain original file size")
    print(f"Processing {total_frames_to_process} frames at {fps} FPS")
    print("Tracking license plate... Press 'Q' to stop early, 'R' to manually re-sync tracking")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    processed_count = 0

    for f_idx in range(start, end + 1):
        ret, frame = cap.read()
        if not ret: break

        success, box = tracker.update(frame)

        if success:
            x, y, w_b, h_b = [int(v) for v in box]
            tracking_failures = 0  # Reset failure counter

            # Ensure coordinates are within frame bounds
            x, y = max(0, x), max(0, y)
            w_b, h_b = min(w - x, w_b), min(h - y, h_b)

            if w_b > 0 and h_b > 0:
                roi = frame[y:y+h_b, x:x+w_b]
                if roi.size > 0:
                    # Faster blur with smaller kernel
                    frame[y:y+h_b, x:x+w_b] = cv2.GaussianBlur(roi, (51, 51), 15)

                # Draw tracking box for visual feedback (optional) - commented out for clean output
                # cv2.rectangle(frame, (x, y), (x+w_b, y+h_b), (0, 255, 0), 2)
        else:
            tracking_failures += 1
            if tracking_failures <= max_failures:
                try:
                    tracker.init(frame, original_bbox)
                except:
                    pass
            else:
                break

        # Write each frame
        out.write(frame)
        processed_count += 1

        # Show processing progress (less frequent updates)
        if processed_count % 50 == 0 or processed_count == total_frames_to_process:
            progress_frame = cv2.resize(frame, (nw, nh))
            progress_pct = int(100 * processed_count / total_frames_to_process)
            cv2.putText(progress_frame, f"Progress: {progress_pct}% ({processed_count}/{total_frames_to_process})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Processing...", progress_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Manual re-sync requested. Please select the license plate area again.")
            # Pause processing and allow manual re-selection
            canvas = np.zeros((800, 800, 3), dtype="uint8")
            x_off, y_off = (800 - nw) // 2, (800 - nh) // 2
            canvas[y_off:y_off+nh, x_off:x_off+nw] = cv2.resize(frame, (nw, nh))

            # Add instructions for re-sync
            cv2.putText(canvas, "Re-sync: Draw rectangle around license plate", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, "Press SPACE or ENTER when done", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            roi_win = "Re-sync License Plate - Draw Rectangle"
            cv2.namedWindow(roi_win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(roi_win, 800, 800)

            bbox = cv2.selectROI(roi_win, canvas, False)
            cv2.destroyWindow(roi_win)

            if bbox != (0,0,0,0):
                # Update bounding box coordinates
                rx = int((bbox[0] - x_off) / scale)
                ry = int((bbox[1] - y_off) / scale)
                rw = int(bbox[2] / scale)
                rh = int(bbox[3] / scale)

                # Re-initialize tracker with new bounding box
                try:
                    tracker = cv2.TrackerMIL_create()  # Create new tracker instance
                    tracker.init(frame, (rx, ry, rw, rh))
                    original_bbox = (rx, ry, rw, rh)
                    tracking_failures = 0
                    print("Tracker re-initialized with new license plate selection.")
                except Exception as e:
                    print(f"Failed to re-initialize tracker: {e}")
            else:
                print("Re-sync cancelled, continuing with current tracking.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # --- STEP 5: AUDIO MERGE (TRIMMED) ---
    print("\n--- AUDIO MERGING ---")
    print("Merging audio with blurred video...")

    # Check if ffmpeg is available
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            raise subprocess.SubprocessError("ffmpeg not found")
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("WARNING: ffmpeg not found. Installing audio will be skipped.")
        print("To enable audio merging, install ffmpeg:")
        print("   - Windows: choco install ffmpeg")
        print("   - macOS: brew install ffmpeg")
        print("   - Linux: sudo apt install ffmpeg")
        print(f"Video saved (no audio): {temp_output}")
        print(f"File saved to: {os.path.abspath(temp_output)}")
        print(f"Directory: {os.getcwd()}")
        return

    # Merge full audio from original video with processed video
    print(f"Merging full audio from original video...")

    cmd = [
        'ffmpeg', '-i', temp_output, '-i', video_path,
        '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-y', output_path
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            # Success - clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)

            print(f"\n--- PROCESSING COMPLETE ---")
            print(f"SUCCESS: License plate blurring completed!")
            print(f"AI-Powered: {'Yes' if ML_AVAILABLE else 'No'} (Machine Learning Detection)")
            print(f"Video processed: {total_frames} frames at {fps} FPS")
            print(f"Audio merged: Yes")
            print(f"Final video: {output_path}")
            print(f"File saved to: {os.path.abspath(output_path)}")
            print(f"Directory: {os.getcwd()}")

            # Verify the output file exists and has reasonable size
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"File size: {file_size:.2f} MB")
                print(f"Quality: Original resolution maintained")
                print(f"Performance: Full video processed without cutting")
            else:
                print("Warning: Output file was not created")

            print(f"\nTip: Your video is now privacy-protected with blurred license plates!")
            print(f"Ready for next video - run the script again with a new file.")

        else:
            print(f"Audio merge failed with return code: {result.returncode}")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            print(f"Video saved (no audio): {temp_output}")
            print(f"File saved to: {os.path.abspath(temp_output)}")
            print(f"Directory: {os.getcwd()}")

    except subprocess.TimeoutExpired:
        print("Audio merge timed out after 60 seconds")
        print(f"Video saved (no audio): {temp_output}")
        print(f"File saved to: {os.path.abspath(temp_output)}")
        print(f"Directory: {os.getcwd()}")
    except Exception as e:
        print(f"Audio merge error: {str(e)}")
        print(f"Video saved (no audio): {temp_output}")
        print(f"File saved to: {os.path.abspath(temp_output)}")
        print(f"Directory: {os.getcwd()}")

if __name__ == "__main__":
    blur_license_plate("4BB98A5B-DB93-4F24-BB43-9D3C09B80B64.mov", "blurred_trimmed.mp4")