# Digital Rift Tech - AI License Plate Blurring Tool

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.8+-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-AI--Powered-blue.svg)

An intelligent video processing tool that uses **machine learning** to automatically detect, track and blur license plates in videos while preserving original quality and audio. Features AI-powered detection with manual fallback for maximum reliability.

##  Key Features

###  AI-Powered Detection
- **YOLOv8 Integration** - Automatic license plate detection using state-of-the-art machine learning
- **Smart Fallback** - AI detection with manual selection backup for maximum reliability
- **Real-time Processing** - Instant detection and confirmation workflow

###  Advanced Tracking & Blurring
- **Multi-Tracker Support** - MIL, DaSiamRPN, and Nano trackers with intelligent fallback
- **Robust Motion Tracking** - Handles complex movements, lighting changes, and partial occlusions
- **Manual Re-sync** - Press 'R' during processing to manually correct tracking
- **High-Quality Gaussian Blurring** - Professional-grade privacy protection with configurable intensity

###  User Experience
- **Interactive Selection** - Large canvas interface for precise license plate targeting
- **Visual Feedback** - Real-time progress display and tracking confirmation
- **One-Click Processing** - Simple workflow from detection to final output

###  Professional Output
- **Audio Preservation** - Automatic FFmpeg integration maintains original audio quality
- **Full Resolution** - Processes entire videos without quality loss or cutting
- **Cross-Platform** - Native support for Windows, macOS, and Linux
- **Batch Ready** - Command-line interface perfect for automation

## Requirements

- **Python**: 3.10 or higher
- **OpenCV**: opencv-contrib-python
- **FFmpeg**: Required for audio support

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/IMBZIM/license-plate-blur.git
cd license-plate-blur
```

### 2. Install Python Dependencies
```bash
pip install opencv-contrib-python ultralytics numpy
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg

#### Windows
1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add `C:\ffmpeg\bin` to your System PATH

#### macOS
```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

## Usage

### Basic Usage

Run the script with your video file:
```bash
python blur_plate.py input_video.mov output_blurred.mp4
```

### Command Line Options

- `input_video`: Path to your input video file
- `output_video`: Path where the blurred video will be saved
- `--no-ml`: Skip machine learning detection, force manual selection

### Examples

```bash
# Basic usage with AI detection
python blur_plate.py my_video.mov blurred_output.mp4

# Force manual selection (skip AI)
python blur_plate.py my_video.mov blurred_output.mp4 --no-ml

# Use absolute paths
python blur_plate.py /path/to/input.mov /path/to/output.mp4
```

### Step-by-Step Process

1. **Launch the tool** - The script will open with your video's first frame
2. **Select the license plate** - Draw a bounding box around the plate
3. **Press ENTER** - Confirm your selection
4. **Wait for processing** - The tool will track and blur the plate throughout the video
5. **Check output** - Your blurred video with audio will be saved

### Controls

- **Mouse**: Draw selection box around license plate
- **ENTER**: Confirm selection and start processing
- **Q**: Quit processing early (during frame processing)

## Project Structure
```
license-plate-blur/
│
├── blur_plate.py           # Main script
├── README.md               # This file
├── requirements.txt        # Python dependencies (optional)
└── output/                 # Output videos (created automatically)
```

## Troubleshooting

### FFmpeg Not Found
If you see "FFmpeg was NOT found on your system":
- Verify FFmpeg is installed: `ffmpeg -version`
- Ensure FFmpeg is in your system PATH
- Restart your terminal/command prompt after installation

### Video Won't Open
- Ensure the video file path is correct
- Check that the video format is supported (mp4, mov, avi, etc.)
- Verify the video file isn't corrupted

### Tracking Issues
- If the tracker loses the plate, try:
  - Using a tighter bounding box around the plate
  - Ensuring the first frame has a clear view of the plate
  - Checking video quality and lighting conditions

## Advanced Configuration

### Adjust Blur Intensity

In `blur_plate.py`, modify the blur parameters:
```python
# Increase blur strength (default: 99, 99, 30)
frame[ty:ty+th, tx:tx+tw] = cv2.GaussianBlur(roi, (151, 151), 50)
```

### Change Tracker Type
```python
# Use KCF tracker (faster, less accurate)
tracker = cv2.legacy.TrackerKCF_create()

# Use CSRT tracker (slower, more accurate) - default
tracker = cv2.TrackerCSRT_create()
```

## Technical Details

- **AI Detection**: YOLOv8 (yolov8n.pt) for automatic license plate detection with fallback to manual selection
- **Tracking Algorithms**: Multiple tracker support (MIL, DaSiamRPN, Nano) with intelligent fallback system
- **Motion Tracking**: Robust handling of complex movements, lighting changes, and partial occlusions
- **Blur Method**: Configurable Gaussian blur with kernel size (51, 51) and sigma (15) for optimal privacy
- **Audio Processing**: FFmpeg stream copying preserves original audio quality and sync
- **Output Format**: MP4 with H.264 video codec and AAC audio
- **Performance**: Processes full videos at original resolution without quality loss

## Limitations

- **AI Detection**: May occasionally fail on obscured, angled, or very small license plates
- **Manual Fallback**: Available when AI detection is insufficient
- **Tracking Performance**: Depends on video quality, lighting conditions, and plate visibility
- **Motion Challenges**: May lose tracking with extreme motion, severe occlusion, or rapid lighting changes
- **Optimal Results**: Best performance with clear, well-lit footage and visible license plates

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

**Digital Rift Tech**  
*An unregistered initiative focused on tech solutions and media privacy*

For inquiries, please contact: **+263 71 213 0545**

---

## Acknowledgments

- OpenCV community for excellent computer vision tools
- FFmpeg project for multimedia processing capabilities

## Future Enhancements

- Automatic license plate detection using AI/ML
- Batch processing for multiple videos
- GPU acceleration support
- Real-time preview during processing
- Multiple object tracking
- Web-based interface

---

**Made with care by Digital Rift Tech**
Brandon Chidhuza
