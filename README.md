# Digital Rift Tech - AI License Plate Blurring Tool

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.0+-red.svg)

An intelligent video processing tool that automatically detects, tracks, and blurs license plates in high-resolution portrait videos while preserving the original audio quality.

## Project Features

- **High-Quality Gaussian Blurring** - Professional-grade privacy protection
- **Intelligent Motion Tracking** - Uses CSRT/KCF algorithms for accurate plate tracking
- **Zoom-Out Selection Mode** - Easy-to-use ROI targeting interface
- **Automatic Audio Merging** - Preserves original audio using FFmpeg
- **Optimized Performance** - Designed for high-resolution portrait video processing
- **Cross-Platform** - Works on Windows, macOS, and Linux

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
pip install opencv-contrib-python
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

1. Place your video file in the project directory
2. Update the filename in `blur_plate.py`:
```python
if __name__ == "__main__":
    blur_license_plate("your_video.mov", "output_blurred.mp4")
```

3. Run the script:
```bash
python blur_plate.py
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

- **Tracking Algorithm**: CSRT (Channel and Spatial Reliability Tracker) with KCF fallback
- **Blur Method**: Gaussian blur with configurable kernel size
- **Audio Processing**: FFmpeg AAC encoding with stream mapping
- **Output Format**: MP4 with H.264 video and AAC audio

## Limitations

- Requires manual selection of the initial license plate position
- Performance depends on video quality and lighting conditions
- May lose tracking with extreme motion or occlusion
- Best results with clear, well-lit footage

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
