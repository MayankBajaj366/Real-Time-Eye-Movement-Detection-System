# Real-Time Eye Direction Tracking System

A real-time eye tracking application built using Python, OpenCV, and MediaPipe Face Mesh.

## Features

- Real-time iris tracking
- Eye direction detection
  - Left
  - Right
  - Up
  - Down
  - Center
- Live X/Y movement graphs
- Video recording
- CSV data logging
- Noise smoothing for stable detection

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Eye-Tracking-System.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python eye_tracking.py
```

## Controls

| Key | Action |
|------|--------|
| R | Start/Stop Recording |
| Q | Quit |
| ESC | Quit |

## Output

- Recorded dashboard videos (.mp4)
- Eye movement CSV data
- Live movement graphs

## Future Improvements

- Blink detection
- Head pose estimation
- Cursor control
- Machine learning-based gaze estimation
