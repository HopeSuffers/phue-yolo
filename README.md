# YOLOv8 Object Detection with Philips Hue Integration

This project uses YOLOv8 for real-time person detection in a live camera feed, integrated with Philips Hue smart lights to dynamically control lighting

- Dynamically controls Philips Hue lights based on detected person's position

## Requirements

- **Hardware**: Webcam, Philips Hue&#x20;
- **Software**: Python 3.8+, `torch`, `numpy`, `opencv-python`, `ultralytics`, `supervision`, `phue`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HopeSuffers/phue-yolo.git
   cd your_repository
   ```
2. Install dependencies:
   ```bash
   pip install torch numpy opencv-python ultralytics supervision phue
   ```
3. Update Philips Hue Bridge IP in the script:
   ```python
   b = Bridge('Bridge IP')
   ```

## Usage

Run the script:

You can adjust the lights that are controlled by modifying their names or identifiers in the script's light control section (e.g., `Hue color candle 1`, `Hue color candle 2`)

```bash
python object_detection.py
```

The feed will show bounding boxes for persons only, and lights adjust dynamically based on position
