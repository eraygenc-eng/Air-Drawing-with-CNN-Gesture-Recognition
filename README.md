# Air Drawing with CNN Gesture Recognition

A real-time air drawing application built with **OpenCV**, **MediaPipe**, and a custom **PyTorch CNN model**.

This project allows users to draw on the screen using hand gestures captured from a webcam.  
The drawing system uses MediaPipe hand landmarks for stable real-time control, while a custom CNN model predicts hand gestures from cropped hand images.

---

## Project Overview

This project combines two different computer vision approaches:

1. **MediaPipe hand tracking**
   - Detects hand landmarks in real time.
   - Tracks finger positions.
   - Controls drawing, erasing, and canvas clearing actions.

2. **Custom CNN gesture classifier**
   - Trained on a custom webcam-based hand gesture dataset.
   - Predicts hand gestures from cropped hand images.
   - Displays real-time CNN predictions on the screen.

The main goal of this project is to build an interactive computer vision application while also integrating a custom deep learning model instead of relying only on prebuilt hand-tracking tools.

---

## Features

- Real-time webcam-based air drawing
- Hand landmark detection with MediaPipe
- Draw using the index finger
- Erase using the thumb-only gesture
- Clear the canvas using an open palm gesture
- Custom CNN model for hand gesture prediction
- Real-time CNN prediction display
- Modular Python project structure
- PyTorch-based training pipeline
- Custom dataset collection script

---

## Demo Controls

| Gesture / Key | Action |
|---|---|
| Index finger | Draw |
| Thumb only | Eraser |
| Open palm | Clear canvas |
| C key | Clear canvas manually |
| Q key | Quit application |

---

## Gesture Classes

The CNN model is trained to classify the following hand gestures:

| Class | Description |
|---|---|
| `idle` | Hand is visible but no drawing action is intended |
| `index_finger` | Index finger is raised for drawing |
| `open_palm` | Open hand gesture used to clear the canvas |
| `thumb_only` | Thumb-only gesture used as an eraser |

---

## How It Works

The application uses the webcam as input and processes each frame in real time.

### 1. Hand Tracking

MediaPipe detects the hand and returns 21 hand landmarks.

These landmarks are used to determine which fingers are open or closed.

For example:

- If only the index finger is open, drawing mode is activated.
- If only the thumb is open, eraser mode is activated.
- If the hand is fully open, the canvas is cleared.

### 2. Virtual Canvas

A separate canvas is created using NumPy.

Drawing is performed on this canvas instead of directly modifying the webcam frame.  
Then the canvas and the webcam frame are merged together.

This allows the drawing to stay on the screen even while the hand moves.

### 3. CNN Gesture Prediction

The hand region is cropped from the webcam frame using MediaPipe landmark coordinates.

The cropped hand image is resized to `64x64` and passed into the CNN model.

The CNN predicts one of the gesture classes:

```text
idle
index_finger
open_palm
thumb_only


```
### 4. DATASET

The CNN model was trained on approximately 3,000 custom hand gesture images collected from webcam frames.

The dataset was collected in a controlled environment using the same camera setup, similar lighting conditions, and the same background.

Because most images were captured under similar conditions, the model may sometimes make incorrect predictions when the lighting, background, camera angle, or hand position changes significantly.


## Installation and Running on Another Computer

Follow these steps to download and run the project on another computer.

### 1. Clone the repository

```bash
git clone https://github.com/eraygenc-eng/Air-Drawing-with-CNN-Gesture-Recognition.git
cd Air-Drawing-with-CNN-Gesture-Recognition
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python main.py
```

## Project Structure

```text
Air-Drawing-with-CNN-Gesture-Recognition/
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── hand_tracker.py
│   ├── gesture_rules.py
│   ├── canvas.py
│   └── utils.py
│
├── cnn/
│   ├── __init__.py
│   ├── classes.py
│   ├── model.py
│   ├── collect_data.py
│   ├── train.py
│   └── predict.py
│
├── data/
│   └── raw/
│       ├── idle/
│       ├── index_finger/
│       ├── open_palm/
│       └── thumb_only/
│
├── models/
│   └── hand_gesture_cnn.pth
│
└── screenshots/
       ├── drawing.png
       ├── eraser.png
       ├── open_palm.png
