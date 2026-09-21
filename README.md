# Person with Pose & Backpack Pick/Place Detection

## Overview

This project detects a person and a backpack in a video and identifies the backpack **pick and place** sequence.

The system uses **YOLO object detection, person tracking, pose estimation, ROI checking, and a state machine** to understand the interaction between the person and the backpack.

The expected sequence is:

```text
PLACED → PICKING → PICKED → PERSON LEAVES → PLACING → PLACED
```

---

## Pipeline

```text
Input Video
     ↓
Person + Backpack Detection
     ↓
Person Tracking + Pose Detection
     ↓
Wrist & Backpack Position
     ↓
ROI + Temporal Checks
     ↓
State Machine
     ↓
Pick / Place Detection
     ↓
Annotated Output Video
```

---

## Technologies Used

* Python
* OpenCV
* Ultralytics YOLO
* YOLO Pose
* ByteTrack
* Object Detection
* Human Pose Estimation
* Region of Interest (ROI)
* State Machine

---

## How It Works

### 1. Person and Backpack Detection

YOLO is used to detect the **person** and **backpack** in each frame.

The backpack bounding box is used to determine its position relative to the initial location.

### 2. Person Tracking

ByteTrack is used to track the person while they are visible in the video.

### 3. Wrist Detection

YOLO Pose provides human body keypoints. The wrist position is used to identify when the person interacts with the backpack.

### 4. ROI

A rectangular ROI is created around the backpack's initial position.

```text
+-----------------------------+
|                             |
|          ROI                |
|       +---------+           |
|       | Backpack|           |
|       | Initial |           |
|       | Position|           |
|       +---------+           |
|                             |
+-----------------------------+
```

The ROI helps determine whether the backpack has moved away from or returned to its original position.

### 5. State Machine

The state machine controls the complete interaction:

```text
PLACED
   ↓
PICKING
   ↓
PICKED
   ↓
PERSON LEAVES
   ↓
PLACING
   ↓
PLACED
```

The system only allows meaningful state transitions instead of changing the state based on a single frame.

---

## State Description

| State             | Description                                                       |
| ----------------- | ----------------------------------------------------------------- |
| **PLACED**        | Backpack is at its initial location                               |
| **PICKING**       | Person is interacting with the backpack                           |
| **PICKED**        | Backpack has moved away from the initial ROI                      |
| **PERSON LEAVES** | Person leaves the camera view after picking up the backpack       |
| **PLACING**       | Person returns and places the backpack near its original position |
| **PLACED**        | Backpack is placed and the person moves away                      |

---

## Temporal Checks

Detection results can temporarily fluctuate between frames.

For example, a wrist or backpack may not be detected for a few frames. The system therefore uses consecutive-frame checks and tolerance values before confirming important state changes.

This helps reduce false transitions caused by:

* Temporary detection loss
* Wrist detection fluctuations
* Backpack detection fluctuations
* Short ROI movements
* Tracking changes

---

## Project Structure

```text
task1_pick_place/
│
├── README.md
├── pipeline.py
├── state_machine.py
├── roi_utils.py
├── test_state_machine.py
├── requirements.txt
│
├── videos/
│   └── backpack1.mp4
│
└── outputnew_annotated.mp4
```

### Main Files

| File                    | Purpose                              |
| ----------------------- | ------------------------------------ |
| `pipeline.py`           | Main computer vision pipeline        |
| `state_machine.py`      | Handles pick/place state transitions |
| `roi_utils.py`          | ROI-related operations               |
| `test_state_machine.py` | State machine testing                |
| `requirements.txt`      | Required Python packages             |

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Make sure the required YOLO model files are available locally.

---

## Run the Project

Place the input video inside the `videos` folder.

```text
videos/
└── backpack1.mp4
```

Run:

```bash
python pipeline.py
```

The processed video is generated as an annotated output showing the detections, pose information, ROI, and current state.

---

## Output

The output video shows:

* Person detection
* Backpack detection
* Person tracking
* Pose/keypoints
* Wrist position
* Backpack ROI
* Current state

The final state sequence is:

```text
PLACED
   ↓
PICKING
   ↓
PICKED
   ↓
PERSON LEAVES
   ↓
PLACING
   ↓
PLACED
```

---

## Validation

The system was tested using a recorded video containing the complete backpack interaction.

The output was checked for:

* Person and backpack detection
* Person tracking
* Wrist detection
* Correct ROI placement
* Pickup detection
* Person leaving detection
* Backpack return
* Final placement detection

---

## Limitations

* The ROI depends on the initial backpack position.
* Camera movement can affect ROI-based detection.
* Heavy occlusion can affect person, backpack, or wrist detection.
* Detection quality depends on the input video and trained models.

---

## Conclusion

This project combines **object detection, tracking, pose estimation, ROI-based spatial reasoning, and a state machine** to detect a backpack pick-and-place interaction from video.

It demonstrates how multiple computer vision techniques can be combined with simple temporal logic to handle a real-world interaction more reliably than frame-by-frame detection alone.
