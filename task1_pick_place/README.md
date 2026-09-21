# Task 1 — Person with Pose & Backpack Pick/Place Detection

## 1. Project Overview

This project implements a computer vision pipeline to detect and track a **person and a backpack** in a video and identify **backpack pick and place events**.

The system uses object detection, person tracking, human pose/keypoint information, and a configurable Region of Interest (ROI) around the backpack's initial position.

The backpack initially starts in the **PLACED** state. The system then monitors the person's wrist and backpack position to determine whether the backpack is being picked up, carried away, brought back, and placed at its original location.

The pick-and-place process is handled using a **state machine**, so the system does not change states based on a single noisy frame.

---

## 2. Problem Statement

The objective is to build a computer vision system that can recognize the following sequence:

```text
PLACED
   ↓
PICKING
   ↓
PICKED
   ↓
PERSON LEAVES
   ↓
PERSON RETURNS WITH BACKPACK
   ↓
PLACING
   ↓
PLACED
```

The system should:

* Detect the person.
* Detect the backpack.
* Track the person while visible.
* Obtain wrist/keypoint information from the detected person.
* Define an ROI around the backpack's initial location.
* Detect when the person's wrist interacts with the backpack.
* Determine when the backpack is picked up and moved away.
* Detect when the person leaves the camera view.
* Detect when the person returns carrying the backpack.
* Detect when the backpack is placed back inside the ROI.
* Avoid false state changes caused by temporary detection or tracking fluctuations.

---

## 3. Key Features

* Person detection
* Backpack detection
* Human pose/keypoint detection
* Person tracking
* Wrist position tracking
* Configurable backpack ROI
* Pick/place event detection
* State-machine-based event handling
* Temporal consistency checks
* Visualization of detections and current state
* Video-based testing

---

## 4. Computer Vision Pipeline

The overall pipeline is:

```text
Input Video
     ↓
Object Detection
     ↓
Person + Backpack Detection
     ↓
Person Tracking
     ↓
Pose / Keypoint Detection
     ↓
Extract Wrist Position
     ↓
Backpack Position
     ↓
ROI Check
     ↓
Temporal Validation
     ↓
State Machine
     ↓
Pick / Place Event
     ↓
Annotated Output Video
```

---

## 5. Technologies Used

* Python
* OpenCV
* Ultralytics YOLO
* YOLO Pose / Keypoint Detection
* Object Detection
* Human Pose Estimation
* Object Tracking
* Region of Interest (ROI)
* State Machine

---

## 6. Dataset / Video

A short video was recorded using a camera in which a person interacts with a backpack positioned at a fixed initial location.

The video follows the required interaction sequence:

1. Backpack starts at a fixed location.
2. Person approaches the backpack.
3. Person interacts with the backpack.
4. Person picks up the backpack.
5. Person leaves the camera view with the backpack.
6. Person returns carrying the backpack.
7. Person places the backpack near its original location.
8. Person moves their wrist away from the backpack.
9. Final backpack state becomes `PLACED`.

---

## 7. Region of Interest (ROI)

A configurable rectangular ROI is created around the backpack's initial location.

The ROI is used to determine whether the backpack and the person's wrist are inside or outside the original backpack area.

Conceptually:

```text
+--------------------------------+
|                                |
|          ROI                   |
|       +-----------+            |
|       | Backpack  |            |
|       |  Initial  |            |
|       |  Location |            |
|       +-----------+            |
|                                |
+--------------------------------+
```

The ROI coordinates can be adjusted according to the position of the backpack in the input video.

The ROI is used for:

* Detecting the initial `PLACED` condition.
* Detecting wrist interaction during pickup.
* Detecting when the backpack moves away.
* Detecting when the backpack returns.
* Confirming the final placement.

---

## 8. State Management

The pick-and-place process is implemented as a sequence of meaningful states.

### State Flow

```text
PLACED
   |
   | Wrist enters ROI and interacts with backpack
   ↓
PICKING
   |
   | Wrist + backpack move outside ROI
   ↓
PICKED
   |
   | Person leaves camera view
   ↓
PERSON LEAVES
   |
   | Person returns with backpack
   ↓
PLACING
   |
   | Wrist leaves ROI while backpack remains inside
   ↓
PLACED
```

---

## 9. State Definitions

### 9.1 PLACED — Initial

This is the initial condition.

The backpack is located inside the configured ROI.

```text
Backpack → Inside ROI
State    → PLACED
```

---

### 9.2 PICKING

The person interacts with the backpack and the person's wrist enters the ROI.

```text
Wrist → Inside ROI
Backpack → Inside ROI
State → PICKING
```

The state indicates that the person is currently interacting with the backpack and beginning the pickup action.

---

### 9.3 PICKED

After the pickup interaction, the person's wrist and backpack move outside the original ROI.

```text
Wrist → Outside ROI
Backpack → Outside ROI
State → PICKED
```

This indicates that the backpack has been successfully picked up.

---

### 9.4 PERSON LEAVES

After picking up the backpack, the person leaves the camera's field of view.

The system monitors the person tracking/detection to determine when the person is no longer visible.

```text
Person → Not visible
Backpack → Being carried away
State → PERSON LEAVES
```

---

### 9.5 PLACING

The same person returns to the camera view carrying the backpack.

When the backpack and person's wrist enter/interact with the ROI again:

```text
Person → Visible
Backpack → Inside/entering ROI
Wrist → Inside ROI
State → PLACING
```

This indicates that the person is placing the backpack back at its original location.

---

### 9.6 PLACED — Final

After the backpack is inside the ROI, the person's wrist moves outside the ROI while the backpack remains inside.

```text
Backpack → Inside ROI
Wrist → Outside ROI
State → PLACED
```

This confirms that the backpack has been placed.

---

## 10. Temporal Consistency

The system does not immediately change the state because of a single frame.

This is important because object detection and pose estimation can temporarily fluctuate.

For example:

```text
Frame 1 → Wrist outside ROI
Frame 2 → Wrist inside ROI
Frame 3 → Wrist outside ROI
```

A single-frame approach could incorrectly identify this as a pickup interaction.

Therefore, the system considers consecutive frames / previous states before confirming an event.

This helps handle:

* Temporary wrist detection loss
* Temporary backpack detection loss
* Brief ROI entry
* Tracking fluctuations
* Partial backpack overlap with ROI
* Temporary person detection loss

---

## 11. Person Tracking

Person tracking is used to maintain the identity of the person across consecutive frames.

Tracking helps the system distinguish between:

* The person approaching the backpack.
* The same person leaving with the backpack.
* The same person returning with the backpack.

Tracking information is considered together with detection and state information rather than treating every frame independently.

---

## 12. Wrist / Keypoint Detection

Human pose estimation is used to obtain body keypoints from the detected person.

The wrist keypoints are particularly important because they provide information about the person's interaction with the backpack.

The system checks whether the wrist is:

```text
Inside ROI
     or
Outside ROI
```

The wrist position is then combined with the backpack position and current state to identify pickup and placement events.

---

## 13. Backpack Detection

The backpack is detected in each video frame using object detection.

The detected backpack bounding box provides its approximate location.

The system uses the backpack location together with the ROI to determine whether the backpack is:

* Inside the initial ROI
* Leaving the ROI
* Outside the ROI
* Returning to the ROI
* Finally placed inside the ROI

---

## 14. Event Detection Logic

The important transitions are:

### Pickup

```text
PLACED
   ↓
Person approaches
   ↓
Wrist enters ROI
   ↓
PICKING
   ↓
Backpack + wrist move outside ROI
   ↓
PICKED
```

### Person leaves

```text
PICKED
   ↓
Person leaves camera view
   ↓
PERSON LEAVES
```

### Placement

```text
PERSON LEAVES
   ↓
Person returns with backpack
   ↓
Backpack + wrist enter ROI
   ↓
PLACING
   ↓
Wrist leaves ROI
   ↓
Backpack remains inside ROI
   ↓
PLACED
```

---

## 15. Project Structure

```text
task1_pick_place/
│
├── main.py
├── requirements.txt
├── README.md
│
├── models/
│   └── ...
│
├── videos/
│   └── backpack1.mp4
│
├── runs/
│   └── ...
│
└── output/
    └── ...
```

### Important Files

| File / Folder      | Purpose                       |
| ------------------ | ----------------------------- |
| `main.py`          | Main computer vision pipeline |
| `requirements.txt` | Python dependencies           |
| `README.md`        | Project documentation         |
| `models/`          | Model files                   |
| `videos/`          | Input videos                  |
| `runs/`            | Detection/tracking outputs    |
| `output/`          | Final annotated video/results |

---

## 16. Installation

Clone/download the project and open the project directory.

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## 17. Running the Project

Place the input video inside the `videos` folder.

For example:

```text
videos/
└── backpack1.mp4
```

Run:

```bash
python main.py
```

The program processes the video frame by frame, performs detection and pose estimation, tracks the person, checks the ROI, updates the state machine, and generates the annotated output.

---

## 18. Output

The output video displays the computer vision results, including:

* Person bounding box
* Backpack bounding box
* Pose/keypoints
* Wrist position
* ROI
* Tracking information
* Current pick/place state

Example state display:

```text
State: PLACED
```

or

```text
State: PICKING
```

or

```text
State: PICKED
```

or

```text
State: PERSON LEAVES
```

or

```text
State: PLACING
```

---

## 19. Validation

The system is validated using the recorded interaction video.

The expected sequence is:

```text
Initial
    ↓
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

The output video is visually inspected to verify that:

* The person is detected.
* The backpack is detected.
* Person tracking remains consistent while visible.
* Wrist/keypoints are detected.
* The ROI correctly surrounds the initial backpack location.
* Pickup is detected after wrist interaction.
* Backpack movement outside the ROI is detected.
* Person leaving the camera view is recognized.
* The returning person carrying the backpack is detected.
* Placement is detected when the backpack returns to the ROI.
* The final `PLACED` state occurs only after the wrist leaves while the backpack remains inside the ROI.

---

## 20. Handling Detection Problems

Real-world detection can contain temporary errors.

The implementation considers cases such as:

### Temporary wrist loss

If the wrist is temporarily unavailable, the system avoids immediately changing the state.

### Temporary backpack loss

A temporary backpack detection failure does not automatically reset the complete state sequence.

### Wrist briefly entering ROI

A brief ROI entry is not treated as a confirmed pickup without the required interaction and subsequent movement.

### Backpack partially overlapping ROI

The ROI check is designed to tolerate the backpack being partially inside the ROI rather than relying only on a single pixel/location condition.

### Tracking ID changes

Temporary tracking fluctuations are considered when updating the state instead of allowing one tracking failure to incorrectly restart the sequence.

---

## 21. Key Design Decision

The main design decision in this project is using a **state machine instead of independent frame classification**.

A frame-by-frame approach could produce:

```text
PLACED
PICKING
PLACED
PICKED
PLACED
...
```

because detection results may fluctuate.

Instead, the system maintains the previous state and only allows meaningful transitions:

```text
PLACED → PICKING → PICKED
                    ↓
              PERSON LEAVES
                    ↓
                 PLACING
                    ↓
                 PLACED
```

This makes the event detection more stable and consistent with the actual pick-and-place sequence.

---

## 22. Conclusion

This project demonstrates a computer vision pipeline for recognizing a person's interaction with a backpack using **object detection, person tracking, human pose/keypoint information, ROI-based spatial reasoning, and temporal state management**.

Rather than identifying events from individual frames, the system uses the current detections together with the previous state to recognize the complete pick-and-place sequence.

The final system can identify:

```text
PLACED
→ PICKING
→ PICKED
→ PERSON LEAVES
→ PLACING
→ PLACED
```

while considering practical detection and tracking fluctuations in a real video environment.
                                                                                                                                                                                                                                                                    