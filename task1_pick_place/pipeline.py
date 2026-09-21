"""
Task 1 pipeline:
Person + pose + backpack pick/place detection.

Pipeline:

1. Detect person and backpack using YOLO.
2. Track the person using ByteTrack.
3. Detect wrist keypoints using YOLO pose.
4. Create an ROI around the initial backpack position.
5. Generate cleaned signals:
       wrist_in_roi
       backpack_in_roi
       person_visible
6. Feed the signals into the debounced state machine.

State flow:

PLACED
    ↓
PICKING
    ↓
PICKED
    ↓
PERSON_LEAVES
    ↓
PLACING
    ↓
PLACED
"""

import argparse
import json
from collections import deque

import cv2
from ultralytics import YOLO

from roi_utils import ROI, center_distance
from state_machine import PickPlaceStateMachine, Signals


# COCO class IDs
COCO_PERSON_CLASS = 0
COCO_BACKPACK_CLASS = 24

# COCO pose keypoint IDs
LEFT_WRIST_KP = 9
RIGHT_WRIST_KP = 10

# Number of frames for temporary detection loss
MISS_TOLERANCE = 8

# Minimum pose keypoint confidence
WRIST_CONF_THRESHOLD = 0.35


class LastSeen:
    """
    Stores the last known bounding box and how many consecutive
    frames the object has been missing.
    """

    def __init__(self):
        self.bbox = None
        self.missing_frames = 0

    def update(self, bbox):
        if bbox is not None:
            self.bbox = bbox
            self.missing_frames = 0
        else:
            self.missing_frames += 1

    def usable_bbox(self):
        if self.bbox is None:
            return None

        if self.missing_frames > MISS_TOLERANCE:
            return None

        return self.bbox


def get_class_boxes(boxes, class_id):
    """
    Return all YOLO boxes belonging to the requested class.
    """

    return [
        box for box in boxes
        if int(box.cls[0]) == class_id
    ]


def get_highest_confidence_box(boxes, class_id):
    """
    Select the highest-confidence detection for a class.
    """

    candidates = get_class_boxes(boxes, class_id)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda box: float(box.conf[0])
    )


def get_tracked_person(boxes, previous_id=None):
    """
    Select the person to follow.

    If a ByteTrack ID is already known, keep using that ID.
    This provides consistent tracking while the person remains visible.

    If no previous ID exists, choose the highest-confidence person.
    """

    candidates = get_class_boxes(
        boxes,
        COCO_PERSON_CLASS
    )

    if not candidates:
        return None, None

    # Continue following the existing ByteTrack ID.
    if previous_id is not None:

        for box in candidates:

            if box.id is not None:

                track_id = int(box.id[0])

                if track_id == previous_id:
                    return box.xyxy[0].tolist(), track_id

    # No known ID or old ID no longer exists.
    # Start a new track using the highest-confidence person.
    best = max(
        candidates,
        key=lambda box: float(box.conf[0])
    )

    if best.id is not None:
        track_id = int(best.id[0])
    else:
        track_id = None

    return best.xyxy[0].tolist(), track_id


def pick_backpack(boxes, reference_bbox=None):
    """
    Select the backpack.

    If there is no previous backpack position:
        choose the highest-confidence backpack.

    Otherwise:
        choose the backpack closest to the previous position.
    """

    candidates = get_class_boxes(
        boxes,
        COCO_BACKPACK_CLASS
    )

    if not candidates:
        return None

    if reference_bbox is None:

        best = max(
            candidates,
            key=lambda box: float(box.conf[0])
        )

    else:

        best = min(
            candidates,
            key=lambda box: center_distance(
                box.xyxy[0].tolist(),
                reference_bbox
            )
        )

    return best.xyxy[0].tolist()


def get_wrist_point(pose_result, person_bbox):
    """
    Find the pose detection corresponding to the tracked person
    and return a confident wrist point.

    Either the left or right wrist can be used.
    """

    if pose_result.keypoints is None:
        return None

    if len(pose_result.keypoints) == 0:
        return None

    if pose_result.boxes is None:
        return None

    if len(pose_result.boxes) == 0:
        return None

    # Find the pose whose bounding-box center is closest
    # to the tracked person's bounding box.
    best_idx = None
    best_distance = float("inf")

    pose_boxes = pose_result.boxes.xyxy.tolist()

    for i, pose_bbox in enumerate(pose_boxes):

        distance = center_distance(
            pose_bbox,
            person_bbox
        )

        if distance < best_distance:
            best_distance = distance
            best_idx = i

    if best_idx is None:
        return None

    keypoints = pose_result.keypoints.xy[best_idx]

    # Keypoint confidence may be available.
    keypoint_conf = None

    if pose_result.keypoints.conf is not None:
        keypoint_conf = pose_result.keypoints.conf[best_idx]

    # Check left and right wrist.
    wrist_indices = [
        LEFT_WRIST_KP,
        RIGHT_WRIST_KP
    ]

    for wrist_index in wrist_indices:

        wrist = keypoints[wrist_index]

        x = float(wrist[0])
        y = float(wrist[1])

        # Invalid coordinate.
        if x <= 0 and y <= 0:
            continue

        # If confidence information is available,
        # reject low-confidence wrist detections.
        if keypoint_conf is not None:

            confidence = float(
                keypoint_conf[wrist_index]
            )

            if confidence < WRIST_CONF_THRESHOLD:
                continue

        return (x, y)

    return None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="Input video path"
    )

    parser.add_argument(
        "--output",
        default="output_annotated.mp4",
        help="Output annotated video"
    )

    parser.add_argument(
        "--log",
        default="events.json",
        help="JSON event log"
    )

    parser.add_argument(
        "--confirm-frames",
        type=int,
        default=6,
        help="Consecutive frames needed for normal state transition"
    )

    parser.add_argument(
        "--leave-frames",
        type=int,
        default=20,
        help="Consecutive missing-person frames for PERSON_LEAVES"
    )

    parser.add_argument(
        "--roi-padding",
        type=float,
        default=0.35,
        help="ROI padding around initial backpack"
    )

    parser.add_argument(
        "--det-model",
        default="yolov8n.pt",
        help="YOLO detection model"
    )

    parser.add_argument(
        "--pose-model",
        default="yolov8n-pose.pt",
        help="YOLO pose model"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load models
    # ---------------------------------------------------------

    det_model = YOLO(args.det_model)
    pose_model = YOLO(args.pose_model)

    # ---------------------------------------------------------
    # Open video
    # ---------------------------------------------------------

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.source}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # mp4v is generally more compatible with OpenCV.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        args.output,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video: {args.output}"
        )

    # ---------------------------------------------------------
    # Tracking / state variables
    # ---------------------------------------------------------

    roi = None

    backpack_last = LastSeen()
    wrist_last = LastSeen()

    # Person visibility smoothing.
    person_visible_history = deque(maxlen=3)

    # Current ByteTrack ID.
    person_track_id = None

    machine = PickPlaceStateMachine(
        confirm_frames=args.confirm_frames,
        leave_frames=args.leave_frames
    )

    frame_idx = 0

    # ---------------------------------------------------------
    # Main video loop
    # ---------------------------------------------------------

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        # -----------------------------------------------------
        # PERSON + BACKPACK DETECTION + TRACKING
        # -----------------------------------------------------

        det_result = det_model.track(
            frame,
            persist=True,
            classes=[
                COCO_PERSON_CLASS,
                COCO_BACKPACK_CLASS
            ],
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        boxes = det_result.boxes

        # -----------------------------------------------------
        # BACKPACK
        # -----------------------------------------------------

        backpack_bbox = pick_backpack(
            boxes,
            backpack_last.bbox
        )

        backpack_last.update(
            backpack_bbox
        )

        # Create ROI only once from the initial backpack detection.
        if roi is None and backpack_bbox is not None:

            roi = ROI.from_bbox(
                backpack_bbox,
                padding_ratio=args.roi_padding
            )

        # -----------------------------------------------------
        # PERSON TRACKING
        # -----------------------------------------------------

        person_bbox, current_person_id = get_tracked_person(
            boxes,
            person_track_id
        )

        if person_bbox is not None:

            person_track_id = current_person_id

        person_visible_history.append(
            person_bbox is not None
        )

        # A short detection dropout does not immediately mean
        # that the person has left.
        person_visible = any(
            person_visible_history
        )

        # -----------------------------------------------------
        # POSE / WRIST
        # -----------------------------------------------------

        wrist_point = None

        if person_bbox is not None:

            pose_result = pose_model.predict(
                frame,
                verbose=False
            )[0]

            wrist_point = get_wrist_point(
                pose_result,
                person_bbox
            )

        # If the person is currently not detected,
        # don't reuse an old wrist for a new interaction.
        if person_bbox is None:

            wrist_last.update(None)

        else:

            wrist_last.update(
                wrist_point
            )

        # -----------------------------------------------------
        # GENERATE STATE-MACHINE SIGNALS
        # -----------------------------------------------------

        wrist_in_roi = False
        backpack_in_roi = False

        if roi is not None:

            # Wrist
            wp = wrist_last.usable_bbox()

            if wp is not None:

                wrist_in_roi = roi.contains_point(
                    wp[0],
                    wp[1]
                )

            # Backpack
            bb = backpack_last.usable_bbox()

            if bb is not None:

                backpack_in_roi = roi.contains_center(
                    bb
                )

        # -----------------------------------------------------
        # STATE MACHINE
        # -----------------------------------------------------

        state = machine.step(
            Signals(
                wrist_in_roi=wrist_in_roi,
                backpack_in_roi=backpack_in_roi,
                person_visible=person_visible,
                frame_idx=frame_idx
            )
        )

        # -----------------------------------------------------
        # DRAW ROI
        # -----------------------------------------------------

        if roi is not None:

            x1, y1, x2, y2 = roi.as_int_tuple()

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "ROI",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        # -----------------------------------------------------
        # DRAW PERSON
        # -----------------------------------------------------

        if person_bbox is not None:

            x1, y1, x2, y2 = map(
                int,
                person_bbox
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            if person_track_id is not None:

                cv2.putText(
                    frame,
                    f"Person ID: {person_track_id}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

        # -----------------------------------------------------
        # DRAW BACKPACK
        # -----------------------------------------------------

        if backpack_last.bbox is not None:

            x1, y1, x2, y2 = map(
                int,
                backpack_last.bbox
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 128, 255),
                2
            )

            cv2.putText(
                frame,
                "Backpack",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 128, 255),
                2
            )

        # -----------------------------------------------------
        # DRAW WRIST
        # -----------------------------------------------------

        if wrist_last.bbox is not None:

            wx = int(wrist_last.bbox[0])
            wy = int(wrist_last.bbox[1])

            cv2.circle(
                frame,
                (wx, wy),
                6,
                (0, 0, 255),
                -1
            )

        # -----------------------------------------------------
        # STATE OVERLAY
        # -----------------------------------------------------

        cv2.putText(
            frame,
            f"STATE: {state.value}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        writer.write(frame)

        frame_idx += 1

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    cap.release()
    writer.release()

    # ---------------------------------------------------------
    # Save events
    # ---------------------------------------------------------

    events_out = []

    for event in machine.events:

        events_out.append({
            "frame": event.frame_idx,
            "time_sec": round(
                event.frame_idx / fps,
                2
            ),
            "from": event.from_state.value,
            "to": event.to_state.value
        })

    with open(
        args.log,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "fps": fps,
                "total_frames": frame_idx,
                "events": events_out
            },
            f,
            indent=2
        )

    print(
        f"Done. Wrote {args.output} and {args.log}"
    )

    print(
        f"Final state: {machine.state.value}, "
        f"{len(machine.events)} transitions"
    )


if __name__ == "__main__":
    main()

#python pipeline.py --source videos\backpack4.mp4 --output outputnew_annotated.mp4 --log events.json