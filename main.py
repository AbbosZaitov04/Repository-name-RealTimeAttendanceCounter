import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "yolo11s.pt"       # <-- Change this to your YOLO model
SOURCE = 0                  # 0 = webcam, or "video.mp4"

CONFIDENCE = 0.35
IOU = 0.5

# How strongly to smooth bounding boxes
# 0.0 = no smoothing
# 1.0 = completely frozen
SMOOTHING = 0.65

# Number of frames a person can temporarily disappear
# before we remove them from the display.
MAX_MISSING_FRAMES = 10

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)

# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    print("ERROR: Could not open camera/video.")
    exit()

# Try to set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ============================================================
# TRACKING DATA
# ============================================================

# Stores smoothed bounding boxes
smooth_boxes = {}

# Stores how many frames since each person was detected
missing_frames = defaultdict(int)

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended or camera frame could not be read.")
        break

    h, w = frame.shape[:2]

    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        classes=[0],       # COCO class 0 = person
        conf=CONFIDENCE,
        iou=IOU,
        verbose=False
    )

    current_ids = set()

    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    if results[0].boxes is not None:

        boxes = results[0].boxes

        if boxes.id is not None:

            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()

            for track_id, box, conf in zip(ids, xyxy, confidences):

                current_ids.add(track_id)

                x1, y1, x2, y2 = box

                # Convert to float for smoothing
                new_box = np.array(
                    [x1, y1, x2, y2],
                    dtype=np.float32
                )

                # ------------------------------------------------
                # SMOOTH BOUNDING BOX
                # ------------------------------------------------

                if track_id not in smooth_boxes:

                    smooth_boxes[track_id] = new_box

                else:

                    old_box = smooth_boxes[track_id]

                    smooth_boxes[track_id] = (
                        SMOOTHING * old_box
                        + (1 - SMOOTHING) * new_box
                    )

                # Reset missing counter
                missing_frames[track_id] = 0

    # --------------------------------------------------------
    # HANDLE TEMPORARILY LOST PEOPLE
    # --------------------------------------------------------

    for track_id in list(smooth_boxes.keys()):

        if track_id not in current_ids:

            missing_frames[track_id] += 1

            # Remove after several missed frames
            if missing_frames[track_id] > MAX_MISSING_FRAMES:

                del smooth_boxes[track_id]
                del missing_frames[track_id]

    # ========================================================
    # DRAW PEOPLE
    # ========================================================

    for track_id, box in smooth_boxes.items():

        x1, y1, x2, y2 = box.astype(int)

        # Make sure coordinates stay inside frame
        h, w = frame.shape[:2]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # ----------------------------------------------------
        # BOUNDING BOX
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # "STUDENT" LABEL
        # ----------------------------------------------------

        label = "student"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        # Put label at top-right of bounding box
        label_x = x2 - text_w

        # If it would go outside the left edge
        if label_x < x1:
            label_x = x1

        label_y = y1 - 8

        # If box is near the top of the screen,
        # put label inside the box
        if label_y - text_h < 0:
            label_y = y1 + text_h + 8

        # Background rectangle for label
        cv2.rectangle(
            frame,
            (
                label_x - 4,
                label_y - text_h - 4
            ),
            (
                label_x + text_w + 4,
                label_y + baseline + 4
            ),
            (0, 255, 0),
            -1
        )

        # Text
        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    # ========================================================
    # TOTAL NUMBER OF PEOPLE
    # ========================================================

    person_count = len(smooth_boxes)

    count_text = f"Students: {person_count}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 3

    (text_w, text_h), baseline = cv2.getTextSize(
        count_text,
        font,
        font_scale,
        thickness
    )

    # Top-right corner
    margin = 20

    count_x = w - text_w - margin
    count_y = margin + text_h

    # Background box
    cv2.rectangle(
        frame,
        (
            count_x - 12,
            count_y - text_h - 12
        ),
        (
            count_x + text_w + 12,
            count_y + baseline + 12
        ),
        (0, 0, 0),
        -1
    )

    # Count text
    cv2.putText(
        frame,
        count_text,
        (count_x, count_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow("YOLO Student Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()