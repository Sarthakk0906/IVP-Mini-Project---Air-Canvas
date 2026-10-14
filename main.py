import cv2
import mediapipe as mp
import numpy as np
import math
from datetime import datetime


WIDTH = 1280
HEIGHT = 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

colors = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "YELLOW": (0, 255, 255),
    "PURPLE": (255, 0, 255),
    "WHITE": (255, 255, 255)
}

color_names = list(colors.keys())
color_index = 0
draw_color = colors[color_names[color_index]]

brush_size = 8
eraser_size = 50

previous_point = None
last_gesture = ""

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

MODEL_PATH = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

hand_landmarker = HandLandmarker.create_from_options(options)


def get_point(landmark, width, height):
    """Convert normalized landmark coordinates to pixels."""
    x = int(landmark.x * width)
    y = int(landmark.y * height)

    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))

    return x, y


def fingers_up(landmarks):
    """
    Detect which fingers are raised.

    Returns:
    [thumb, index, middle, ring, pinky]
    """

    fingers = []

    # Thumb
    thumb_up = landmarks[4].x < landmarks[3].x
    fingers.append(1 if thumb_up else 0)

    # Index
    fingers.append(
        1 if landmarks[8].y < landmarks[6].y else 0
    )

    # Middle
    fingers.append(
        1 if landmarks[12].y < landmarks[10].y else 0
    )

    # Ring
    fingers.append(
        1 if landmarks[16].y < landmarks[14].y else 0
    )

    # Pinky
    fingers.append(
        1 if landmarks[20].y < landmarks[18].y else 0
    )

    return fingers


def distance(p1, p2):
    """Calculate distance between two points."""
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def save_drawing():
    """Save canvas as PNG."""
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = f"air_canvas_{timestamp}.png"

    cv2.imwrite(filename, canvas)

    print(f"Drawing saved: {filename}")


# ============================================================
# MAIN LOOP
# ============================================================

frame_timestamp = 0

while True:

    success, frame = cap.read()

    if not success:
        print("Camera could not be accessed.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # FRAME PROCESSING
    # Convert BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # --------------------------------------------------------
    # HAND DETECTION / OBJECT TRACKING
    # --------------------------------------------------------

    frame_timestamp += 33

    result = hand_landmarker.detect_for_video(
        mp_image,
        frame_timestamp
    )

    current_point = None
    gesture = "NO HAND"

    # --------------------------------------------------------
    # HAND FOUND
    # --------------------------------------------------------

    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        # Index fingertip = landmark 8
        current_point = get_point(
            landmarks[8],
            width,
            height
        )

        # Thumb fingertip
        thumb_point = get_point(
            landmarks[4],
            width,
            height
        )

        # Detect fingers
        fingers = fingers_up(landmarks)

        thumb = fingers[0]
        index = fingers[1]
        middle = fingers[2]
        ring = fingers[3]
        pinky = fingers[4]

        # ----------------------------------------------------
        # DRAWING
        # Index finger only
        # ----------------------------------------------------

        if (
            index == 1 and
            middle == 0 and
            ring == 0 and
            pinky == 0
        ):

            pinch_distance = distance(
                current_point,
                thumb_point
            )

            # ------------------------------------------------
            # ERASER
            # Thumb + index close together
            # ------------------------------------------------

            if pinch_distance < 45:

                gesture = "ERASER"

                if previous_point is not None:

                    cv2.line(
                        canvas,
                        previous_point,
                        current_point,
                        (0, 0, 0),
                        eraser_size
                    )

                previous_point = current_point

            # ------------------------------------------------
            # NORMAL DRAWING
            # ------------------------------------------------

            else:

                gesture = "DRAW"

                if previous_point is not None:

                    cv2.line(
                        canvas,
                        previous_point,
                        current_point,
                        draw_color,
                        brush_size
                    )

                previous_point = current_point

        # ----------------------------------------------------
        # CHANGE COLOR
        # Index + Middle
        # ----------------------------------------------------

        elif (
            index == 1 and
            middle == 1 and
            ring == 0 and
            pinky == 0
        ):

            gesture = "CHANGE COLOR"

            if last_gesture != "CHANGE COLOR":

                color_index = (
                    color_index + 1
                ) % len(color_names)

                draw_color = colors[
                    color_names[color_index]
                ]

                print(
                    "Color:",
                    color_names[color_index]
                )

            previous_point = None

        # ----------------------------------------------------
        # CLEAR CANVAS
        # Four fingers
        # ----------------------------------------------------

        elif (
            index == 1 and
            middle == 1 and
            ring == 1 and
            pinky == 1
        ):

            gesture = "CLEAR"

            if last_gesture != "CLEAR":

                canvas[:] = 0
                previous_point = None

                print("Canvas cleared.")

        # ----------------------------------------------------
        # PAUSE
        # ----------------------------------------------------

        else:

            gesture = "PAUSE"
            previous_point = None

        # ----------------------------------------------------
        # DRAW HAND LANDMARKS
        # ----------------------------------------------------

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

        for start, end in connections:

            p1 = get_point(
                landmarks[start],
                width,
                height
            )

            p2 = get_point(
                landmarks[end],
                width,
                height
            )

            cv2.line(
                frame,
                p1,
                p2,
                (0, 255, 0),
                2
            )

        # Draw landmarks
        for landmark in landmarks:

            point = get_point(
                landmark,
                width,
                height
            )

            cv2.circle(
                frame,
                point,
                4,
                (0, 0, 255),
                -1
            )

    else:

        previous_point = None

    last_gesture = gesture

    # ========================================================
    # MASKING
    # ========================================================

    gray_canvas = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2GRAY
    )

    _, mask = cv2.threshold(
        gray_canvas,
        1,
        255,
        cv2.THRESH_BINARY
    )

    inverse_mask = cv2.bitwise_not(mask)

    # ========================================================
    # IMAGE OVERLAY
    # ========================================================

    background = cv2.bitwise_and(
        frame,
        frame,
        mask=inverse_mask
    )

    foreground = cv2.bitwise_and(
        canvas,
        canvas,
        mask=mask
    )

    output = cv2.add(
        background,
        foreground
    )

    # ========================================================
    # USER INTERFACE
    # ========================================================

    cv2.rectangle(
        output,
        (0, 0),
        (WIDTH, 100),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        output,
        "AIR CANVAS",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Color: {color_names[color_index]}",
        (25, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        draw_color,
        2
    )

    cv2.putText(
        output,
        f"Gesture: {gesture}",
        (850, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "Index = Draw | Pinch = Erase | 2 Fingers = Color | 4 Fingers = Clear",
        (390, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1
    )

    cv2.putText(
        output,
        "S = Save | C = Clear | N = Next Color | +/- = Brush | Q = Quit",
        (25, HEIGHT - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Air Canvas - IVP Project",
        output
    )

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Save
    elif key == ord("s"):
        save_drawing()

    # Clear
    elif key == ord("c"):
        canvas[:] = 0
        previous_point = None
        print("Canvas cleared.")

    # Next color
    elif key == ord("n"):

        color_index = (
            color_index + 1
        ) % len(color_names)

        draw_color = colors[
            color_names[color_index]
        ]

    # Increase brush
    elif key in (ord("+"), ord("=")):

        brush_size = min(
            brush_size + 2,
            40
        )

    # Decrease brush
    elif key in (ord("-"), ord("_")):

        brush_size = max(
            brush_size - 2,
            2
        )


# ============================================================
# CLEANUP
# ============================================================

hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()