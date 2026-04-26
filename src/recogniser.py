# src/recogniser.py

"""
Live Instrument Gesture Recogniser
Runs the webcam, detects hands, predicts gestures in real time, and
displays colour-coded visual feedback on screen.
"""

import cv2
import sys
import os

# We need to import from the same src/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier


# ── Colour mapping ──────────────────────────────────────────────────────────────
# Each instrument gets its own overlay colour in BGR format (Blue, Green, Red).
# This gives instant visual feedback about which instrument is being detected.
INSTRUMENT_COLOURS = {
    "piano":  (255, 180,   0),   # Gold/amber
    "guitar": (0,   200,  50),   # Green
    "drums":  (0,    80, 255),   # Red-orange
}

# Maps each gesture name back to its parent instrument.
# Used to look up the correct colour without parsing the gesture string.
GESTURE_TO_INSTRUMENT = {
    "piano_ready":      "piano",
    "piano_press":      "piano",
    "piano_left":       "piano",
    "piano_right":      "piano",
    "guitar_chord":     "guitar",
    "guitar_strum":     "guitar",
    "guitar_pick":      "guitar",
    "guitar_mute":      "guitar",
    "drums_stick_grip": "drums",
    "drums_hit":        "drums",
    "drums_left":       "drums",
    "drums_right":      "drums",
}

# Minimum confidence required to display a prediction.
# Below this threshold we show "Uncertain" instead, to filter out noisy guesses.
CONFIDENCE_THRESHOLD = 0.60


def draw_prediction_overlay(frame, gesture: str, confidence: float, instrument: str):
    """
    Draws the prediction result directly onto the video frame.

    Args:
        frame:      The current BGR video frame (a numpy array from OpenCV)
        gesture:    The predicted gesture name, e.g. "guitar_chord"
        confidence: Prediction confidence as a float between 0.0 and 1.0
        instrument: The instrument the gesture belongs to, e.g. "guitar"

    Returns:
        frame: The frame with all overlays drawn on it
    """
    # Get the display colour for this instrument (default to white if unknown)
    colour = INSTRUMENT_COLOURS.get(instrument, (255, 255, 255))

    # ── Instrument label (top-left, large text) ──────────────────────────────
    cv2.putText(
        frame,
        instrument.upper(),       # e.g. "GUITAR"
        (10, 40),                 # Position: 10px from left, 40px from top
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,                      # Font scale (size)
        colour,
        3                         # Line thickness
    )

    # ── Gesture name (below the instrument label) ─────────────────────────────
    cv2.putText(
        frame,
        gesture.replace("_", " ").title(),  # e.g. "Guitar Chord" (readable format)
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        colour,
        2
    )

    # ── Confidence percentage ─────────────────────────────────────────────────
    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",  # e.g. "Confidence: 87.3%"
        (10, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 200, 200),   # Light grey — secondary info
        1
    )

    # ── Coloured border around the frame to indicate instrument ───────────────
    # cv2.rectangle draws a filled or outlined rectangle.
    # Here we draw a thin coloured border along the entire frame edge.
    h, w = frame.shape[:2]  # Get frame height and width
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), colour, 4)

    return frame


def extract_landmarks_as_list(hand_landmarks) -> list:
    """
    Converts a MediaPipe hand_landmarks object into a flat list of 63 floats.
    Format: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
    This matches exactly the format used in data_collector.py during training.

    Args:
        hand_landmarks: A MediaPipe NormalizedLandmarkList object

    Returns:
        A Python list of 63 floats
    """
    landmark_list = []
    for lm in hand_landmarks.landmark:
        # Each landmark has .x, .y (normalised 0–1 relative to frame size) and .z (depth)
        landmark_list.extend([lm.x, lm.y, lm.z])
    return landmark_list


def main():
    """
    Main loop:
      1. Load the trained model
      2. Open the webcam
      3. For each frame: detect hand → predict gesture → draw overlay → display
    """

    # ── Load the trained model ────────────────────────────────────────────────
    model_path = "models/gesture_model.joblib"

    classifier = GestureClassifier()
    try:
        classifier.load(model_path)
    except FileNotFoundError:
        print(f"ERROR: No trained model found at '{model_path}'.")
        print("Please run 'python src/gesture_classifier.py' first to train the model.")
        sys.exit(1)

    # ── Initialise the hand tracker ───────────────────────────────────────────
    tracker = HandTracker(
        static_image_mode=False,
        max_num_hands=1,           # Only track one hand at a time for cleaner predictions
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # ── Open the webcam ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    print("Starting live gesture recognition...")
    print("Press Esc to quit.")

    # ── Main frame loop ───────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read from webcam.")
            break

        # Flip the frame horizontally so it acts like a mirror.
        # This makes left/right gestures feel natural to the user.
        frame = cv2.flip(frame, 1)

        # ── Detect hand landmarks ─────────────────────────────────────────────
        results = tracker.process(frame)

        # ── Draw the MediaPipe landmark skeleton onto the frame ───────────────
        frame = tracker.draw(frame, results)

        # ── Predict gesture if a hand is detected ─────────────────────────────
        if results.multi_hand_landmarks:
            # Take the first detected hand (we only track one hand here)
            hand_landmarks = results.multi_hand_landmarks[0]

            # Convert the MediaPipe landmarks to a flat list of 63 floats
            landmarks = extract_landmarks_as_list(hand_landmarks)

            # Ask the classifier for a prediction and confidence score
            gesture, confidence = classifier.predict(landmarks)

            if confidence >= CONFIDENCE_THRESHOLD:
                # Map gesture → instrument for colour coding
                instrument = GESTURE_TO_INSTRUMENT.get(gesture, "unknown")
                # Draw the prediction text and coloured border on the frame
                frame = draw_prediction_overlay(frame, gesture, confidence, instrument)
            else:
                # Confidence is too low — display "Uncertain" to avoid misleading output
                cv2.putText(
                    frame,
                    "Uncertain...",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (128, 128, 128),   # Grey text for uncertain state
                    2
                )
        else:
            # No hand in frame — show a status message
            cv2.putText(
                frame,
                "No hand detected",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 200),   # Red text
                2
            )

        # ── Show the keyboard hint at the bottom ──────────────────────────────
        cv2.putText(
            frame,
            "Esc to quit",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        # ── Display the annotated frame ───────────────────────────────────────
        cv2.imshow("Hand Gesture Instrument Recogniser", frame)

        # ── Handle keyboard input ─────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc key
            print("Exiting...")
            break

    # ── Clean up ──────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("Recogniser stopped.")


if __name__ == "__main__":
    main()