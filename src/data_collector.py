# src/data_collector.py

"""
Instrument-Specific Gesture Data Collector
Collects training data for gestures that directly relate to playing instruments.
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from hand_tracker import HandTracker


class InstrumentGestureCollector:
    """
    Collects hand landmark data for instrument-specific gestures.
    
    This focuses on gestures that directly map to instrument controls:
    - Piano: key press, hand position (left/right), hand height
    - Violin: bow movement (up/down), vibrato
    - Drums: hitting motion, hand position
    - Flute: finger positions, hand shape
    """
    
    # Define gestures organized by instrument
    INSTRUMENT_GESTURES = {
        "piano": {
            "piano_ready": {
                "description": "Open hand with relaxed, slightly curved fingers. Ready to play piano keys.",
                "tips": [
                    "Keep fingers curved naturally",
                    "Palm facing downward",
                    "Fingers should be spread",
                    "Relax your hand"
                ]
            },
            "piano_press": {
                "description": "Fingers pressed together, as if pressing down on piano keys.",
                "tips": [
                    "Curl fingers inward",
                    "Press fingertips together",
                    "Keep palm open",
                    "Simulate pressing a key"
                ]
            },
            "piano_left": {
                "description": "Hand positioned to the left (for playing lower notes).",
                "tips": [
                    "Move entire hand to the left side",
                    "Keep fingers in ready position",
                    "Hand should be in lower region"
                ]
            },
            "piano_right": {
                "description": "Hand positioned to the right (for playing higher notes).",
                "tips": [
                    "Move entire hand to the right side",
                    "Keep fingers in ready position",
                    "Hand should be in upper region"
                ]
            }
        },
        "violin": {
            "violin_bow_up": {
                "description": "Hand moving upward as if moving a violin bow upward.",
                "tips": [
                    "Make upward motion with hand",
                    "Keep fingers slightly curved",
                    "Move at shoulder level",
                    "Smooth, continuous motion"
                ]
            },
            "violin_bow_down": {
                "description": "Hand moving downward as if moving a violin bow downward.",
                "tips": [
                    "Make downward motion with hand",
                    "Keep fingers slightly curved",
                    "Move at shoulder level",
                    "Smooth, continuous motion"
                ]
            },
            "violin_position": {
                "description": "Hand in violin playing position (holding bow grip).",
                "tips": [
                    "Curl fingers as if holding bow",
                    "Thumb underneath",
                    "Fingers on top",
                    "Hand at shoulder height"
                ]
            },
            "violin_vibrato": {
                "description": "Hand shaking/vibrating motion (vibrato effect).",
                "tips": [
                    "Shake hand back and forth",
                    "Small, quick movements",
                    "Keep fingers curved",
                    "Wrist movement"
                ]
            }
        },
        "drums": {
            "drums_stick_grip": {
                "description": "Fist with thumb pointing (holding drumstick grip).",
                "tips": [
                    "Make a fist with your hand",
                    "Thumb pointing outward",
                    "Index finger slightly extended",
                    "Hold at shoulder height"
                ]
            },
            "drums_hit": {
                "description": "Hand making a hitting/striking motion.",
                "tips": [
                    "Start with hand open",
                    "Quickly close to fist",
                    "Quick, snappy motion",
                    "Simulate hitting a drum pad"
                ]
            },
            "drums_left": {
                "description": "Left hand in drumming position.",
                "tips": [
                    "Position left hand lower",
                    "Make fist grip",
                    "Ready to strike",
                    "Keep hand steady"
                ]
            },
            "drums_right": {
                "description": "Right hand in drumming position.",
                "tips": [
                    "Position right hand higher",
                    "Make fist grip",
                    "Ready to strike",
                    "Keep hand steady"
                ]
            }
        },
        "flute": {
            "flute_play": {
                "description": "Hands positioned as if holding and playing a flute.",
                "tips": [
                    "Both hands in front of mouth",
                    "Fingers positioned on sides",
                    "As if holding flute horizontally",
                    "Relaxed hand position"
                ]
            },
            "flute_high": {
                "description": "Fingers positioned for playing high notes (more fingers curled).",
                "tips": [
                    "Curl more fingers inward",
                    "Simulate covering more holes",
                    "Hand higher up",
                    "Precise finger positions"
                ]
            },
            "flute_low": {
                "description": "Fingers positioned for playing low notes (fewer fingers curled).",
                "tips": [
                    "Keep most fingers extended",
                    "Simulate covering fewer holes",
                    "Hand lower",
                    "Open finger positions"
                ]
            },
            "flute_breath": {
                "description": "Hand near mouth in breath/breathing gesture.",
                "tips": [
                    "Hand near mouth area",
                    "Open palm gesture",
                    "Simulate taking breath",
                    "Relaxed hand position"
                ]
            }
        }
    }
    
    def __init__(self, data_dir: str = "data/instrument_gestures"):
        """
        Initialize the instrument gesture collector.
        
        Args:
            data_dir (str): Directory to store collected data
        """
        self.data_dir = data_dir
        self.tracker = HandTracker(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        os.makedirs(data_dir, exist_ok=True)
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
    
    def extract_landmarks_as_list(self, hand_landmarks) -> list:
        """
        Convert MediaPipe landmarks to a list of floats.
        Returns 63 values: 21 landmarks × 3 coordinates (x, y, z)
        """
        landmark_list = []
        for landmark in hand_landmarks.landmark:
            landmark_list.extend([landmark.x, landmark.y, landmark.z])
        return landmark_list
    
    def display_gesture_instructions(self, frame, gesture_name: str, instrument: str):
        """
        Display detailed instructions for the current gesture.
        """
        gesture_info = self.INSTRUMENT_GESTURES[instrument][gesture_name]
        
        # Title with gesture name
        cv2.putText(
            frame,
            f"Gesture: {gesture_name.upper()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Description
        description = gesture_info["description"]
        # Wrap text if too long
        words = description.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) < 50:
                current_line += word + " "
            else:
                lines.append(current_line)
                current_line = word + " "
        lines.append(current_line)
        
        y_offset = 70
        for line in lines:
            cv2.putText(
                frame,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )
            y_offset += 25
        
        # Tips
        y_offset += 10
        cv2.putText(
            frame,
            "Tips:",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )
        y_offset += 25
        
        for tip in gesture_info["tips"][:3]:  # Show first 3 tips
            cv2.putText(
                frame,
                f"• {tip}",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1
            )
            y_offset += 22
        
        return frame
    
    def collect_gesture_data(self, instrument: str, gesture_name: str, num_samples: int = 100):
        """
        Collect training data for a specific instrument gesture.
        
        Args:
            instrument (str): Name of the instrument (piano, violin, drums, flute)
            gesture_name (str): Name of the gesture
            num_samples (int): Number of samples to collect
        """
        
        # Create subdirectory for this instrument
        instrument_dir = os.path.join(self.data_dir, instrument)
        os.makedirs(instrument_dir, exist_ok=True)
        
        gesture_file = os.path.join(instrument_dir, f"{gesture_name}.pkl")
        
        # Check if data already exists
        if os.path.exists(gesture_file):
            response = input(f"\nData for '{gesture_name}' already exists. Append more data? (y/n): ")
            if response.lower() == 'y':
                with open(gesture_file, 'rb') as f:
                    gesture_data = pickle.load(f)
            else:
                print("Cancelled.")
                return
        else:
            gesture_data = []
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Could not open webcam")
            return
        
        collected = len(gesture_data)
        frame_count = 0
        
        print(f"\n{'='*70}")
        print(f"COLLECTING: {instrument.upper()} - {gesture_name.upper()}")
        print(f"{'='*70}")
        
        while collected < num_samples:
            ret, frame = cap.read()
            
            if not ret:
                print("ERROR: Failed to read frame")
                break
            
            frame_count += 1
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            results = self.tracker.process(frame)
            frame = self.tracker.draw(frame, results)
            
            # Display gesture instructions
            frame = self.display_gesture_instructions(frame, gesture_name, instrument)
            
            # Display progress
            progress_y = frame.shape[0] - 150
            cv2.putText(
                frame,
                f"Progress: {collected}/{num_samples}",
                (10, progress_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Progress bar
            bar_length = 200
            filled = int((collected / num_samples) * bar_length)
            cv2.rectangle(frame, (10, progress_y + 20), (10 + bar_length, progress_y + 40), (100, 100, 100), -1)
            cv2.rectangle(frame, (10, progress_y + 20), (10 + filled, progress_y + 40), (0, 255, 0), -1)
            
            # Hand detection status
            status_y = frame.shape[0] - 80
            if results.multi_hand_landmarks:
                cv2.putText(
                    frame,
                    "✓ Hand Detected - Press 'c' to capture",
                    (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "✗ No Hand Detected",
                    (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
            
            # Controls
            cv2.putText(
                frame,
                "'c' - Capture | 'q' - Finish | 'SPACE' - Skip",
                (10, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                1
            )
            
            cv2.imshow(f"Collecting: {instrument} - {gesture_name}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                if results.multi_hand_landmarks:
                    landmarks = self.extract_landmarks_as_list(results.multi_hand_landmarks[0])
                    gesture_data.append(landmarks)
                    collected += 1
                    print(f"  ✓ Captured {collected}/{num_samples}")
                else:
                    print(f"  ✗ No hand detected in frame")
            
            elif key == ord('q') or key == 27:
                print(f"Finishing data collection.")
                break
            
            elif key == ord(' '):
                continue
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save collected data
        if collected > 0:
            with open(gesture_file, 'wb') as f:
                pickle.dump(gesture_data, f)
            print(f"\n✓ Data saved to: {gesture_file}")
            print(f"  Total samples for '{gesture_name}': {len(gesture_data)}")
        else:
            print("No data collected.")
    
    def interactive_collection(self):
        """
        Interactive menu for collecting data for all instruments and gestures.
        """
        print("\n" + "="*70)
        print("INSTRUMENT-SPECIFIC GESTURE DATA COLLECTOR")
        print("="*70)
        print("Collect training data for playing virtual instruments with hand gestures")
        print("="*70 + "\n")
        
        while True:
            # Show available instruments
            instruments = list(self.INSTRUMENT_GESTURES.keys())
            print("\nAvailable Instruments:")
            for i, instrument in enumerate(instruments, 1):
                num_gestures = len(self.INSTRUMENT_GESTURES[instrument])
                print(f"  {i}. {instrument.upper()} ({num_gestures} gestures)")
            print(f"  {len(instruments) + 1}. Collect all")
            print(f"  {len(instruments) + 2}. Exit")
            
            choice = input(f"\nSelect instrument (1-{len(instruments) + 2}): ").strip()
            
            if not choice.isdigit():
                print("Invalid input. Please enter a number.")
                continue
            
            choice = int(choice)
            
            if choice == len(instruments) + 2:
                print("\n✓ Thank you for collecting data!")
                break
            
            elif choice == len(instruments) + 1:
                # Collect all
                for instrument in instruments:
                    self._collect_for_instrument(instrument)
                break
            
            elif 1 <= choice <= len(instruments):
                instrument = instruments[choice - 1]
                self._collect_for_instrument(instrument)
            
            else:
                print("Invalid choice. Try again.")
    
    def _collect_for_instrument(self, instrument: str):
        """
        Collect data for all gestures of a specific instrument.
        """
        gestures = self.INSTRUMENT_GESTURES[instrument]
        
        print(f"\n{'='*70}")
        print(f"COLLECTING DATA FOR: {instrument.upper()}")
        print(f"{'='*70}")
        
        gesture_list = list(gestures.keys())
        
        for i, gesture in enumerate(gesture_list, 1):
            print(f"\n[{i}/{len(gesture_list)}] {gesture}")
            
            num_samples = input(f"How many samples for '{gesture}'? (default: 100): ").strip()
            num_samples = int(num_samples) if num_samples.isdigit() else 100
            
            self.collect_gesture_data(instrument, gesture, num_samples)
            
            if i < len(gesture_list):
                cont = input("Continue to next gesture? (y/n): ").strip().lower()
                if cont != 'y':
                    break
        
        print(f"\n✓ Finished collecting data for {instrument}")


def main():
    collector = InstrumentGestureCollector()
    collector.interactive_collection()


if __name__ == "__main__":
    main()