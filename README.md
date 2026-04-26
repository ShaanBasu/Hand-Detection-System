
# Hand Detection System

This project uses MediaPipe, OpenCV, and scikit-learn to detect hand landmarks, collect gesture data, train a gesture classifier, and run live gesture recognition for instrument-style hand motions.

It is designed around three instrument groups:

- Piano
- Guitar
- Drums

The current gesture pipeline is built for a single hand per sample during data collection and recognition. The visual demo in `main.py` can show up to two hands for display purposes.

## What the system does

The project has four main parts:

1. Real-time hand tracking demo
2. Gesture data collection from webcam input
3. Model training from saved landmark samples
4. Live gesture recognition with visual feedback

## Project structure

```text
Hand-Detection-System/
├── data/
│   └── raw_landmarks/
├── models/
├── src/
│   ├── __init__.py
│   ├── data_collector.py
│   ├── gesture_classifier.py
│   ├── hand_tracker.py
│   ├── main.py
│   └── recogniser.py
├── plan.txt
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.10+ recommended
- Webcam
- Windows, macOS, or Linux with camera access

Python dependencies are listed in `requirements.txt`:

- `mediapipe`
- `opencv-python`
- `numpy`
- `scikit-learn`
- `joblib`

## How the system works

### 1. Hand tracking

`src/hand_tracker.py` wraps MediaPipe Hands and provides:

- landmark detection
- skeleton drawing
- handedness detection

### 2. Data collection

`src/data_collector.py` opens the webcam, shows gesture instructions, and saves captured hand landmarks as `.pkl` files.

Each sample is stored as a flat list of 63 values:

- 21 landmarks
- 3 values each: `x`, `y`, `z`

The collector now uses a burst flow:

- press `c`
- wait 3 seconds
- capture 5 frames at 0.2 second intervals

This gives you time to move into position before the burst starts.

### 3. Training

`src/gesture_classifier.py` loads the `.pkl` files, trains a Random Forest model, evaluates it, and saves the trained pipeline to `models/gesture_model.joblib`.

### 4. Live recognition

`src/recogniser.py` loads the saved model and predicts the gesture in real time from webcam input. It overlays:

- gesture name
- confidence
- instrument-specific color feedback

## Setup

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your PowerShell execution policy blocks activation, run PowerShell as needed for your environment or use the equivalent command prompt activation script.

## Run the demo

To test webcam hand tracking only:

```powershell
python src/main.py
```

Controls:

- `Esc` to exit
- `s` to save a frame image

## Collect training data

Run:

```powershell
python src/data_collector.py
```

Then follow the interactive menu:

1. Choose an instrument group.
2. Choose how many samples to collect per gesture.
3. Press `c` to start the 3-second countdown.
4. Move your hand into the correct pose.
5. Let the burst capture 5 frames automatically.

Collector controls:

- `c` starts the delayed burst capture
- `SPACE` cancels a pending countdown/burst
- `q` finishes the current capture session
- `Esc` also exits the session

Collected files are saved under:

```text
data/instrument_gestures/<instrument>/<gesture>.pkl
```

## Train the classifier

Once you have gesture `.pkl` files, run:

```powershell
python src/gesture_classifier.py
```

This will:

1. Load all gesture samples from `data/instrument_gestures`
2. Split the data into train/test sets
3. Train the classifier
4. Print accuracy and a classification report
5. Save the model to `models/gesture_model.joblib`

### Important training note

If training errors out in `gesture_classifier.py`, make sure the scaler line in `train()` uses `self.scaler.transform(X_test)` rather than a misspelled attribute name.

## Run live recognition

After the model has been trained and saved, run:

```powershell
python src/recogniser.py
```

This opens the webcam and shows real-time gesture predictions.

Controls:

- `Esc` to quit

## Expected folder outputs

After collecting and training, you should have:

```text
data/instrument_gestures/
	piano/
		piano_ready.pkl
		piano_press.pkl
		piano_left.pkl
		piano_right.pkl
	guitar/
		guitar_chord.pkl
		guitar_strum.pkl
		guitar_pick.pkl
		guitar_mute.pkl
	drums/
		drums_stick_grip.pkl
		drums_hit.pkl
		drums_left.pkl
		drums_right.pkl

models/
	gesture_model.joblib
```

## Troubleshooting

- If the webcam does not open, make sure no other app is using it.
- If recognition says no trained model exists, train the classifier first.
- If training fails because there is no data, collect gesture samples first.
- If the model performs poorly, collect more samples and make sure each gesture is captured clearly and consistently.

## Suggested workflow

1. Run `python src/main.py` to confirm the webcam and hand tracking work.
2. Run `python src/data_collector.py` and record gesture samples.
3. Run `python src/gesture_classifier.py` to train and save the model.
4. Run `python src/recogniser.py` to test live predictions.

## Notes

- The project currently focuses on visual feedback and instrument-style gesture classification.
- The training data format is intentionally simple so it can be expanded later.
- If you want to move to two-hand gesture support later, the collector, model, and recogniser will all need to be updated together.
