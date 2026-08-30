# ASL Translation

A real-time American Sign Language (ASL) translator that uses **MediaPipe Holistic** landmark tracking and a **sequence-based deep learning model** to recognize signs from a webcam feed and translate them into text.

Currently supports **6 signs**:

| Sign | Gesture |
|------|---------|
| `idle` | No active sign / resting position |
| `hello` | Hello |
| `yes` | Yes |
| `no` | No |
| `thank you` | Thank you |
| `help` | Help |

More signs will be added as the dataset grows — see [Roadmap](#roadmap).

---

## How It Works

1. **Data Collection** — Webcam footage is captured and passed through MediaPipe Holistic to extract pose and hand landmarks for each sign, recorded as multiple short sequences.
2. **Feature Extraction** — Pose, left-hand, and right-hand landmarks are flattened into numerical feature vectors per frame.
3. **Training** — A sequence model (trained on stacked frame-by-frame landmark sequences) learns to classify a window of frames into one of the six sign classes.
4. **Export** — The trained model is exported to **ONNX** for fast, cross-platform, framework-independent inference.
5. **Inference** — A lightweight engine runs the ONNX model on a live webcam feed and streams predictions to a simple web front-end.

---

## Project Structure

```
ASL-Translation/
├── data/                     # Collected landmark sequences (per sign, per sample)
├── models/                   # Saved / exported models (.onnx, checkpoints)
├── server/
│   └── static/
│       └── index.html        # Front-end UI for live translation
├── src/
│   ├── __init__.py
│   ├── config.py             # ACTIONS, sequence length, data paths, etc.
│   ├── extract_features.py   # MediaPipe detection + landmark extraction helpers
│   ├── collect_data.py       # Webcam-based dataset recording script
│   ├── dataset.py            # Dataset loading / preprocessing for training
│   ├── model.py              # Model architecture definition
│   ├── train.py              # Training loop
│   ├── export_onnx.py        # Exports the trained model to ONNX
│   └── engine.py             # Inference engine (loads ONNX model, runs predictions)
├── app.py                    # Application entry point / server
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- A webcam
- pip

### Installation

git clone https://github.com/<your-username>/ASL-Translation.git
cd ASL-Translation
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS / Linux
pip install -r requirements.txt

---

## Usage

### 1. Collect Training Data

Records webcam sequences for each configured sign in `src/config.py`, saving landmark data under `data/`.

python -m src.collect_data

A 3-second countdown appears before each sequence so you have time to get in position.

### 2. Train the Model

python -m src.train

### 3. Export to ONNX

python -m src.export_onnx


### 4. Run the Translator


python app.py


Then open your browser to the address printed in the terminal to see live sign predictions on the web UI (`server/static/index.html`).

---

## Tech Stack

- **[MediaPipe](https://developers.google.com/mediapipe)** — pose & hand landmark detection
- **OpenCV** — webcam capture and frame processing
- **NumPy** — numerical feature handling
- **ONNX Runtime** — fast, portable model inference
- **Python** (Flask/FastAPI-style `app.py` server) — serves the web front-end and prediction API

---

## Roadmap

-  Expand vocabulary beyond the initial 6 signs
-  Improve model accuracy with more training samples per class
-  Add confidence smoothing / debouncing for steadier predictions
-  Package as a standalone desktop app
-  Mobile support

---

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you'd like to add new signs, improve accuracy, or enhance the UI.

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
