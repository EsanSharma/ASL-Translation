ASL Translation

Real-time American Sign Language (ASL) recognition system that captures hand gestures via webcam, extracts landmark features with MediaPipe, classifies them with a PyTorch model, and streams predictions live through a FastAPI + WebSocket server.

Overview

This project turns raw webcam video into live ASL predictions through a full machine learning pipeline:

Collect — record labeled gesture samples from a webcam
Extract — convert raw frames into MediaPipe hand landmark features
Train — fit a PyTorch classification model on the extracted features
Export — convert the trained model to ONNX for fast, portable inference
Serve — run a FastAPI backend that streams predictions over WebSockets in real time
Features
Real-time hand tracking using MediaPipe
Custom-trained PyTorch model for ASL gesture classification
ONNX export for lightweight, low-latency inference
WebSocket-based live prediction streaming via FastAPI
Modular pipeline — collect, extract, train, and export are all separate, reusable scripts
Tech Stack
Layer	Technology
Computer Vision	MediaPipe, OpenCV
Model	PyTorch, scikit-learn
Inference	ONNX, ONNX Runtime
Backend	FastAPI, Uvicorn, WebSockets
Project Structure
ASL-Translation/
├── server/
│   ├── __init__.py
│   └── app.py              # FastAPI app — serves real-time predictions over WebSockets
├── src/
│   ├── __init__.py
│   ├── collect_data.py     # Captures labeled gesture samples from the webcam
│   ├── extract_features.py # Extracts MediaPipe hand landmarks from raw data
│   ├── dataset.py          # PyTorch Dataset for loading extracted features
│   ├── model.py            # Model architecture definition
│   ├── engine.py           # Training and evaluation loops
│   ├── train.py            # Trains the classification model
│   ├── export_onnx.py      # Exports the trained model to ONNX format
│   └── config.py           # Shared paths, constants, and hyperparameters
├── requirements.txt
└── .gitignore
Getting Started
Prerequisites
Python 3.9+
A webcam
pip
Installation

Clone the repository and install dependencies:

bash
git clone https://github.com/EsanSharma/ASL-Translation.git
cd ASL-Translation
pip install -r requirements.txt
Usage

1. Collect gesture data

Record labeled samples of ASL gestures using your webcam:

bash
python src/collect_data.py

2. Extract features

Convert the raw collected data into MediaPipe hand landmark features:

bash
python src/extract_features.py

3. Train the model

Train the classifier on the extracted features:

bash
python src/train.py

4. Export to ONNX

Export the trained model for fast inference:

bash
python src/export_onnx.py

5. Run the server

Start the FastAPI server to serve live predictions:

bash
uvicorn server.app:app --reload

The server will be available at http://localhost:8000.

Configuration

Adjust paths, hyperparameters, and other settings in src/config.py before running the pipeline.

Roadmap
 Add a frontend client for live webcam demo
 Expand gesture vocabulary
 Add model evaluation metrics and benchmarks
 Deploy a hosted demo
Contributing

Contributions are welcome. Feel free to open an issue or submit a pull request.

License

Specify a license (e.g. MIT) for this project.

Author

Esan Sharma GitHub