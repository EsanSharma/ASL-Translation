import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import base64
import logging
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import json
import base64
import logging
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketState

from src.config import SEQUENCE_LENGTH
from src.extract_features import mediapipe_detection, extract_landmarks, mp_holistic
from src.engine import ASLInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="ASL Translation Engine")

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

engine = None
holistic_detector = None

@app.on_event("startup")
def startup_event():
    global engine, holistic_detector
    engine = ASLInferenceEngine()
    # Initialize holistic once globally
    holistic_detector = mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    logging.info("ONNX Inference Engine and MediaPipe Holistic initialized.")

@app.on_event("shutdown")
def shutdown_event():
    global holistic_detector
    if holistic_detector:
        holistic_detector.close()

@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"status": "online", "message": "ASL WebSocket Backend Ready"}

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    logging.info("Client connected to stream.")
    sequence_buffer = []

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Base64 decode frame
            img_bytes = base64.b64decode(message["image"].split(",")[-1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            # Extract features using global detector
            results = mediapipe_detection(frame, holistic_detector)
            keypoints = extract_landmarks(results)
            sequence_buffer.append(keypoints)

            if len(sequence_buffer) > SEQUENCE_LENGTH:
                sequence_buffer.pop(0)

            # Predict once window buffer has 30 frames
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                prediction = engine.predict(np.array(sequence_buffer))
                await websocket.send_text(json.dumps(prediction))
            else:
                await websocket.send_text(json.dumps({
                    "status": "buffering", 
                    "frames": len(sequence_buffer)
                }))

    except WebSocketDisconnect:
        logging.info("Client disconnected.")
    except Exception as e:
        logging.error(f"WebSocket streaming error: {e}")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()