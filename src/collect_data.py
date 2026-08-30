import time
import cv2
import numpy as np
from src.config import ACTIONS, NUM_SEQUENCES_PER_CLASS, SEQUENCE_LENGTH, DATA_DIR
from src.extract_features import mediapipe_detection, extract_landmarks, mp_holistic, mp_drawing

def collect_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for action in ACTIONS:
        for seq in range(NUM_SEQUENCES_PER_CLASS):
            (DATA_DIR / action / str(seq)).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in ACTIONS:
            print(f"\n==========================================")
            print(f"  PREPARING RECORDING FOR: '{action.upper()}'")
            print(f"==========================================")
            
        
            for countdown in range(3, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    continue
                cv2.putText(frame, f'GET READY FOR: {action.upper()} ({countdown}s)', (50, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.imshow('ASL Data Collector', frame)
                cv2.waitKey(1000)

            for seq in range(NUM_SEQUENCES_PER_CLASS):
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    results = mediapipe_detection(frame, holistic)
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                    # Display status header
                    status_text = f"Action: {action} | Sequence: #{seq + 1}/{NUM_SEQUENCES_PER_CLASS} | Frame: {frame_num + 1}/{SEQUENCE_LENGTH}"
                    cv2.rectangle(frame, (0, 0), (640, 40), (20, 20, 20), -1)
                    cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    if frame_num == 0:
                        cv2.putText(frame, 'START GESTURE NOW', (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('ASL Data Collector', frame)
                        cv2.waitKey(600)  
                    else:
                        cv2.imshow('ASL Data Collector', frame)

                    keypoints = extract_landmarks(results)
                    np.save(DATA_DIR / action / str(seq) / f"{frame_num}.npy", keypoints)

                 
                    if cv2.waitKey(50) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()
    print("\n[SUCCESS] Data collection complete for all classes.")

if __name__ == "__main__":
    collect_data()