import numpy as np
import onnxruntime as ort
from collections import deque
from src.config import ACTIONS, MODEL_DIR

class ASLInferenceEngine:
    def __init__(self):
        model_path = str(MODEL_DIR / "asl_model.onnx")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Buffer to keep the last 5 frame predictions for temporal smoothing
        self.prediction_history = deque(maxlen=6)

    def predict(self, sequence: np.ndarray):
        input_tensor = np.expand_dims(sequence.astype(np.float32), axis=0)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0][0]

        # Numerically stable softmax with temperature scaling
        temperature = 1.2
        exp_logits = np.exp((logits - np.max(logits)) / temperature)
        probabilities = exp_logits / exp_logits.sum()

        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])
        raw_action = ACTIONS[predicted_idx]

        # Store prediction in smoothing deque
        self.prediction_history.append(raw_action)

        # Check if the same action appeared in at least 4 of the last 6 evaluations
        most_frequent_action = max(set(self.prediction_history), key=self.prediction_history.count)
        frequent_count = self.prediction_history.count(most_frequent_action)

        # Trigger detection only if confident and consistent
        if confidence > 0.70 and frequent_count >= 4 and most_frequent_action != "idle":
            final_action = most_frequent_action
        else:
            final_action = "Listening..."

        return {
            "action": final_action,
            "raw_action": raw_action,
            "confidence": confidence,
            "all_probabilities": {action: round(float(prob), 3) for action, prob in zip(ACTIONS, probabilities)}
        }