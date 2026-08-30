import torch
from src.config import NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES, SEQUENCE_LENGTH, MODEL_DIR
from src.model import ASLGestureClassifier

def export():
    model_path = MODEL_DIR / "best_model.pt"
    onnx_path = MODEL_DIR / "asl_model.onnx"

    model = ASLGestureClassifier(NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, SEQUENCE_LENGTH, NUM_FEATURES, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_landmarks'],
        output_names=['class_logits'],
        dynamic_axes={'input_landmarks': {0: 'batch_size'}, 'class_logits': {0: 'batch_size'}}
    )
    print(f"ONNX exported successfully to: {onnx_path}")

if __name__ == "__main__":
    export()