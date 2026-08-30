import numpy as np
import torch
from torch.utils.data import Dataset
from src.config import ACTIONS, DATA_DIR, SEQUENCE_LENGTH

class ASLDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def load_data():
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(ACTIONS)}

    for action in ACTIONS:
        action_dir = DATA_DIR / action
        if not action_dir.exists():
            continue
        for seq_folder in sorted(action_dir.iterdir()):
            if not seq_folder.is_dir():
                continue
            window = []
            for frame_num in range(SEQUENCE_LENGTH):
                frame_path = seq_folder / f"{frame_num}.npy"
                if frame_path.exists():
                    res = np.load(frame_path)
                    window.append(res)
            if len(window) == SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(label_map[action])

    return np.array(sequences), np.array(labels)