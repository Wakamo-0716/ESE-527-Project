import numpy as np
import torch
from torch.utils.data import Dataset

class MoseiDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.text = data["text"].astype(np.float32)      # (N, 50, 300)
        self.audio = data["audio"].astype(np.float32)    # (N, 50, 74)
        self.vision = data["vision"].astype(np.float32)  # (N, 50, 35)
        self.labels = data["labels"].astype(np.float32)  # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": torch.tensor(self.text[idx], dtype=torch.float32),
            "audio": torch.tensor(self.audio[idx], dtype=torch.float32),
            "vision": torch.tensor(self.vision[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }