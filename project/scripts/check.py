import numpy as np

data = np.load("data/processed/train_processed.npz", allow_pickle=True)

for k in data.files:
    print(k, data[k].shape, data[k].dtype)