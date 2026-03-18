import pickle
import numpy as np

pkl_path = "data/raw/mosei_senti_data.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(type(data))

if isinstance(data, dict):
    print("Top-level keys:", data.keys())
    for k, v in data.items():
        print(f"\n[{k}] type = {type(v)}")
        if isinstance(v, dict):
            print(" subkeys:", v.keys())
            for sk, sv in v.items():
                if hasattr(sv, "shape"):
                    print(f"   {sk}: shape={sv.shape}, dtype={sv.dtype}")
                else:
                    print(f"   {sk}: type={type(sv)}")