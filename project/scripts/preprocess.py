import os
import json
import pickle
import numpy as np

RAW_PATH = "data/raw/mosei_senti_data.pkl"
SAVE_DIR = "data/processed"

os.makedirs(SAVE_DIR, exist_ok=True)


def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def print_split_info(split_name, split_data):
    print(f"\n[{split_name}]")
    for key, value in split_data.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}")


def clean_array(x, name="array"):
    x = np.asarray(x)

    nan_count = np.isnan(x).sum() if np.issubdtype(x.dtype, np.number) else 0
    inf_count = np.isinf(x).sum() if np.issubdtype(x.dtype, np.number) else 0

    if nan_count > 0 or inf_count > 0:
        print(f"{name}: nan={nan_count}, inf={inf_count}, replace with 0")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return x.astype(np.float32)


def reshape_labels(y):
    y = np.asarray(y, dtype=np.float32)
    return y.reshape(-1)


def compute_norm_stats(train_x):
    # 沿着样本维 N 和时间维 T 求统计量，保留 feature 维
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)

    # 防止某些维度 std 太小
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def process_modality(train_x, valid_x, test_x, modality_name):
    print(f"\nProcessing modality: {modality_name}")

    train_x = clean_array(train_x, f"{modality_name}_train")
    valid_x = clean_array(valid_x, f"{modality_name}_valid")
    test_x  = clean_array(test_x,  f"{modality_name}_test")

    mean, std = compute_norm_stats(train_x)

    train_x = normalize(train_x, mean, std)
    valid_x = normalize(valid_x, mean, std)
    test_x  = normalize(test_x,  mean, std)

    print(f"{modality_name}: mean shape={mean.shape}, std shape={std.shape}")
    return train_x, valid_x, test_x, mean, std


def main():
    data = load_pkl(RAW_PATH)

    train = data["train"]
    valid = data["valid"]
    test  = data["test"]

    print("Loaded PKL successfully.")
    print_split_info("train", train)
    print_split_info("valid", valid)
    print_split_info("test", test)

    # labels
    y_train = reshape_labels(train["labels"])
    y_valid = reshape_labels(valid["labels"])
    y_test  = reshape_labels(test["labels"])

    # modalities
    text_train, text_valid, text_test, text_mean, text_std = process_modality(
        train["text"], valid["text"], test["text"], "text"
    )
    audio_train, audio_valid, audio_test, audio_mean, audio_std = process_modality(
        train["audio"], valid["audio"], test["audio"], "audio"
    )
    vision_train, vision_valid, vision_test, vision_mean, vision_std = process_modality(
        train["vision"], valid["vision"], test["vision"], "vision"
    )

    # 保存每个 split
    np.savez_compressed(
        os.path.join(SAVE_DIR, "train_processed.npz"),
        text=text_train,
        audio=audio_train,
        vision=vision_train,
        labels=y_train,
        id=train["id"]
    )

    np.savez_compressed(
        os.path.join(SAVE_DIR, "valid_processed.npz"),
        text=text_valid,
        audio=audio_valid,
        vision=vision_valid,
        labels=y_valid,
        id=valid["id"]
    )

    np.savez_compressed(
        os.path.join(SAVE_DIR, "test_processed.npz"),
        text=text_test,
        audio=audio_test,
        vision=vision_test,
        labels=y_test,
        id=test["id"]
    )

    # 保存 normalization 参数
    np.savez_compressed(
        os.path.join(SAVE_DIR, "norm_stats.npz"),
        text_mean=text_mean, text_std=text_std,
        audio_mean=audio_mean, audio_std=audio_std,
        vision_mean=vision_mean, vision_std=vision_std
    )

    meta = {
        "train_size": int(text_train.shape[0]),
        "valid_size": int(text_valid.shape[0]),
        "test_size": int(text_test.shape[0]),
        "text_shape": list(text_train.shape),
        "audio_shape": list(audio_train.shape),
        "vision_shape": list(vision_train.shape),
        "label_shape": list(y_train.shape)
    }

    with open(os.path.join(SAVE_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nPreprocessing complete.")
    print(f"Saved processed files to: {SAVE_DIR}")


if __name__ == "__main__":
    main()