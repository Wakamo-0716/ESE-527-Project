import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

DATA_PATH = "data/processed/train_processed.npz"
REPORT_DIR = "data/reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def load_data(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def summarize_modality(x, name):
    return {
        "modality": name,
        "shape": str(x.shape),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "zero_ratio": float((x == 0).mean()),
        "nan_ratio": float(np.isnan(x).mean()),
    }


def plot_label_distribution(labels):
    plt.figure(figsize=(6, 4))
    plt.hist(labels, bins=50)
    plt.xlabel("Sentiment Label")
    plt.ylabel("Count")
    plt.title("Label Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "label_distribution.png"), dpi=200)
    plt.close()


def plot_temporal_stats(x, name):
    mean_t = x.mean(axis=(0, 2))
    std_t = x.std(axis=(0, 2))
    t = np.arange(x.shape[1])

    plt.figure(figsize=(7, 4))
    plt.plot(t, mean_t)
    plt.xlabel("Time Step")
    plt.ylabel("Mean")
    plt.title(f"{name} Temporal Mean")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f"{name.lower()}_temporal_mean.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(t, std_t)
    plt.xlabel("Time Step")
    plt.ylabel("Std")
    plt.title(f"{name} Temporal Std")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f"{name.lower()}_temporal_std.png"), dpi=200)
    plt.close()


def cross_modal_consistency(text, audio, vision):
    text_scalar = text.mean(axis=(1, 2))
    audio_scalar = audio.mean(axis=(1, 2))
    vision_scalar = vision.mean(axis=(1, 2))

    arr = np.stack([text_scalar, audio_scalar, vision_scalar], axis=0)
    corr = np.corrcoef(arr)
    return corr


def isolation_forest_outliers(text, audio, vision, contamination=0.05):
    x = np.concatenate([
        text.reshape(text.shape[0], -1),
        audio.reshape(audio.shape[0], -1),
        vision.reshape(vision.shape[0], -1)
    ], axis=1)

    clf = IsolationForest(contamination=contamination, random_state=42)
    pred = clf.fit_predict(x)
    score = clf.decision_function(x)

    return pred, score


def temporal_outlier_ratio(x, z_threshold=3.0):
    diff = np.diff(x, axis=1)  # (N, T-1, D)
    diff_mean = diff.mean()
    diff_std = diff.std()
    z = (diff - diff_mean) / (diff_std + 1e-8)
    mask = np.abs(z) > z_threshold
    return float(mask.mean())


def main():
    data = load_data(DATA_PATH)

    text = data["text"]
    audio = data["audio"]
    vision = data["vision"]
    labels = data["labels"]

    # 基本统计
    stats = [
        summarize_modality(text, "text"),
        summarize_modality(audio, "audio"),
        summarize_modality(vision, "vision"),
    ]
    pd.DataFrame(stats).to_csv(os.path.join(REPORT_DIR, "modality_stats.csv"), index=False)

    # 标签分布
    plot_label_distribution(labels)

    # 时间稳定性
    plot_temporal_stats(text, "text")
    plot_temporal_stats(audio, "audio")
    plot_temporal_stats(vision, "vision")

    # 模态一致性
    corr = cross_modal_consistency(text, audio, vision)

    # Isolation Forest
    pred, score = isolation_forest_outliers(text, audio, vision, contamination=0.05)
    outlier_ratio = float((pred == -1).mean())

    outlier_df = pd.DataFrame({
        "sample_idx": np.arange(len(pred)),
        "is_outlier": (pred == -1).astype(int),
        "iforest_score": score
    })
    outlier_df.to_csv(os.path.join(REPORT_DIR, "iforest_outliers.csv"), index=False)

    # Temporal outlier
    text_temp_ratio = temporal_outlier_ratio(text)
    audio_temp_ratio = temporal_outlier_ratio(audio)
    vision_temp_ratio = temporal_outlier_ratio(vision)

    summary = {
        "cross_modal_corr": corr.tolist(),
        "iforest_outlier_ratio": outlier_ratio,
        "text_temporal_outlier_ratio": text_temp_ratio,
        "audio_temporal_outlier_ratio": audio_temp_ratio,
        "vision_temporal_outlier_ratio": vision_temp_ratio,
        "label_mean": float(labels.mean()),
        "label_std": float(labels.std()),
        "label_min": float(labels.min()),
        "label_max": float(labels.max())
    }

    with open(os.path.join(REPORT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Analysis complete.")
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()