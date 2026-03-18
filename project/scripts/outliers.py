import os
import json
import numpy as np
import pandas as pd

REPORT_DIR = "data/reports"
DATA_PATH = "data/processed/train_processed.npz"
OUTLIER_CSV = os.path.join(REPORT_DIR, "iforest_outliers.csv")
SAVE_PREFIX = os.path.join(REPORT_DIR, "iforest_stats")


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def sample_std(x):
    # 每个样本整体波动，输出 shape (N,)
    return x.std(axis=(1, 2))


def temporal_jump(x):
    # 每个样本相邻时间步平均绝对差分，输出 shape (N,)
    diff = np.diff(x, axis=1)  # (N, T-1, D)
    return np.mean(np.abs(diff), axis=(1, 2))


def summarize_group(df, group_name):
    summary = {
        "group": group_name,
        "count": int(len(df)),
        "label_mean": float(df["label"].mean()),
        "label_std": float(df["label"].std()),
        "label_min": float(df["label"].min()),
        "label_max": float(df["label"].max()),
        "abs_label_mean": float(df["abs_label"].mean()),
        "extreme_label_ratio_|label|>2": float((df["abs_label"] > 2).mean()),

        "text_std_mean": float(df["text_std"].mean()),
        "audio_std_mean": float(df["audio_std"].mean()),
        "vision_std_mean": float(df["vision_std"].mean()),

        "text_jump_mean": float(df["text_jump"].mean()),
        "audio_jump_mean": float(df["audio_jump"].mean()),
        "vision_jump_mean": float(df["vision_jump"].mean()),

        "iforest_score_mean": float(df["iforest_score"].mean()),
        "iforest_score_min": float(df["iforest_score"].min()),
        "iforest_score_max": float(df["iforest_score"].max()),
    }
    return summary


def main():
    # 读取数据
    arr = load_npz(DATA_PATH)
    outlier_df = pd.read_csv(OUTLIER_CSV)

    text = arr["text"]
    audio = arr["audio"]
    vision = arr["vision"]
    labels = arr["labels"]

    # 安全检查
    assert len(outlier_df) == len(labels), "CSV 行数和 train 样本数不一致"
    assert "sample_idx" in outlier_df.columns
    assert "is_outlier" in outlier_df.columns
    assert "iforest_score" in outlier_df.columns

    # 构造统计列
    df = outlier_df.copy()
    df = df.sort_values("sample_idx").reset_index(drop=True)

    df["label"] = labels
    df["abs_label"] = np.abs(labels)

    df["text_std"] = sample_std(text)
    df["audio_std"] = sample_std(audio)
    df["vision_std"] = sample_std(vision)

    df["text_jump"] = temporal_jump(text)
    df["audio_jump"] = temporal_jump(audio)
    df["vision_jump"] = temporal_jump(vision)

    # 分组
    df_out = df[df["is_outlier"] == 1].copy()
    df_norm = df[df["is_outlier"] == 0].copy()

    # 总体统计
    overall = {
        "total_samples": int(len(df)),
        "outlier_count": int((df["is_outlier"] == 1).sum()),
        "normal_count": int((df["is_outlier"] == 0).sum()),
        "outlier_ratio": float((df["is_outlier"] == 1).mean()),
        "score_label_corr": float(df["iforest_score"].corr(df["abs_label"])),
        "score_text_jump_corr": float(df["iforest_score"].corr(df["text_jump"])),
        "score_audio_jump_corr": float(df["iforest_score"].corr(df["audio_jump"])),
        "score_vision_jump_corr": float(df["iforest_score"].corr(df["vision_jump"])),
    }

    # 分组汇总
    summaries = [
        summarize_group(df_norm, "normal"),
        summarize_group(df_out, "outlier"),
    ]
    summary_df = pd.DataFrame(summaries)

    # 最异常 top 20
    top20 = df.nsmallest(20, "iforest_score")[
        [
            "sample_idx", "iforest_score", "is_outlier", "label", "abs_label",
            "text_std", "audio_std", "vision_std",
            "text_jump", "audio_jump", "vision_jump"
        ]
    ]

    # 保存
    df.to_csv(f"{SAVE_PREFIX}_full.csv", index=False)
    summary_df.to_csv(f"{SAVE_PREFIX}_summary.csv", index=False)
    top20.to_csv(f"{SAVE_PREFIX}_top20_most_anomalous.csv", index=False)

    with open(f"{SAVE_PREFIX}_overall.json", "w") as f:
        json.dump(overall, f, indent=2)

    print("=== Overall ===")
    print(json.dumps(overall, indent=2))

    print("\n=== Group Summary ===")
    print(summary_df)

    print("\n=== Top 20 Most Anomalous Samples ===")
    print(top20.head(10))


if __name__ == "__main__":
    main()