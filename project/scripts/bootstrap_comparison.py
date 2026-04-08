import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import MoseiDataset
from models.early_fusion import EarlyFusionLSTM
from models.gated_fusion import GatedFusionLSTM
from models.cross_modal_attention import CrossModalAttentionLSTM
from models.tensor_fusion import TensorFusionLSTM
from models.unimodal_lstm import UnimodalLSTM


# =========================
# 1. Config
# =========================
DATA_DIR = "data/processed"
BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoints
CKPT_EARLY = "checkpoints/best_early_bs32_lr0.001_wd1e-05_pd64_hd128_nl1_do0.2.pt"
CKPT_GATED = "checkpoints/best_gated_bs32_lr0.001_wd1e-05_pd64_hd128_nl1_do0.2.pt"
CKPT_TENSOR = "checkpoints/best_tensor_bs32_lr0.001_wd1e-05_pd64_hd128_nl1_do0.2.pt"
CKPT_CROSS = "checkpoints_best_cross/best_cross_attn_bs32_lr0.001_wd1e-05_pd64_hd256_nl2_do0.2.pt"
CKPT_TEXT = "checkpoints_best_text/best_text_bs32_lr0.001_wd1e-05_hd128_nl1_do0.2.pt"
CKPT_BI_TA = "checkpoints_bimodal_cross/best_cross_ta.pt"
CKPT_BI_TV = "checkpoints_bimodal_cross/best_cross_tv.pt"
CKPT_BI_AV = "checkpoints_bimodal_cross/best_cross_av.pt"

OUTPUT_PRED_CSV = "bootstrap_predictions_group.csv"
OUTPUT_DIR = "prediction_npz"
OUTPUT_COMPARE_JSON = "bootstrap_comparison_results_group.json"


# =========================
# 2. Dataset / Loader
# =========================
def make_test_loader(data_dir, batch_size=32, num_workers=0):
    test_path = os.path.join(data_dir, "test_processed.npz")
    test_dataset = MoseiDataset(test_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader


# =========================
# 3. Build models
# =========================
def build_early():
    return EarlyFusionLSTM(
        proj_dim=64,
        hidden_dim=128,
        num_layers=1,
        dropout=0.2,
    )


def build_gated():
    return GatedFusionLSTM(
        proj_dim=64,
        hidden_dim=128,
        num_layers=1,
        dropout=0.2,
    )


def build_tensor():
    return TensorFusionLSTM(
        proj_dim=64,
        hidden_dim=128,
        num_layers=1,
        dropout=0.2,
    )


def build_cross():
    return CrossModalAttentionLSTM(
        proj_dim=64,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2,
    )


def build_text():
    return UnimodalLSTM(
        input_dim=300,
        hidden_dim=128,
        num_layers=1,
        dropout=0.2,
    )


def load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# =========================
# 4. Prediction helpers
# =========================
@torch.no_grad()
def predict_multimodal(model, loader, device):
    all_preds = []
    all_labels = []

    for batch in loader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        labels = batch["label"].to(device)

        preds = model(text, audio, vision)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


@torch.no_grad()
def predict_text(model, loader, device):
    all_preds = []
    all_labels = []

    for batch in loader:
        text = batch["text"].to(device)
        labels = batch["label"].to(device)

        preds = model(text)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


@torch.no_grad()
def predict_bimodal_cross(model, loader, device, pair):
    all_preds = []
    all_labels = []

    for batch in loader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        labels = batch["label"].to(device)

        if pair == "ta":
            vision = torch.zeros_like(vision)
        elif pair == "tv":
            audio = torch.zeros_like(audio)
        elif pair == "av":
            text = torch.zeros_like(text)
        else:
            raise ValueError(f"Unknown bimodal pair: {pair}")

        preds = model(text, audio, vision)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


# =========================
# 5. Metrics
# =========================
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def corr(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    c = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(c):
        return 0.0
    return float(c)


def compute_metrics(y_true, y_pred):
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "Corr": corr(y_true, y_pred),
    }


# =========================
# 6. Bootstrap comparison
# =========================
def bootstrap_mae_diff(y_true, pred_a, pred_b, n_boot=10000, seed=42):
    """
    Returns distribution of delta = MAE_A - MAE_B.
    If delta > 0, model B is better (lower MAE).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_bs = y_true[idx]
        a_bs = pred_a[idx]
        b_bs = pred_b[idx]
        deltas[i] = mae(y_bs, a_bs) - mae(y_bs, b_bs)

    obs_delta = mae(y_true, pred_a) - mae(y_true, pred_b)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])

    return {
        "observed_delta_mae": float(obs_delta),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "bootstrap_mean_delta": float(np.mean(deltas)),
    }, deltas



def permutation_test_mae(y_true, pred_a, pred_b, n_perm=10000, seed=42):
    """
    Paired permutation test on sample-level absolute errors.
    H0: the two models are exchangeable on each sample.
    """
    rng = np.random.default_rng(seed)

    err_a = np.abs(y_true - pred_a)
    err_b = np.abs(y_true - pred_b)

    observed = float(np.mean(err_a - err_b))
    perm_stats = np.empty(n_perm, dtype=np.float64)

    for i in range(n_perm):
        swap = rng.integers(0, 2, size=len(y_true)).astype(bool)
        perm_a = np.where(swap, err_b, err_a)
        perm_b = np.where(swap, err_a, err_b)
        perm_stats[i] = np.mean(perm_a - perm_b)

    p_value = float(np.mean(np.abs(perm_stats) >= abs(observed)))

    return {
        "observed_mean_error_diff": float(observed),
        "p_value_two_sided": p_value,
    }, perm_stats


# =========================
# 7. Main
# =========================
def main():
    print(f"Using device: {DEVICE}")

    test_loader = make_test_loader(DATA_DIR, BATCH_SIZE, NUM_WORKERS)

    # load models
    early_model = load_checkpoint(build_early(), CKPT_EARLY, DEVICE)
    gated_model = load_checkpoint(build_gated(), CKPT_GATED, DEVICE)
    tensor_model = load_checkpoint(build_tensor(), CKPT_TENSOR, DEVICE)
    cross_model = load_checkpoint(build_cross(), CKPT_CROSS, DEVICE)
    text_model = load_checkpoint(build_text(), CKPT_TEXT, DEVICE)

    bimodal_models = {}
    if os.path.exists(CKPT_BI_TA):
        bimodal_models["text+audio"] = (load_checkpoint(build_cross(), CKPT_BI_TA, DEVICE), "ta")
    if os.path.exists(CKPT_BI_TV):
        bimodal_models["text+vision"] = (load_checkpoint(build_cross(), CKPT_BI_TV, DEVICE), "tv")
    if os.path.exists(CKPT_BI_AV):
        bimodal_models["audio+vision"] = (load_checkpoint(build_cross(), CKPT_BI_AV, DEVICE), "av")

    # predictions
    pred_early, y_true = predict_multimodal(early_model, test_loader, DEVICE)
    pred_gated, _ = predict_multimodal(gated_model, test_loader, DEVICE)
    pred_tensor, _ = predict_multimodal(tensor_model, test_loader, DEVICE)
    pred_cross, _ = predict_multimodal(cross_model, test_loader, DEVICE)
    pred_text, _ = predict_text(text_model, test_loader, DEVICE)

    bimodal_preds = {}
    for name, (model, pair) in bimodal_models.items():
        pred_bi, _ = predict_bimodal_cross(model, test_loader, DEVICE, pair)
        bimodal_preds[name] = pred_bi

    # save raw predictions as csv
    pred_dict = {
        "y_true": y_true,
        "pred_text": pred_text,
        "pred_early": pred_early,
        "pred_gated": pred_gated,
        "pred_tensor": pred_tensor,
        "pred_cross": pred_cross,
    }
    for name, pred in bimodal_preds.items():
        safe_name = name.replace("+", "_").replace("-", "_")
        pred_dict[f"pred_{safe_name}"] = pred

    pred_df = pd.DataFrame(pred_dict)
    pred_df.to_csv(OUTPUT_PRED_CSV, index=False)
    print(f"Saved predictions to: {OUTPUT_PRED_CSV}")

    # save per-model npz files for pairwise comparison
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUTPUT_DIR, "text.npz"), y_true=y_true, y_pred=pred_text)
    np.savez(os.path.join(OUTPUT_DIR, "early.npz"), y_true=y_true, y_pred=pred_early)
    np.savez(os.path.join(OUTPUT_DIR, "gated.npz"), y_true=y_true, y_pred=pred_gated)
    np.savez(os.path.join(OUTPUT_DIR, "tensor.npz"), y_true=y_true, y_pred=pred_tensor)
    np.savez(os.path.join(OUTPUT_DIR, "cross.npz"), y_true=y_true, y_pred=pred_cross)
    if "text+audio" in bimodal_preds:
        np.savez(os.path.join(OUTPUT_DIR, "text_audio.npz"), y_true=y_true, y_pred=bimodal_preds["text+audio"])
    if "text+vision" in bimodal_preds:
        np.savez(os.path.join(OUTPUT_DIR, "text_vision.npz"), y_true=y_true, y_pred=bimodal_preds["text+vision"])
    if "audio+vision" in bimodal_preds:
        np.savez(os.path.join(OUTPUT_DIR, "audio_vision.npz"), y_true=y_true, y_pred=bimodal_preds["audio+vision"])
    print(f"Saved per-model prediction files to: {OUTPUT_DIR}")

    # metric summary
    summary = {
        "text": compute_metrics(y_true, pred_text),
        "early": compute_metrics(y_true, pred_early),
        "gated": compute_metrics(y_true, pred_gated),
        "tensor": compute_metrics(y_true, pred_tensor),
        "cross": compute_metrics(y_true, pred_cross),
    }
    for name, pred in bimodal_preds.items():
        summary[name] = compute_metrics(y_true, pred)

    print("\nMetric summary:")
    for name, m in summary.items():
        print(f"{name:>6} | MAE={m['MAE']:.4f} | RMSE={m['RMSE']:.4f} | Corr={m['Corr']:.4f}")

    # pairwise comparisons using bootstrap CI + permutation test on MAE
    comparisons = []
    if "text+audio" in bimodal_preds:
        comparisons.append(("text+audio", bimodal_preds["text+audio"], "cross", pred_cross))
    if "text+vision" in bimodal_preds:
        comparisons.append(("text+vision", bimodal_preds["text+vision"], "cross", pred_cross))
    if "audio+vision" in bimodal_preds:
        comparisons.append(("audio+vision", bimodal_preds["audio+vision"], "cross", pred_cross))

    results = []
    for name_a, pred_a, name_b, pred_b in comparisons:
        metrics_a = compute_metrics(y_true, pred_a)
        metrics_b = compute_metrics(y_true, pred_b)

        bootstrap_result, _ = bootstrap_mae_diff(
            y_true, pred_a, pred_b, n_boot=10000, seed=42
        )
        permutation_result, _ = permutation_test_mae(
            y_true, pred_a, pred_b, n_perm=10000, seed=42
        )

        result = {
            "model_a": name_a,
            "model_b": name_b,
            "metrics_a": metrics_a,
            "metrics_b": metrics_b,
            "interpretation": {
                "delta_mae_definition": "delta = MAE(model_a) - MAE(model_b); positive means model_b is better",
                "significant_by_bootstrap_ci": not (
                    bootstrap_result["ci_95_low"] <= 0 <= bootstrap_result["ci_95_high"]
                ),
            },
            "bootstrap_mae_diff": bootstrap_result,
            "permutation_test_mae": permutation_result,
        }
        results.append(result)

    with open(OUTPUT_COMPARE_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved bootstrap comparison results to: {OUTPUT_COMPARE_JSON}")
    print("\nBootstrap / permutation comparison results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()