import os
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models import (
    UnimodalTransformerModel,
    EarlyFusionModel,
    GatedFusionModel,
    CrossModalAttentionModel,
    TensorFusionModel
)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_corr(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


class MOSEILikeDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)

        # 你这里如果键名不同，改这里即可
        self.text = data["text"].astype(np.float32)
        self.audio = data["audio"].astype(np.float32)
        self.vision = data["vision"].astype(np.float32)

        if "labels" in data:
            self.labels = data["labels"].astype(np.float32)
        else:
            self.labels = data["y"].astype(np.float32)

        self.text_mask = data["text_mask"].astype(np.int64) if "text_mask" in data else None
        self.audio_mask = data["audio_mask"].astype(np.int64) if "audio_mask" in data else None
        self.vision_mask = data["vision_mask"].astype(np.int64) if "vision_mask" in data else None

        # 若没有单独 mask，且三个模态已经按时间对齐，可用全1 mask
        if self.text_mask is None:
            self.text_mask = np.ones(self.text.shape[:2], dtype=np.int64)
        if self.audio_mask is None:
            self.audio_mask = np.ones(self.audio.shape[:2], dtype=np.int64)
        if self.vision_mask is None:
            self.vision_mask = np.ones(self.vision.shape[:2], dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        y = self.labels[idx]
        if np.ndim(y) > 0:
            y = np.squeeze(y)

        return {
            "text": torch.tensor(self.text[idx], dtype=torch.float32),
            "audio": torch.tensor(self.audio[idx], dtype=torch.float32),
            "vision": torch.tensor(self.vision[idx], dtype=torch.float32),
            "text_mask": torch.tensor(self.text_mask[idx], dtype=torch.long),
            "audio_mask": torch.tensor(self.audio_mask[idx], dtype=torch.long),
            "vision_mask": torch.tensor(self.vision_mask[idx], dtype=torch.long),
            "label": torch.tensor(y, dtype=torch.float32),
        }


def build_model(args, text_dim, audio_dim, vision_dim):
    modalities = [m.strip() for m in args.modalities.split(",")]

    if args.model_type == "text":
        return UnimodalTransformerModel(
            input_dim=text_dim,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout
        )
    elif args.model_type == "audio":
        return UnimodalTransformerModel(
            input_dim=audio_dim,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout
        )
    elif args.model_type == "vision":
        return UnimodalTransformerModel(
            input_dim=vision_dim,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout
        )
    elif args.model_type == "early":
        return EarlyFusionModel(
            text_dim=text_dim,
            audio_dim=audio_dim,
            vision_dim=vision_dim,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            modalities=modalities
        )
    elif args.model_type == "gated":
        return GatedFusionModel(
            text_dim=text_dim,
            audio_dim=audio_dim,
            vision_dim=vision_dim,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            modalities=modalities
        )
    elif args.model_type == "cross":
        return CrossModalAttentionModel(
            text_dim=text_dim,
            audio_dim=audio_dim,
            vision_dim=vision_dim,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            modalities=modalities
        )
    elif args.model_type == "tensor":
        return TensorFusionModel(
            text_dim=text_dim,
            audio_dim=audio_dim,
            vision_dim=vision_dim,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")


def forward_batch(model, batch, model_type, device):
    text = batch["text"].to(device)
    audio = batch["audio"].to(device)
    vision = batch["vision"].to(device)
    text_mask = batch["text_mask"].to(device)
    audio_mask = batch["audio_mask"].to(device)
    vision_mask = batch["vision_mask"].to(device)

    if model_type == "text":
        preds = model(text, text_mask)
    elif model_type == "audio":
        preds = model(audio, audio_mask)
    elif model_type == "vision":
        preds = model(vision, vision_mask)
    elif model_type == "early":
        # 假设已时间对齐，这里用 text_mask 作为统一 mask
        preds = model(text, audio, vision, mask=text_mask)
    else:
        preds = model(
            text, audio, vision,
            text_mask=text_mask,
            audio_mask=audio_mask,
            vision_mask=vision_mask
        )
    return preds


def evaluate(model, loader, criterion, model_type, device, return_preds=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            preds = forward_batch(model, batch, model_type, device)

            loss = criterion(preds, labels)
            total_loss += loss.item() * labels.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    corr = compute_corr(all_labels, all_preds)

    avg_loss = total_loss / len(loader.dataset)

    metrics = {
        "loss": float(avg_loss),
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "corr": float(corr)
    }

    if return_preds:
        return metrics, all_labels, all_preds
    return metrics


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    train_set = MOSEILikeDataset(args.train_path)
    valid_set = MOSEILikeDataset(args.valid_path)
    test_set = MOSEILikeDataset(args.test_path)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    text_dim = train_set.text.shape[-1]
    audio_dim = train_set.audio.shape[-1]
    vision_dim = train_set.vision.shape[-1]

    print(f"text_dim={text_dim}, audio_dim={audio_dim}, vision_dim={vision_dim}")

    model = build_model(args, text_dim, audio_dim, vision_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid_mae = float("inf")
    best_ckpt_path = Path(args.output_dir) / f"best_{args.model_type}.pt"
    os.makedirs(args.output_dir, exist_ok=True)

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            preds = forward_batch(model, batch, args.model_type, device)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item() * labels.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)
        valid_metrics = evaluate(model, valid_loader, criterion, args.model_type, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_mse={train_loss:.4f} | "
            f"valid_mse={valid_metrics['mse']:.4f} | "
            f"valid_mae={valid_metrics['mae']:.4f} | "
            f"valid_rmse={valid_metrics['rmse']:.4f} | "
            f"valid_corr={valid_metrics['corr']:.4f}"
        )

        history.append({
            "epoch": int(epoch),
            "train_mse": float(train_loss),
            **{k: float(v) for k, v in valid_metrics.items()}
        })

        if valid_metrics["mae"] < best_valid_mae:
            best_valid_mae = valid_metrics["mae"]
            torch.save(model.state_dict(), best_ckpt_path)

    print(f"\nBest model saved to: {best_ckpt_path}")

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_metrics, test_labels, test_preds = evaluate(
        model, test_loader, criterion, args.model_type, device, return_preds=True
    )

    print("\nTest Results:")
    print(json.dumps(test_metrics, indent=2))

    result = {
    "model_type": args.model_type,
    "hidden_dim": int(args.hidden_dim),
    "n_heads": int(args.n_heads),
    "n_layers": int(args.n_layers),
    "dropout": float(args.dropout),
    "lr": float(args.lr),
    "batch_size": int(args.batch_size),
    "epochs": int(args.epochs),
    "best_valid_mae": float(best_valid_mae),
    "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    "history": history
    }

    result_path = Path(args.output_dir) / f"result_{args.model_type}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved result to: {result_path}")

    pred_path = Path(args.output_dir) / f"pred_{args.model_type}.npz"
    np.savez(
        pred_path,
        y_true=test_labels.astype(np.float32),
        y_pred=test_preds.astype(np.float32)
    )
    print(f"Saved predictions to: {pred_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="data/processed/train_processed.npz")
    parser.add_argument("--valid_path", type=str, default="data/processed/valid_processed.npz")
    parser.add_argument("--test_path", type=str, default="data/processed/test_processed.npz")
    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["text", "audio", "vision", "early", "gated", "cross", "tensor"]
    )

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--modalities", type=str, default="text,audio,vision")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)