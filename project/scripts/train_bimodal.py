import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from datasets import MoseiDataset
from models.cross_modal_attention import CrossModalAttentionLSTM


def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device, pair):
    model.train()
    total_loss = 0.0

    for batch in loader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        labels = batch["label"].to(device)

        # zero out the unused modality
        if pair == "ta":
            vision = torch.zeros_like(vision)
        elif pair == "tv":
            audio = torch.zeros_like(audio)
        elif pair == "av":
            text = torch.zeros_like(text)
        else:
            raise ValueError(f"Unknown pair: {pair}")

        optimizer.zero_grad()
        preds = model(text, audio, vision)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, pair):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
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
                raise ValueError(f"Unknown pair: {pair}")

            preds = model(text, audio, vision)
            loss = criterion(preds, labels)

            total_loss += loss.item() * labels.size(0)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    mse = np.mean((all_preds - all_labels) ** 2)
    mae = np.mean(np.abs(all_preds - all_labels))
    rmse = np.sqrt(mse)

    if len(all_preds) > 1 and np.std(all_preds) > 0 and np.std(all_labels) > 0:
        corr = pearsonr(all_preds, all_labels)[0]
    else:
        corr = 0.0

    return {
        "loss": total_loss / len(loader.dataset),
        "mae": float(mae),
        "rmse": float(rmse),
        "corr": float(corr),
    }


def make_loaders(data_dir, batch_size, num_workers=0):
    train_dataset = MoseiDataset(os.path.join(data_dir, "train_processed.npz"))
    valid_dataset = MoseiDataset(os.path.join(data_dir, "valid_processed.npz"))
    test_dataset = MoseiDataset(os.path.join(data_dir, "test_processed.npz"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, valid_loader, test_loader


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        save_path,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, required=True, choices=["ta", "tv", "av"])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--save_dir", type=str, default="checkpoints_bimodal_cross")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)

    # fixed best hyperparameters
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Bi-modal pair: {args.pair}")

    train_loader, valid_loader, test_loader = make_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    model = CrossModalAttentionLSTM(
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ckpt_path = os.path.join(args.save_dir, f"best_cross_{args.pair}.pt")

    best_val_rmse = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.pair)
        val_metrics = evaluate(model, valid_loader, criterion, device, args.pair)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"Val Corr: {val_metrics['corr']:.4f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_metrics, ckpt_path)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"Best validation RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")

    if Path(ckpt_path).exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device, args.pair)
    print(
        f"Test Results | "
        f"Loss: {test_metrics['loss']:.4f} | "
        f"MAE: {test_metrics['mae']:.4f} | "
        f"RMSE: {test_metrics['rmse']:.4f} | "
        f"Corr: {test_metrics['corr']:.4f}"
    )


if __name__ == "__main__":
    main()