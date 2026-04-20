import argparse
import csv
import os
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from datasets import MoseiDataset
from models.early_fusion import EarlyFusionLSTM
from models.gated_fusion import GatedFusionLSTM
from models.cross_modal_attention import CrossModalAttentionLSTM
from models.tensor_fusion import TensorFusionLSTM


def build_model(
    name: str,
    proj_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.2,
):
    common_kwargs = {
        "proj_dim": proj_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
    }

    if name == "early":
        return EarlyFusionLSTM(**common_kwargs)
    if name == "gated":
        return GatedFusionLSTM(**common_kwargs)
    if name == "cross_attn":
        return CrossModalAttentionLSTM(**common_kwargs)
    if name == "tensor":
        return TensorFusionLSTM(**common_kwargs)
    raise ValueError(f"Unknown model: {name}")


def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        preds = model(text, audio, vision)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)



def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            labels = batch["label"].to(device)

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
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "corr": float(corr),
    }


def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            labels = batch["label"].to(device)

            preds = model(text, audio, vision)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_preds = torch.cat(all_preds).numpy().reshape(-1)
    all_labels = torch.cat(all_labels).numpy().reshape(-1)
    return all_preds, all_labels



def make_loaders(data_dir: str, batch_size: int, num_workers: int = 0):
    train_path = os.path.join(data_dir, "train_processed.npz")
    valid_path = os.path.join(data_dir, "valid_processed.npz")
    test_path = os.path.join(data_dir, "test_processed.npz")

    train_dataset = MoseiDataset(train_path)
    valid_dataset = MoseiDataset(valid_path)
    test_dataset = MoseiDataset(test_path)

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



def save_checkpoint(model, optimizer, epoch, metrics, save_path: str):
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



def parse_int_list(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]



def parse_float_list(raw: str):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]



def parse_str_list(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()]



def parse_args():
    parser = argparse.ArgumentParser(description="Train multimodal sentiment regression models on CMU-MOSEI")
    parser.add_argument("--model", type=str, default="early", choices=["early", "gated", "cross_attn", "tensor"])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Grid search mode
    parser.add_argument("--search", action="store_true", help="Run grid search instead of a single training run")
    parser.add_argument("--models", type=str, default="cross_attn", help="Comma-separated models for grid search")
    parser.add_argument("--batch_size_list", type=str, default="32", help="Comma-separated batch sizes")
    parser.add_argument("--lr_list", type=str, default="1e-3", help="Comma-separated learning rates")
    parser.add_argument("--weight_decay_list", type=str, default="1e-5", help="Comma-separated weight decays")
    parser.add_argument("--proj_dim_list", type=str, default="64", help="Comma-separated projection dimensions")
    parser.add_argument("--hidden_dim_list", type=str, default="128", help="Comma-separated hidden dimensions")
    parser.add_argument("--num_layers_list", type=str, default="1", help="Comma-separated layer counts")
    parser.add_argument("--dropout_list", type=str, default="0.2", help="Comma-separated dropout values")
    parser.add_argument("--results_csv", type=str, default="search_results.csv", help="CSV file for search results")
    parser.add_argument("--save_predictions", action="store_true", help="Save test-set predictions after training")
    parser.add_argument("--pred_dir", type=str, default="demo_outputs", help="Directory for saved prediction files")
    return parser.parse_args()



def run_experiment(args, device, trial_config, trial_id=None):
    model_name = trial_config["model"]
    batch_size = trial_config["batch_size"]
    lr = trial_config["lr"]
    weight_decay = trial_config["weight_decay"]
    proj_dim = trial_config["proj_dim"]
    hidden_dim = trial_config["hidden_dim"]
    num_layers = trial_config["num_layers"]
    dropout = trial_config["dropout"]

    print("=" * 80)
    if trial_id is not None:
        print(f"Trial {trial_id}")
    print(
        f"model={model_name}, batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}, "
        f"proj_dim={proj_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}"
    )

    train_loader, valid_loader, test_loader = make_loaders(
        data_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(
        model_name,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    ckpt_name = (
        f"best_{model_name}_bs{batch_size}_lr{lr}_wd{weight_decay}_pd{proj_dim}"
        f"_hd{hidden_dim}_nl{num_layers}_do{dropout}.pt"
    )
    best_ckpt_path = os.path.join(args.save_dir, ckpt_name)

    best_val_rmse = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, valid_loader, criterion, device)

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
            save_checkpoint(model, optimizer, epoch, val_metrics, best_ckpt_path)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if Path(best_ckpt_path).exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device)

    if args.save_predictions:
        preds, labels = collect_predictions(model, test_loader, device)
        pred_subdir = os.path.join(args.pred_dir, f"lstm_{model_name}")
        os.makedirs(pred_subdir, exist_ok=True)
        np.savez(
            os.path.join(pred_subdir, f"pred_{model_name}.npz"),
            preds=preds,
            labels=labels,
        )
        with open(os.path.join(pred_subdir, f"result_{model_name}.json"), "w") as f:
            import json
            json.dump(
                {
                    "mae": test_metrics["mae"],
                    "rmse": test_metrics["rmse"],
                    "corr": test_metrics["corr"],
                    "best_epoch": best_epoch,
                    "best_val_rmse": best_val_rmse,
                },
                f,
                indent=2,
            )
        print(f"Saved predictions to: {pred_subdir}")

    result = {
        "model": model_name,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "proj_dim": proj_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "test_loss": test_metrics["loss"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_corr": test_metrics["corr"],
    }

    print(
        f"Best validation RMSE: {best_val_rmse:.4f} at epoch {best_epoch}\n"
        f"Test Results | Loss: {test_metrics['loss']:.4f} | MAE: {test_metrics['mae']:.4f} | "
        f"RMSE: {test_metrics['rmse']:.4f} | Corr: {test_metrics['corr']:.4f}"
    )
    return result



def run_grid_search(args, device):
    models = parse_str_list(args.models)
    batch_sizes = parse_int_list(args.batch_size_list)
    lrs = parse_float_list(args.lr_list)
    weight_decays = parse_float_list(args.weight_decay_list)
    proj_dims = parse_int_list(args.proj_dim_list)
    hidden_dims = parse_int_list(args.hidden_dim_list)
    num_layers_list = parse_int_list(args.num_layers_list)
    dropouts = parse_float_list(args.dropout_list)

    all_trials = list(product(models, batch_sizes, lrs, weight_decays, proj_dims, hidden_dims, num_layers_list, dropouts))
    print(f"Total trials: {len(all_trials)}")

    results = []
    for idx, (model_name, batch_size, lr, weight_decay, proj_dim, hidden_dim, num_layers, dropout) in enumerate(all_trials, start=1):
        seed_everything(args.seed)
        trial_config = {
            "model": model_name,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "proj_dim": proj_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        result = run_experiment(args, device, trial_config, trial_id=idx)
        results.append(result)

    results = sorted(results, key=lambda x: x["best_val_rmse"])

    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
    with open(args.results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\nTop 5 trials by best validation RMSE:")
    for row in results[:5]:
        print(row)
    print(f"\nSaved search results to: {args.results_csv}")



def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.search:
        run_grid_search(args, device)
        return

    print(f"Model: {args.model}")
    trial_config = {
        "model": args.model,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "proj_dim": args.proj_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    run_experiment(args, device, trial_config)


if __name__ == "__main__":
    main()
