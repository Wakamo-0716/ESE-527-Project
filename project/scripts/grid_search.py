import os
import json
import csv
import itertools
import subprocess
import argparse
from pathlib import Path


def run_command(cmd):
    print("=" * 100)
    print("RUNNING:", " ".join(cmd))
    print("=" * 100)
    result = subprocess.run(cmd, check=False)
    return result.returncode


def load_result_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_exec", type=str, default="python")
    parser.add_argument("--train_script", type=str, default="scripts/train_transformer.py")
    parser.add_argument("--output_root", type=str, default="grid_outputs")
    parser.add_argument("--summary_csv", type=str, default="grid_outputs/summary.csv")

    parser.add_argument(
        "--models",
        nargs="+",
        default=["gated", "cross"],
        choices=["text", "audio", "vision", "early", "gated", "cross", "tensor"]
    )

    parser.add_argument("--epochs", nargs="+", type=int, default=[20])
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4])
    parser.add_argument("--n_heads", nargs="+", type=int, default=[4])
    parser.add_argument("--n_layers", nargs="+", type=int, default=[2])
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    # 不同模型可用不同搜索空间
    for model_type in args.models:
        if model_type == "cross":
            search_space = itertools.product(
                args.epochs,
                args.hidden_dims,
                args.dropouts,
                args.lrs,
                args.n_heads,
                args.n_layers,
                args.batch_sizes,
            )
        else:
            # early / gated / text 等模型通常不用特别展开 n_heads/n_layers 太多
            search_space = itertools.product(
                args.epochs,
                args.hidden_dims,
                args.dropouts,
                args.lrs,
                [args.n_heads[0]],
                [args.n_layers[0]],
                args.batch_sizes,
            )

        for (
            epochs,
            hidden_dim,
            dropout,
            lr,
            n_heads,
            n_layers,
            batch_size,
        ) in search_space:

            run_name = (
                f"{model_type}"
                f"_hd{hidden_dim}"
                f"_do{dropout}"
                f"_lr{lr}"
                f"_nh{n_heads}"
                f"_nl{n_layers}"
                f"_bs{batch_size}"
                f"_ep{epochs}"
                f"_sd{args.seed}"
            )

            run_output_dir = output_root / run_name
            run_output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                args.python_exec,
                args.train_script,
                "--model_type", model_type,
                "--hidden_dim", str(hidden_dim),
                "--dropout", str(dropout),
                "--lr", str(lr),
                "--n_heads", str(n_heads),
                "--n_layers", str(n_layers),
                "--batch_size", str(batch_size),
                "--epochs", str(epochs),
                "--weight_decay", str(args.weight_decay),
                "--seed", str(args.seed),
                "--output_dir", str(run_output_dir),
            ]

            if args.cpu:
                cmd.append("--cpu")

            if args.dry_run:
                print("DRY RUN:", " ".join(cmd))
                continue

            return_code = run_command(cmd)

            result_json_path = run_output_dir / f"result_{model_type}.json"

            row = {
                "run_name": run_name,
                "model_type": model_type,
                "epochs": epochs,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "lr": lr,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "batch_size": batch_size,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "return_code": return_code,
                "best_valid_mae": None,
                "test_mae": None,
                "test_rmse": None,
                "test_corr": None,
                "result_json": str(result_json_path),
            }

            if return_code == 0 and result_json_path.exists():
                try:
                    result = load_result_json(result_json_path)
                    row["best_valid_mae"] = result.get("best_valid_mae", None)

                    test_metrics = result.get("test_metrics", {})
                    row["test_mae"] = test_metrics.get("mae", None)
                    row["test_rmse"] = test_metrics.get("rmse", None)
                    row["test_corr"] = test_metrics.get("corr", None)
                except Exception as e:
                    print(f"Failed to read result json for {run_name}: {e}")
            else:
                print(f"Run failed or missing result file: {run_name}")

            summary_rows.append(row)

            # 每跑完一个就立刻刷新 summary.csv，避免中途断掉丢结果
            summary_csv_path = Path(args.summary_csv)
            summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

            fieldnames = [
                "run_name",
                "model_type",
                "epochs",
                "hidden_dim",
                "dropout",
                "lr",
                "n_heads",
                "n_layers",
                "batch_size",
                "weight_decay",
                "seed",
                "return_code",
                "best_valid_mae",
                "test_mae",
                "test_rmse",
                "test_corr",
                "result_json",
            ]

            with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)

    print("\nFinished all runs.")
    print(f"Summary saved to: {args.summary_csv}")


if __name__ == "__main__":
    main()