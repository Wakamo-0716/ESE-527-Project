import argparse
import json
import numpy as np


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


def bootstrap_mae_diff(y_true, pred_a, pred_b, n_boot=10000, seed=42):
    """
    Returns distribution of delta = MAE_A - MAE_B
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
        "bootstrap_mean_delta": float(np.mean(deltas))
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
        "p_value_two_sided": p_value
    }, perm_stats


def load_predictions(npz_path):
    data = np.load(npz_path)
    y_true = data["y_true"].astype(np.float64)
    y_pred = data["y_pred"].astype(np.float64)
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, required=True, help="Path to model A prediction npz")
    parser.add_argument("--model_b", type=str, required=True, help="Path to model B prediction npz")
    parser.add_argument("--name_a", type=str, default="model_a")
    parser.add_argument("--name_b", type=str, default="model_b")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    y_true_a, pred_a = load_predictions(args.model_a)
    y_true_b, pred_b = load_predictions(args.model_b)

    if len(y_true_a) != len(y_true_b):
        raise ValueError("Prediction files have different numbers of samples.")

    if not np.allclose(y_true_a, y_true_b, atol=1e-6):
        raise ValueError("The y_true arrays do not match. Make sure both models are evaluated on the same test set in the same order.")

    y_true = y_true_a

    metrics_a = {
        "mae": mae(y_true, pred_a),
        "rmse": rmse(y_true, pred_a),
        "corr": corr(y_true, pred_a),
    }

    metrics_b = {
        "mae": mae(y_true, pred_b),
        "rmse": rmse(y_true, pred_b),
        "corr": corr(y_true, pred_b),
    }

    bootstrap_result, _ = bootstrap_mae_diff(
        y_true, pred_a, pred_b, n_boot=args.n_boot, seed=args.seed
    )

    permutation_result, _ = permutation_test_mae(
        y_true, pred_a, pred_b, n_perm=args.n_perm, seed=args.seed
    )

    result = {
        "model_a": args.name_a,
        "model_b": args.name_b,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "interpretation": {
            "delta_mae_definition": "delta = MAE(model_a) - MAE(model_b); positive means model_b is better",
            "significant_by_bootstrap_ci": not (
                bootstrap_result["ci_95_low"] <= 0 <= bootstrap_result["ci_95_high"]
            )
        },
        "bootstrap_mae_diff": bootstrap_result,
        "permutation_test_mae": permutation_result
    }

    print(json.dumps(result, indent=2))

    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved result to: {args.output}")


if __name__ == "__main__":
    main()