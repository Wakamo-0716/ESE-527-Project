import matplotlib.pyplot as plt
import numpy as np

# =========================
# Experimental Results
# =========================
multimodal_models = [
    "Early Fusion\nLSTM",
    "Gated Fusion\nLSTM",
    "Cross-modal Attention\nLSTM",
    "Tensor Fusion\nLSTM"
]

multimodal_mae = [0.6131, 0.6161, 0.6000, 0.6043]
multimodal_rmse = [0.8120, 0.8165, 0.7978, 0.8006]
multimodal_corr = [0.6846, 0.6828, 0.6994, 0.6953]

comparison_models = [
    "Text-only\nLSTM",
    "Audio-only\nLSTM",
    "Vision-only\nLSTM",
    "Cross-modal Attention\nLSTM"
]

comparison_mae = [0.6236, 0.8240, 0.8098, 0.6000]
comparison_rmse = [0.8256, 1.0706, 1.0650, 0.7978]
comparison_corr = [0.6672, 0.2620, 0.2821, 0.6994]

x_multi = np.arange(len(multimodal_models))
x_comp = np.arange(len(comparison_models))


def add_value_labels(ax, bars, fmt="{:.4f}", fontsize=9):
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.015
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize
        )


def style_axes(ax, ylabel, title, xticks, xticklabels):
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def apply_academic_bar_style(bars):
    for bar in bars:
        bar.set_linewidth(0.6)
        bar.set_edgecolor("0.35")


def set_metric_ylim(ax, values, higher_is_better=False):
    values = np.asarray(values)
    vmin = float(values.min())
    vmax = float(values.max())
    margin = max((vmax - vmin) * 0.25, 0.02)

    if higher_is_better:
        lower = max(0.0, vmin - margin)
        upper = min(1.0, vmax + margin)
    else:
        lower = max(0.0, vmin - margin)
        upper = vmax + margin

    ax.set_ylim(lower, upper)


def annotate_best(ax, bars, values, higher_is_better=False, label_prefix="Best"):
    best_idx = int(np.argmax(values)) if higher_is_better else int(np.argmin(values))
    best_bar = bars[best_idx]
    best_value = values[best_idx]

    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.06

    ax.annotate(
        f"{label_prefix}: {best_value:.4f}",
        xy=(best_bar.get_x() + best_bar.get_width() / 2, best_bar.get_height()),
        xytext=(best_bar.get_x() + best_bar.get_width() / 2, best_bar.get_height() + offset),
        ha="center",
        va="bottom",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", linewidth=0.8)
    )


# =========================
# Figures 1–3 Combined: Multimodal Results
# =========================
fig, axes = plt.subplots(3, 1, figsize=(10, 16))

# Panel A: Multimodal MAE
ax = axes[0]
bars = ax.bar(x_multi, multimodal_mae, width=0.65)
apply_academic_bar_style(bars)
set_metric_ylim(ax, multimodal_mae, higher_is_better=False)
style_axes(
    ax,
    ylabel="MAE",
    title="(a) Multimodal Fusion Strategies: Test MAE",
    xticks=x_multi,
    xticklabels=multimodal_models,
)
add_value_labels(ax, bars)
annotate_best(ax, bars, multimodal_mae, higher_is_better=False)

# Panel B: Multimodal RMSE
ax = axes[1]
bars = ax.bar(x_multi, multimodal_rmse, width=0.65)
apply_academic_bar_style(bars)
set_metric_ylim(ax, multimodal_rmse, higher_is_better=False)
style_axes(
    ax,
    ylabel="RMSE",
    title="(b) Multimodal Fusion Strategies: Test RMSE",
    xticks=x_multi,
    xticklabels=multimodal_models,
)
add_value_labels(ax, bars)
annotate_best(ax, bars, multimodal_rmse, higher_is_better=False)

# Panel C: Multimodal Correlation
ax = axes[2]
bars = ax.bar(x_multi, multimodal_corr, width=0.65)
apply_academic_bar_style(bars)
set_metric_ylim(ax, multimodal_corr, higher_is_better=True)
style_axes(
    ax,
    ylabel="Pearson Correlation",
    title="(c) Multimodal Fusion Strategies: Test Pearson Correlation",
    xticks=x_multi,
    xticklabels=multimodal_models,
)
add_value_labels(ax, bars)
annotate_best(ax, bars, multimodal_corr, higher_is_better=True)

plt.tight_layout(h_pad=2.0)
plt.savefig("figure_multimodal_combined.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# Figure 4: Unimodal vs Best Multimodal
# =========================
width = 0.24
fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x_comp - width, comparison_mae, width, label="MAE")
bars2 = ax.bar(x_comp, comparison_rmse, width, label="RMSE")
bars3 = ax.bar(x_comp + width, comparison_corr, width, label="Correlation")

apply_academic_bar_style(bars1)
apply_academic_bar_style(bars2)
apply_academic_bar_style(bars3)

style_axes(
    ax,
    ylabel="Metric Value",
    title="Unimodal Baselines vs. Best Multimodal Model",
    xticks=x_comp,
    xticklabels=comparison_models,
)
ax.legend(frameon=False, fontsize=10)

for bars in [bars1, bars2, bars3]:
    add_value_labels(ax, bars, fontsize=8)

best_x = x_comp[-1]
best_y = max(comparison_mae[-1], comparison_rmse[-1], comparison_corr[-1])
ymin, ymax = ax.get_ylim()
offset = (ymax - ymin) * 0.08
ax.annotate(
    "Best multimodal",
    xy=(best_x, best_y),
    xytext=(best_x, best_y + offset),
    ha="center",
    va="bottom",
    fontsize=10,
    arrowprops=dict(arrowstyle="->", linewidth=0.8)
)

plt.tight_layout()
plt.savefig("figure_unimodal_vs_best_multimodal_proj64.png", dpi=300, bbox_inches="tight")
plt.show()