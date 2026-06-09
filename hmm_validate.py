import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


def draw_parallel_step_lines(
    true: NDArray,
    predict: NDArray,
    ax: Axes = None,
    figsize: tuple = (12, 4),
    dpi: int = 100,
    true_label: str = "True",
    predict_label: str = "Predict",
    y_labels: list = None,
    subtitle: str = "HMM Preformance Comparision",
):

    is_standalone = ax is None
    if is_standalone:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    time_steps = np.arange(len(true))

    ax.step(
        time_steps,
        true,
        label=true_label,
        color="#1f77b4",
        linewidth=2.5,
        where="mid",
    )
    ax.step(
        time_steps,
        predict + 0.01,
        label=predict_label,
        color="#ff7f0e",
        linewidth=2,
        linestyle="--",
        where="mid",
    )
    if y_labels:
        ax.set_yticks(list(range(len(y_labels))))
        ax.set_yticklabels(y_labels)

    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("Hidden State")
    ax.set_title(subtitle)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    ax.figure.tight_layout()

    if is_standalone:
        plt.show()
        plt.close()


def draw_dual_track_color_blocks(
    true: NDArray,
    predict: NDArray,
    ax: Axes = None,
    figsize: tuple = (15, 3.2),
    dpi: int = 100,
    y_labels: list = None,
    subtitle: str = "Hidden State Sequence Alignment Track",
):

    heatmap_data = np.vstack([true, predict])

    is_standalone = ax is None
    if is_standalone:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Use Seaborn heatmap without colorbar, using distinct colors for states
    sns.heatmap(
        heatmap_data, cmap="YlGnBu", cbar=False, linewidths=0.5, linecolor="white"
    )

    # Adjust ticks and labels
    if y_labels:
        ax.set_yticks([i + 0.5 for i in range(len(y_labels))])
        ax.set_yticklabels(y_labels)
    ax.set_xlabel("Sequence Position (Base Pairs)")
    ax.set_title(subtitle)
    ax.figure.tight_layout()

    if is_standalone:
        plt.show()
        plt.close()


def draw_confusion_matrix(
    backgroud: NDArray,
    predict: NDArray,
    ax: Axes = None,
    figsize: tuple = (5, 4),
    dpi: int = 100,
    xticklabels: list = ["State 0", "State 1"],
    yticklabels: list = ["State 0", "State 1"],
    xlabel: str = "Prediction",
    ylabel: str = "Background",
):
    acc = accuracy_score(backgroud, predict)
    cm = confusion_matrix(backgroud, predict)

    is_standalone = ax is None
    if is_standalone:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=xticklabels,
        yticklabels=yticklabels,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Accuracy: {acc:.2%}")

    if is_standalone:
        plt.show()
        plt.close()
