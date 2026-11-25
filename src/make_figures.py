"""Membuat semua gambar laporan dari hasil training di folder results/.

    python src/make_figures.py --out results
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from data import CLASSES, load_images  # noqa: E402

COL = {"lenet5": "#eb6834", "alexnet": "#2f9e6b", "minivgg": "#2a78d6"}
NAME = {"lenet5": "LeNet-5", "alexnet": "AlexNet", "minivgg": "Mini-VGG (VGGNet)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#a9c8ef", "#2a78d6", "#0d3a73"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "savefig.dpi": 200,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})
SHORT = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled-in", "Scratches"]


def samples(df, fig_dir):
    fig, ax = plt.subplots(3, 6, figsize=(12, 6.4))
    for j, c in enumerate(CLASSES):
        paths = df[df.label == c].path.iloc[:3]
        for i, p in enumerate(paths):
            ax[i, j].imshow(Image.open(p), cmap="gray", vmin=0, vmax=255)
            ax[i, j].axis("off")
        ax[0, j].set_title(c.replace("_", " "), fontsize=12, color=INK)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(fig_dir / "dataset_samples.png"); plt.close(fig)


def curves(res, models, fig_dir):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in models:
        h = pd.read_csv(res / m / "history.csv")
        e = h.epoch + 1
        ax[0].plot(e, h.accuracy, color=COL[m], lw=2, alpha=0.45, label=f"{NAME[m]} – train")
        ax[0].plot(e, h.val_accuracy, color=COL[m], lw=2, label=f"{NAME[m]} – validasi")
        ax[1].plot(e, h.loss, color=COL[m], lw=2, alpha=0.45, label=f"{NAME[m]} – train")
        ax[1].plot(e, h.val_loss, color=COL[m], lw=2, label=f"{NAME[m]} – validasi")
    ax[0].set(xlabel="Epoch", ylabel="Akurasi", ylim=(0, 1.02), title="Akurasi")
    ax[1].set(xlabel="Epoch", ylabel="Loss (cross-entropy)", ylim=(0, 2.0), title="Loss")
    for a in ax:
        a.title.set_color(INK); a.legend(frameon=False, fontsize=9)
    fig.savefig(fig_dir / "training_curves.png"); plt.close(fig)
    for m in models:  # kurva per model
        h = pd.read_csv(res / m / "history.csv"); e = h.epoch + 1
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(e, h.accuracy, color=COL[m], lw=2, alpha=0.45, label="train")
        ax[0].plot(e, h.val_accuracy, color=COL[m], lw=2, label="validasi")
        ax[1].plot(e, h.loss, color=COL[m], lw=2, alpha=0.45, label="train")
        ax[1].plot(e, h.val_loss, color=COL[m], lw=2, label="validasi")
        ax[0].set(xlabel="Epoch", ylabel="Akurasi", ylim=(0, 1.02))
        ax[1].set(xlabel="Epoch", ylabel="Loss", ylim=(0, min(2.0, h[["loss", "val_loss"]].max().max() * 1.05)))
        for a in ax: a.legend(frameon=False)
        fig.savefig(fig_dir / f"curves_{m}.png"); plt.close(fig)


def confmats(res, models, fig_dir):
    for m in models:
        cm = pd.read_csv(res / m / "confusion_matrix.csv", index_col=0).values
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        ax.imshow(cm, cmap=BLUES, vmin=0, vmax=cm.sum(1).max()); ax.grid(False)
        for i in range(6):
            for j in range(6):
                v = cm[i, j]
                ax.text(j, i, v, ha="center", va="center", fontsize=12,
                        color="white" if v > cm.max() * 0.55 else INK)
        ax.set_xticks(range(6), SHORT, rotation=35, ha="right")
        ax.set_yticks(range(6), SHORT)
        ax.set(xlabel="Prediksi", ylabel="Label sebenarnya")
        for s in ax.spines.values(): s.set_visible(False)
        fig.savefig(fig_dir / f"cm_{m}.png"); plt.close(fig)


def f1_bars(res, models, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    n = len(models); w = 0.8 / n; x = np.arange(6)
    for k, m in enumerate(models):
        r = pd.read_csv(res / m / "classification_report.csv", index_col=0)
        f1 = r.loc[CLASSES, "f1-score"].values
        b = ax.bar(x + (k - (n - 1) / 2) * w, f1, w * 0.94, color=COL[m], label=NAME[m])
        ax.bar_label(b, labels=[f"{v:.2f}" for v in f1], fontsize=8.5, color=MUTED, padding=2)
    ax.set_xticks(x, SHORT); ax.set_ylim(0, 1.1); ax.set_ylabel("F1-score (data uji)")
    ax.grid(axis="x", visible=False); ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=n)
    fig.savefig(fig_dir / "f1_per_class.png"); plt.close(fig)


def feature_maps(res, df, fig_dir):
    from tensorflow import keras
    model = keras.models.load_model(res / "minivgg" / "model.keras")
    row = df[(df.split == "test") & (df.label == "Scratches")].iloc[0]
    x, _ = load_images(row.to_frame().T, 64)
    names = ["block1_relu1", "block2_relu2", "block3_relu2"]
    sub = keras.Model(model.input, [model.get_layer(n).output for n in names])
    outs = sub(x, training=False)
    fig, ax = plt.subplots(3, 7, figsize=(12, 5.4))
    for r, (n, o) in enumerate(zip(names, outs)):
        o = o.numpy()[0]
        ax[r, 0].imshow(x[0, ..., 0], cmap="gray"); ax[r, 0].axis("off")
        ax[r, 0].set_title("input" if r == 0 else "", fontsize=10)
        idx = np.argsort(o.mean((0, 1)))[::-1][:6]
        for k, c in enumerate(idx):
            ax[r, k + 1].imshow(o[..., c], cmap="viridis"); ax[r, k + 1].axis("off")
        ax[r, 0].text(-0.06, 0.5, f"{n.split('_')[0]}\n{o.shape[0]}×{o.shape[1]}×{o.shape[2]}",
                      transform=ax[r, 0].transAxes, ha="right", va="center", fontsize=10, color=MUTED)
    plt.subplots_adjust(wspace=0.06, hspace=0.12)
    fig.savefig(fig_dir / "feature_maps.png"); plt.close(fig)


def predictions(res, df, fig_dir):
    te = df[df.split == "test"].reset_index(drop=True)
    prob = np.load(res / "minivgg" / "test_prob.npy"); pred = prob.argmax(1)
    rng = np.random.default_rng(1)
    idx = np.concatenate([rng.choice(np.where(te.y == c)[0], 2, replace=False) for c in range(6)])
    wrong = np.where(pred != te.y.values)[0]
    fig, ax = plt.subplots(2, 6, figsize=(12, 4.6))
    for a, i in zip(ax.flat, idx):
        a.imshow(Image.open(te.path[i]), cmap="gray"); a.axis("off")
        ok = pred[i] == te.y[i]
        a.set_title(f"{SHORT[pred[i]]} ({prob[i].max()*100:.0f}%)\nasli: {SHORT[te.y[i]]}",
                    fontsize=9.5, color="#006300" if ok else "#c62828")
    fig.savefig(fig_dir / "pred_samples.png"); plt.close(fig)
    if len(wrong):
        n = min(len(wrong), 6)
        fig, ax = plt.subplots(1, n, figsize=(2 * n, 2.5), squeeze=False)
        for a, i in zip(ax[0], wrong[:n]):
            a.imshow(Image.open(te.path[i]), cmap="gray"); a.axis("off")
            a.set_title(f"pred: {SHORT[pred[i]]}\nasli: {SHORT[te.y[i]]}", fontsize=9.5, color="#c62828")
        fig.savefig(fig_dir / "misclassified.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    res = Path(a.out); fig_dir = res / "figures"; fig_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(res / "split.csv")
    models = [m for m in ("lenet5", "alexnet", "minivgg") if (res / m / "metrics.json").exists()]
    samples(df, fig_dir); curves(res, models, fig_dir); confmats(res, models, fig_dir)
    f1_bars(res, models, fig_dir)
    if "minivgg" in models:
        feature_maps(res, df, fig_dir); predictions(res, df, fig_dir)
    summ = pd.DataFrame([json.load(open(res / m / "metrics.json")) for m in models])
    summ.to_csv(res / "comparison.csv", index=False); print(summ.T)


if __name__ == "__main__":
    main()
