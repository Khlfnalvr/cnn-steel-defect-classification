"""Training + evaluasi satu model CNN pada dataset NEU (modul pustaka).

Dipanggil oleh train.py di folder utama lewat train_model(). Tidak punya CLI sendiri:
untuk melatih satu model gunakan `python train.py --models alexnet`.
"""
import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from data import CLASSES, load_images, make_ds  # noqa: E402
from models import MODELS  # noqa: E402


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


class Tee:
    """Tulis keluaran ke layar sekaligus ke berkas log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


class BestWeights(tf.keras.callbacks.Callback):
    """Simpan bobot dengan val_loss terendah dan pulihkan di akhir training (tanpa menghentikan lebih awal)."""

    def __init__(self):
        super().__init__(); self.best, self.weights, self.epoch = np.inf, None, 0

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.best:
            self.best, self.weights, self.epoch = logs["val_loss"], self.model.get_weights(), epoch + 1

    def on_train_end(self, logs=None):
        if self.weights is not None:
            self.model.set_weights(self.weights)
            print(f"Bobot terbaik dipulihkan dari epoch {self.epoch} (val_loss = {self.best:.4f})")


def train_model(name, df, out_root="results", epochs=100, batch=32, lr=1e-3, seed=42, patience=0):
    """Latih model `name` pada split df, simpan semua hasil ke out_root/<name>/ dan log_<name>.txt.
    patience > 0 mengaktifkan EarlyStopping; 0 = selalu menjalankan semua epoch."""
    out_root = Path(out_root)
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out_root / f"log_{name}.txt", "w", encoding="utf-8") as logf, \
            contextlib.redirect_stdout(Tee(sys.stdout, logf)):
        return _train(name, df, out, epochs, batch, lr, seed, patience)


def _train(name, df, out, epochs, batch, lr, seed, patience):
    set_seed(seed)
    build, size = MODELS[name]
    xs, ys = {}, {}
    for s in ("train", "val", "test"):
        xs[s], ys[s] = load_images(df[df.split == s], size)
    print({s: xs[s].shape for s in xs})

    model = build(input_size=size, n_classes=len(CLASSES))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    with open(out / "summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))

    cb = [
        BestWeights(),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
        tf.keras.callbacks.CSVLogger(str(out / "history.csv")),
    ]
    if patience:
        cb.insert(0, tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience))
    t0 = time.time()
    model.fit(make_ds(xs["train"], ys["train"], batch, shuffle=True, seed=seed),
              validation_data=make_ds(xs["val"], ys["val"], batch),
              epochs=epochs, callbacks=cb, verbose=2)
    train_time = time.time() - t0
    model.save(out / "model.keras")

    # ---- evaluasi pada data uji (tidak pernah dilihat saat training) ----
    t0 = time.time()
    prob = model.predict(xs["test"], verbose=0)
    infer_ms = (time.time() - t0) / len(xs["test"]) * 1000
    pred = prob.argmax(1)
    test_loss, test_acc = model.evaluate(make_ds(xs["test"], ys["test"]), verbose=0)
    rep = classification_report(ys["test"], pred, target_names=CLASSES, output_dict=True, digits=4)
    cm = confusion_matrix(ys["test"], pred)

    np.save(out / "test_prob.npy", prob)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / "confusion_matrix.csv")
    pd.DataFrame(rep).T.to_csv(out / "classification_report.csv")
    metrics = {
        "model": model.name, "input_size": size, "params": int(model.count_params()),
        "epochs_run": len(pd.read_csv(out / "history.csv")),
        "train_time_s": round(train_time, 1), "infer_ms_per_img": round(infer_ms, 2),
        "test_loss": round(float(test_loss), 4), "test_acc": round(float(test_acc), 4),
        "macro_precision": round(rep["macro avg"]["precision"], 4),
        "macro_recall": round(rep["macro avg"]["recall"], 4),
        "macro_f1": round(rep["macro avg"]["f1-score"], 4),
    }
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    print(classification_report(ys["test"], pred, target_names=CLASSES, digits=4))
    return metrics
