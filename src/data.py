"""Persiapan dataset NEU Surface Defect (NEU-CLS).

Dataset: 1800 citra 200x200 px, 6 kelas cacat permukaan baja hot-rolled
(300 citra/kelas). Sumber asli: K. Song & Y. Yan (2013), Northeastern University.

Label diambil dari awalan nama file (crazing_10.jpg, Cr_18.bmp, ...). Berkas label
.txt versi deteksi (YOLO) tidak dipakai karena berisi kotak cacat, bukan label
kelas citra. Semua citra dari semua subfolder (train/valid dsb.) digabung lalu
dibagi ulang secara stratified 70/15/15.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

CLASSES = ["Crazing", "Inclusion", "Patches", "Pitted_Surface", "Rolled-in_Scale", "Scratches"]
# awalan nama file (huruf kecil) -> nama kelas: nama panjang (crazing_1.jpg) dan singkatan (Cr_1.bmp)
PREFIX = {"crazing": "Crazing", "inclusion": "Inclusion", "patches": "Patches",
          "pitted_surface": "Pitted_Surface", "rolled-in_scale": "Rolled-in_Scale", "scratches": "Scratches",
          "cr": "Crazing", "in": "Inclusion", "pa": "Patches", "ps": "Pitted_Surface",
          "rs": "Rolled-in_Scale", "sc": "Scratches"}
EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def label_of(path):
    """crazing_10.jpg -> Crazing, PS_3.bmp -> Pitted_Surface."""
    prefix = Path(path).stem.rsplit("_", 1)[0].lower()
    if prefix not in PREFIX:
        raise ValueError(f"Awalan nama file tidak dikenal: {path}")
    return PREFIX[prefix]


def build_split(raw_dir, out_csv, seed=42, val=0.15, test=0.15):
    """Kumpulkan semua citra lalu bagi stratified 70/15/15."""
    files = sorted(f for f in Path(raw_dir).rglob("*") if f.suffix.lower() in EXT)
    if not files:
        raise FileNotFoundError(f"Tidak ada citra (.jpg/.png/.bmp) di {raw_dir}")
    rows = [{"path": f.as_posix(), "label": label_of(f)} for f in files]
    df = pd.DataFrame(rows).drop_duplicates("path")
    df["y"] = df["label"].map({c: i for i, c in enumerate(CLASSES)})
    tr, tmp = train_test_split(df, test_size=val + test, stratify=df.y, random_state=seed)
    va, te = train_test_split(tmp, test_size=test / (val + test), stratify=tmp.y, random_state=seed)
    tr, va, te = tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")
    out = pd.concat([tr, va, te]).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def load_images(df, size):
    """Baca citra -> array float32 [0,1] bentuk (N, size, size, 1)."""
    imgs = []
    for p in df["path"]:
        img = Image.open(p).convert("L").resize((size, size), Image.Resampling.LANCZOS)
        imgs.append(np.asarray(img, dtype="float32")[..., None] / 255.0)
    return np.stack(imgs), df["y"].to_numpy()


def make_ds(x, y, batch=32, shuffle=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)
