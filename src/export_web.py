"""Ekspor model terlatih (.keras) ke format yang dibaca halaman web (LeNet-5, AlexNet, Mini-VGG).

Keluaran (folder ../web/data):
    weights.json   bobot int8 per kanal + spesifikasi arsitektur LeNet-5, AlexNet, dan Mini-VGG
    samples.json   contoh citra uji NEU (PNG base64) beserta logit acuan dari NumPy
    meta.json      metrik, confusion matrix, dan keterangan kelas

Skrip ini tidak butuh TensorFlow: bobot dibaca langsung dari berkas HDF5 di dalam
.keras, lalu inferensi float32 dan int8 dijalankan ulang dengan NumPy. Akurasi uji
hasil NumPy dibandingkan dengan results/<model>/metrics.json sebagai pemeriksaan.

    python src/export_web.py --data data/NEU
"""
import argparse
import base64
import io
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # akar proyek
OUT = ROOT / "web" / "data"
RES = ROOT / "results"
IMG = {".jpg", ".jpeg", ".png", ".bmp"}

CLASSES = ["Crazing", "Inclusion", "Patches", "Pitted_Surface", "Rolled-in_Scale", "Scratches"]
CLASS_INFO = {
    "Crazing": ("Crazing", "Retak halus", "Jaringan retakan sangat tipis menyerupai jaring di permukaan."),
    "Inclusion": ("Inclusion", "Inklusi", "Partikel atau serpihan non-logam yang terperangkap, tampak sebagai bintik atau goresan gelap tak beraturan."),
    "Patches": ("Patches", "Bercak", "Daerah tidak rata dengan bercak gelap atau terang yang cukup luas."),
    "Pitted_Surface": ("Pitted Surface", "Permukaan berlubang", "Lubang-lubang kecil (pitting) yang tersebar di permukaan baja."),
    "Rolled-in_Scale": ("Rolled-in Scale", "Kerak terguling", "Kerak oksida yang tertekan ke permukaan saat pengerolan, berpola bercak memanjang."),
    "Scratches": ("Scratches", "Goresan", "Garis lurus memanjang akibat gesekan dengan peralatan atau gulungan."),
}


def conv(id_, k, cin, cout, bn, short, title, pad="same", s=1, lrn=False):
    """lrn=True: Local Response Normalization dijalankan sesudah ReLU (AlexNet)."""
    return dict(id=id_, type="conv", k=k, s=s, pad=(k - 1) // 2 if pad == "same" else 0, cin=cin, cout=cout,
                bn=bn, lrn=lrn, bias=not bn, act="relu", short=short, title=title)


def pool(id_, kind, short, title, k=2, s=2):
    return dict(id=id_, type=kind, k=k, s=s, short=short, title=title)


# parameter LRN AlexNet (sama dengan src/models.py: n=5, k=2, alpha=1e-4, beta=0.75)
LRN = dict(radius=2, bias=2.0, alpha=1e-4, beta=0.75)


ARCHS = {
    "lenet5": dict(
        key="lenet5", name="LeNet-5", input=32, h5conv=["c1", "c3", "c5"], h5bn=[], h5dense=["f6", "out"],
        layers=[
            conv("c1", 5, 1, 6, False, "C1", "C1 · Konvolusi 5 × 5", "valid"),
            pool("s2", "avgpool", "S2", "S2 · Average pooling 2 × 2"),
            conv("c3", 5, 6, 16, False, "C3", "C3 · Konvolusi 5 × 5", "valid"),
            pool("s4", "avgpool", "S4", "S4 · Average pooling 2 × 2"),
            conv("c5", 5, 16, 120, False, "C5", "C5 · Konvolusi 5 × 5", "valid"),
            dict(id="f6", type="dense", **{"in": 120}, out=84, act="relu", short="F6", title="F6 · Dense"),
            dict(id="out", type="dense", **{"in": 84}, out=6, act="softmax", short="Output", title="Output · Softmax"),
        ]),
    "alexnet": dict(
        key="alexnet", name="AlexNet", input=128, h5conv=["conv1", "conv2", "conv3", "conv4", "conv5"], h5bn=[],
        h5dense=["fc1", "fc2", "out"], flatten={"fc1": (256, 2, 2)},     # (C, H, W) masukan Flatten sebelum fc1
        lrn=LRN,
        layers=[
            conv("conv1", 11, 1, 64, False, "Konv 1", "Konv 1 · 11 × 11, langkah 4 + LRN", "valid", s=4, lrn=True),
            pool("pool1", "maxpool", "Pool 1", "Pool 1 · Max pooling 3 × 3, langkah 2", 3, 2),
            conv("conv2", 5, 64, 192, False, "Konv 2", "Konv 2 · 5 × 5 + LRN", lrn=True),
            pool("pool2", "maxpool", "Pool 2", "Pool 2 · Max pooling 3 × 3, langkah 2", 3, 2),
            conv("conv3", 3, 192, 384, False, "Konv 3", "Konv 3 · 3 × 3"),
            conv("conv4", 3, 384, 256, False, "Konv 4", "Konv 4 · 3 × 3"),
            conv("conv5", 3, 256, 256, False, "Konv 5", "Konv 5 · 3 × 3"),
            pool("pool5", "maxpool", "Pool 5", "Pool 5 · Max pooling 3 × 3, langkah 2", 3, 2),
            dict(id="fc1", type="dense", **{"in": 1024}, out=1024, act="relu", short="FC1", title="FC1 · Dense 1024"),
            dict(id="fc2", type="dense", **{"in": 1024}, out=1024, act="relu", short="FC2", title="FC2 · Dense 1024"),
            dict(id="out", type="dense", **{"in": 1024}, out=6, act="softmax", short="Output", title="Output · Softmax"),
        ]),
    "minivgg": dict(
        key="minivgg", name="Mini-VGG", input=64,
        h5conv=["b1c1", "b1c2", "b2c1", "b2c2", "b3c1", "b3c2", "b4c1"],
        h5bn=["b1c1", "b1c2", "b2c1", "b2c2", "b3c1", "b3c2", "b4c1"], h5dense=["fc", "out"],
        layers=[
            conv("b1c1", 3, 1, 32, True, "Konv 1.1", "Blok 1 · Konvolusi 1"),
            conv("b1c2", 3, 32, 32, True, "Konv 1.2", "Blok 1 · Konvolusi 2"),
            pool("b1p", "maxpool", "Pool 1", "Blok 1 · Max pooling 2 × 2"),
            conv("b2c1", 3, 32, 64, True, "Konv 2.1", "Blok 2 · Konvolusi 1"),
            conv("b2c2", 3, 64, 64, True, "Konv 2.2", "Blok 2 · Konvolusi 2"),
            pool("b2p", "maxpool", "Pool 2", "Blok 2 · Max pooling 2 × 2"),
            conv("b3c1", 3, 64, 128, True, "Konv 3.1", "Blok 3 · Konvolusi 1"),
            conv("b3c2", 3, 128, 128, True, "Konv 3.2", "Blok 3 · Konvolusi 2"),
            pool("b3p", "maxpool", "Pool 3", "Blok 3 · Max pooling 2 × 2"),
            conv("b4c1", 3, 128, 256, True, "Konv 4.1", "Blok 4 · Konvolusi 1"),
            pool("b4p", "maxpool", "Pool 4", "Blok 4 · Max pooling 2 × 2"),
            dict(id="gap", type="gap", short="GAP", title="Global average pooling"),
            dict(id="fc", type="dense", **{"in": 256}, out=128, act="relu", short="Dense", title="Dense 128"),
            dict(id="out", type="dense", **{"in": 128}, out=6, act="softmax", short="Softmax", title="Output · Softmax"),
        ]),
}


# ---------------------------------------------------------------- bobot
def read_weights(model_dir, arch):
    zf = zipfile.ZipFile(model_dir / "model.keras")
    h5 = h5py.File(io.BytesIO(zf.read("model.weights.h5")), "r")["layers"]

    def grp(base, i):
        return h5[base if i == 0 else f"{base}_{i}"]["vars"]

    W = {}
    for i, id_ in enumerate(arch["h5conv"]):
        v = grp("conv2d", i)
        W[id_] = {"w": np.transpose(v["0"][()], (3, 2, 0, 1)).astype("float32"),   # (O, C, K, K)
                  "b": v["1"][()].astype("float32") if "1" in v else None}
    for i, id_ in enumerate(arch["h5bn"]):
        v = grp("batch_normalization", i)
        W[id_]["bn"] = dict(g=v["0"][()], be=v["1"][()], m=v["2"][()], v=v["3"][()], eps=1e-3)
    for i, id_ in enumerate(arch["h5dense"]):
        v = grp("dense", i)
        w = v["0"][()].T.astype("float32")                                                    # (O, I)
        if id_ in arch.get("flatten", {}):
            # Keras meratakan (H, W, C); mesin web meratakan (C, H, W), jadi kolom bobot diurutkan ulang
            C, H, Wd = arch["flatten"][id_]
            w = w.reshape(w.shape[0], H, Wd, C).transpose(0, 3, 1, 2).reshape(w.shape[0], -1)
        W[id_] = {"w": np.ascontiguousarray(w), "b": v["1"][()].astype("float32")}
    return W


def quantize(W):
    """int8 simetris per kanal keluaran, sama seperti yang dipakai halaman web."""
    Q = {}
    for id_, l in W.items():
        w = l["w"]
        flat = w.reshape(w.shape[0], -1)
        s = np.maximum(np.abs(flat).max(1), 1e-12) / 127.0
        q = np.clip(np.round(flat / s[:, None]), -127, 127).astype("int8")
        Q[id_] = dict(l, q=q, s=s.astype("float32"), wq=(q * s[:, None]).reshape(w.shape).astype("float32"))
    return Q


# ---------------------------------------------------------------- inferensi NumPy
def conv2d(x, w, b, pad, stride=1):
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(x, w.shape[2:], axis=(1, 2))[:, ::stride, ::stride]   # (C, Ho, Wo, K, K)
    y = np.tensordot(w, win, axes=([1, 2, 3], [0, 3, 4]))                  # (O, Ho, Wo)
    return y + (b[:, None, None] if b is not None else 0)


def pool2d(x, k, s, kind):
    win = sliding_window_view(x, (k, k), axis=(1, 2))[:, ::s, ::s]
    return win.max((3, 4)) if kind == "maxpool" else win.mean((3, 4))


def lrn(a, r, bias, alpha, beta):
    """LRN antarkanal TensorFlow: a / (bias + alpha * jumlah kuadrat 2r+1 kanal sekitar)^beta."""
    sq = np.pad(a ** 2, ((r, r), (0, 0), (0, 0))).cumsum(0)
    sq = np.concatenate([np.zeros_like(sq[:1]), sq])
    C = a.shape[0]
    s = sq[2 * r + 1: 2 * r + 1 + C] - sq[:C]
    return a / (bias + alpha * s) ** beta


def forward(arch, W, x, quant):
    """x: (size, size) float32 0..1. Mengembalikan logit."""
    a = x[None]
    for l in arch["layers"]:
        t = l["type"]
        if t == "conv":
            p = W[l["id"]]
            a = conv2d(a, p["wq"] if quant else p["w"], p["b"], l["pad"], l["s"])
            if "bn" in p:
                bn = p["bn"]
                sc = bn["g"] / np.sqrt(bn["v"] + bn["eps"])
                a = a * sc[:, None, None] + (bn["be"] - bn["m"] * sc)[:, None, None]
            a = np.maximum(a, 0)
            if l["lrn"]:
                a = lrn(a, arch["lrn"]["radius"], arch["lrn"]["bias"], arch["lrn"]["alpha"], arch["lrn"]["beta"])
        elif t in ("maxpool", "avgpool"):
            a = pool2d(a, l["k"], l["s"], t)
        elif t == "gap":
            a = a.mean((1, 2))
        elif t == "dense":
            p = W[l["id"]]
            a = (p["wq"] if quant else p["w"]) @ a.reshape(-1) + p["b"]
            if l["act"] == "relu":
                a = np.maximum(a, 0)
    return a


def load_gray(path, size=None):
    img = Image.open(path).convert("L")
    return img if size is None else img.resize((size, size), Image.Resampling.LANCZOS)


def main():
    global RES, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder berisi .bmp NEU (boleh bersarang)")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--results", default=str(RES), help="folder hasil training (berisi split.csv dan <model>/)")
    ap.add_argument("--web-out", default=str(OUT), help="folder keluaran untuk halaman web")
    a = ap.parse_args()
    RES, OUT = Path(a.results), Path(a.web_out)
    OUT.mkdir(parents=True, exist_ok=True)

    files = {f.stem: f for f in Path(a.data).rglob("*") if f.suffix.lower() in IMG}
    df = pd.read_csv(RES / "split.csv")
    df["stem"] = df.path.map(lambda p: Path(p).stem)
    test = df[df.split == "test"].reset_index(drop=True)
    print("uji:", len(test), "citra | tersedia di disk:", sum(s in files for s in test.stem))

    weights, preds, logits = {}, {}, {}
    meta = {"classes": [dict(key=c, name=CLASS_INFO[c][0], idn=CLASS_INFO[c][1], desc=CLASS_INFO[c][2]) for c in CLASSES],
            "models": {}}
    for key, arch in ARCHS.items():
        W = quantize(read_weights(RES / key, arch))
        size = arch["input"]
        X = np.stack([np.asarray(load_gray(files[s], size), dtype="float32") / 255.0 for s in test.stem])
        res = {}
        for mode in ("float", "int8"):
            lg = np.stack([forward(arch, W, x, mode == "int8") for x in X])
            res[mode] = lg
            print(f"{arch['name']:9s} {mode:5s} akurasi uji = {(lg.argmax(1) == test.y.to_numpy()).mean():.4f}")
        mt = json.load(open(RES / key / "metrics.json"))
        print(f"{'':9s} Keras (metrics.json)      = {mt['test_acc']:.4f}")
        preds[key], logits[key] = res["int8"].argmax(1), res["int8"]

        # ----- bobot
        layers = {}
        for l in arch["layers"]:
            if l["id"] not in W:
                continue
            p = W[l["id"]]
            e = dict(shape=list(p["w"].shape), q=base64.b64encode(p["q"].tobytes()).decode(),
                     s=[float(v) for v in p["s"]], b=None if p["b"] is None else [float(v) for v in p["b"]])
            if "bn" in p:
                e["bn"] = {k: ([float(v) for v in x] if k != "eps" else x) for k, x in p["bn"].items()}
            layers[l["id"]] = e
        weights[key] = dict(arch=dict(key=key, name=arch["name"], input=size, layers=arch["layers"], **({"lrn": arch["lrn"]} if "lrn" in arch else {})), layers=layers)

        # ----- metrik
        cm = pd.read_csv(RES / key / "confusion_matrix.csv", index_col=0)
        rep = pd.read_csv(RES / key / "classification_report.csv", index_col=0)
        hist = pd.read_csv(RES / key / "history.csv")
        meta["models"][key] = dict(
            name=arch["name"], input=size, params=mt["params"], epochs=mt["epochs_run"],
            train_time_s=mt["train_time_s"], infer_ms=mt["infer_ms_per_img"], test_acc=mt["test_acc"],
            test_loss=mt["test_loss"], macro_f1=mt["macro_f1"],
            confusion=cm.to_numpy().tolist(),
            f1=[float(rep.loc[c, "f1-score"]) for c in CLASSES],
            precision=[float(rep.loc[c, "precision"]) for c in CLASSES],
            recall=[float(rep.loc[c, "recall"]) for c in CLASSES],
            hist=dict(acc=[round(float(v), 4) for v in hist["accuracy"]], val_acc=[round(float(v), 4) for v in hist["val_accuracy"]],
                      loss=[round(float(v), 4) for v in hist["loss"]], val_loss=[round(float(v), 4) for v in hist["val_loss"]]))
        meta["models"][key]["acc_int8_numpy"] = round(float((preds[key] == test.y.to_numpy()).mean()), 4)
    meta["split"] = df.groupby("split").size().to_dict()

    # ----- contoh citra: per kelas, utamakan citra yang salah di sedikitnya satu model tetapi benar di model lain (contoh sulit), lalu yang benar di semua model
    rng = np.random.default_rng(7)
    y = test.y.to_numpy()
    ok = {k: preds[k] == y for k in preds}
    keys = list(ARCHS)
    samples = []
    for c in range(6):
        idx = np.flatnonzero(y == c)
        hard = [i for i in idx if any(not ok[k][i] for k in keys) and any(ok[k][i] for k in keys)]
        easy = [i for i in idx if all(ok[k][i] for k in keys)]
        pick = (list(rng.permutation(hard))[:1] + list(rng.permutation(easy)))[: a.per_class]
        for i in pick:
            f = files[test.stem[i]]
            buf = io.BytesIO(); load_gray(f).save(buf, "PNG", optimize=True)
            wrong = [k for k in keys if not ok[k][i]]
            samples.append(dict(id=test.stem[i], file=f.name, y=int(c), hard=bool(wrong), wrong=wrong, png=base64.b64encode(buf.getvalue()).decode(),
                                expect={k: [round(float(v), 5) for v in logits[k][i]] for k in logits}))
    # urutkan agar kelas bergantian: semua kelas muncul pada baris pertama
    samples.sort(key=lambda s: (sum(1 for t in samples if t["y"] == s["y"] and samples.index(t) < samples.index(s)), s["y"]))

    json.dump(weights, open(OUT / "weights.json", "w"), separators=(",", ":"))
    json.dump(samples, open(OUT / "samples.json", "w"), separators=(",", ":"))
    json.dump(meta, open(OUT / "meta.json", "w"), ensure_ascii=False, indent=1)
    for n in ("weights", "samples", "meta"):
        print(f"{n}.json  {(OUT / (n + '.json')).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
