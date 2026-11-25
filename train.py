"""Latih ulang LeNet-5, AlexNet, dan VGGNet (Mini-VGG) secara lokal, masing-masing 100 epoch.

    python train.py
    python train.py --epochs 100 --data "Dataset NEU-CLS" --out results
    python train.py --models lenet5 vggnet          # hanya sebagian model

Langkah:
  1. Gabungkan semua citra di folder dataset lalu bagi stratified 70/15/15 (seed 42) ke results/split.csv.
     Ketiga model memakai pembagian yang sama, jadi dievaluasi pada data uji yang sama.
  2. Latih tiap model. Hasil tiap model disimpan di results/<model>/:
        model.keras, history.csv (akurasi dan loss tiap epoch), metrics.json, confusion_matrix.csv,
        classification_report.csv, test_prob.npy, summary.txt     dan log lengkap di results/log_<model>.txt
  3. Buat gambar laporan (results/figures/) dan ringkasan perbandingan (results/comparison.csv).

Peringatan: hasil lama di folder --out untuk model yang sama akan ditimpa.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def fmt(sec):
    m, s = divmod(int(sec), 60)
    return f"{m} mnt {s:02d} dtk" if m else f"{s} dtk"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "Dataset NEU-CLS"), help="folder dataset NEU (boleh bersarang)")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--models", nargs="+", default=["lenet5", "alexnet", "vggnet"], help="lenet5 alexnet vggnet")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="learning rate awal; default per model: lenet5 1e-3, alexnet 3e-4, vggnet 1e-3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=0, help="EarlyStopping; 0 = selalu menjalankan semua epoch")
    a = ap.parse_args()

    import trainer as T                               # impor TensorFlow baru di sini agar --help cepat
    from data import build_split
    from models import ALIASES, DEFAULT_LR, MODELS

    names = []
    for m in a.models:
        n = ALIASES.get(m.lower(), m.lower())
        if n not in MODELS:
            ap.error(f"model tidak dikenal: {m} (pilih {list(MODELS)} atau vggnet)")
        if n not in names:
            names.append(n)

    out = Path(a.out)
    print(f"Dataset : {a.data}\nKeluaran: {out}  (hasil lama untuk model yang sama akan ditimpa)\n"
          f"Model   : {', '.join(names)}  | {a.epochs} epoch | batch {a.batch}\n")
    t_all = time.time()
    df = build_split(a.data, out / "split.csv", a.seed)
    print("Pembagian data:", df.groupby("split").size().to_dict(), "| per kelas:", df.groupby("label").size().to_dict(), "\n")

    results, failed = {}, []
    for i, n in enumerate(names, 1):
        print(f"=== [{i}/{len(names)}] {n} ===", flush=True)
        t0 = time.time()
        try:
            results[n] = T.train_model(n, df, out, a.epochs, a.batch, a.lr or DEFAULT_LR[n], a.seed, a.patience)
            print(f"--- {n} selesai dalam {fmt(time.time() - t0)}: akurasi uji {results[n]['test_acc']:.4f}\n", flush=True)
        except Exception as e:                       # satu model gagal tidak membatalkan yang lain
            failed.append(n); print(f"--- {n} GAGAL: {e!r}\n", flush=True)

    if results:
        print("=== Gambar laporan dan ringkasan ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / "src" / "make_figures.py"), "--out", str(out)], check=False)
    print(f"\nSelesai dalam {fmt(time.time() - t_all)}. Hasil di: {out}")
    if failed:
        print("Gagal:", ", ".join(failed)); sys.exit(1)


if __name__ == "__main__":
    main()
