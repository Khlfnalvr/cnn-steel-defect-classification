# Klasifikasi Cacat Permukaan Baja Hot-Rolled dengan CNN (LeNet-5, AlexNet, VGGNet)

Tugas Mata Kuliah **Jaringan Syaraf Tiruan** – Semester Ganjil 2026/2027
Departemen Teknik Fisika, Institut Teknologi Sepuluh Nopember (ITS)

## Ringkasan
Inspeksi visual cacat permukaan pelat baja di jalur produksi masih banyak dilakukan manual.
Proyek ini menerapkan **Convolutional Neural Network (CNN)** untuk mengklasifikasikan 6 jenis
cacat permukaan baja hot-rolled secara otomatis, dan membandingkan tiga struktur:

| Model | Input | Parameter | Ciri utama |
|---|---|---|---|
| LeNet-5 | 32×32×1 | 61.366 | 3 konvolusi 5×5, average pooling, 2 Dense |
| AlexNet (diskalakan) | 128×128×1 | 4.559.558 | 5 konvolusi (11×11/4, 5×5, 3×3 ×3), ReLU, LRN, max pooling 3×3/2, 2 Dense 1024 + Dropout |
| VGGNet (Mini-VGG) | 64×64×1 | 617.382 | blok konvolusi 3×3 + BatchNorm + max pooling, Global Average Pooling |

Hasilnya juga tersedia sebagai halaman web interaktif (**Lensa Baja**) yang menampilkan setiap lapisan
jaringan dan berjalan sepenuhnya di browser untuk ketiga model (LeNet-5, AlexNet, dan Mini-VGG).

Hasil training terakhir (data uji 270 citra, lihat `results/comparison.csv`): LeNet-5 86,67 %, AlexNet 98,52 %,
Mini-VGG 100 %. Angka ini akan berubah setelah training ulang dengan `train.py`.

## Dataset
Folder `Dataset NEU-CLS/`: **NEU Surface Defect Database** (Northeastern University, Tiongkok), 1.800 citra
200×200 px, 6 kelas × 300 citra: *Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches*.

> K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for
> hot-rolled steel strip surface defects," *Applied Surface Science*, vol. 285, pp. 858–864, 2013.

Label kelas diambil dari awalan nama file (`crazing_10.jpg`). Berkas `.txt` di folder `labels/` adalah kotak
cacat untuk deteksi objek dan tidak dipakai. Semua citra (folder `train` dan `valid`) digabung lalu dibagi ulang
secara stratified, seed 42: **train 1.260 / validasi 270 / uji 270** (lihat `results/split.csv`). Ketiga model
memakai pembagian yang sama.

## Struktur folder
```
CNN Project/
├── train.py             latih LeNet-5, AlexNet, dan VGGNet sekaligus (100 epoch), simpan semua hasil
├── Dataset NEU-CLS/         dataset (gambar .jpg)
├── src/
│   ├── data.py              baca citra, split stratified 70/15/15
│   ├── models.py            arsitektur LeNet-5, AlexNet, Mini-VGG (Keras)
│   ├── trainer.py           fungsi training + evaluasi satu model (dipanggil train.py)
│   ├── make_figures.py      semua grafik untuk laporan
│   └── export_web.py        ekspor model .keras ke web/data (bobot int8, contoh citra, metrik)
├── results/                 model terlatih (.keras), log, metrik, gambar
│   ├── split.csv, comparison.csv, log_<model>.txt
│   ├── <model>/             history.csv, metrics.json, confusion_matrix.csv, classification_report.csv,
│   │                        test_prob.npy, summary.txt, model.keras
│   └── figures/             kurva training, confusion matrix, F1 per kelas, feature map, contoh prediksi
├── web/                     sumber halaman web (engine.js, scene.js, s3ui.js, page.js, template.html, build.py, test.js, data/)
├── dist/neu-steel-defect.html   halaman jadi, satu berkas, buka langsung di browser
└── requirements.txt
```

## Cara menjalankan

### Training ulang (lokal)
Butuh Python 3.9–3.12 (TensorFlow belum mendukung versi yang lebih baru) dan TensorFlow 2.16 ke atas.
```bash
pip install -r requirements.txt
python train.py
```
Perintah ini melatih LeNet-5, AlexNet, dan VGGNet berturut-turut, masing-masing **100 epoch penuh** (bobot dengan
val_loss terendah dipulihkan di akhir), lalu membuat gambar laporan dan `results/comparison.csv`.
Hasil lama di `results/` untuk model yang sama **ditimpa**.

Waktu di CPU (diukur pada 16 inti, tanpa GPU): LeNet-5 sekitar 1–2 menit, AlexNet sekitar 10–12 menit,
Mini-VGG sekitar 15 menit. Total sekitar 30 menit.

Opsi lain:
```bash
python train.py --models lenet5 vggnet          # hanya sebagian model
python train.py --epochs 50 --out results_uji   # epoch lain / folder keluaran lain
python train.py --models alexnet                      # satu model saja
```
Detail training: Adam (lr awal 1e-3; AlexNet 3e-4), loss *sparse categorical cross-entropy*, batch 32,
ReduceLROnPlateau (faktor 0,5; patience 6), augmentasi flip horizontal/vertikal, translasi ±10 %, kontras ±10 %.

### Halaman web
Buka `dist/neu-steel-defect.html` di browser. Three.js dimuat dari cdnjs untuk tampilan 3D, sedangkan model,
contoh citra, dan semua bagian lain sudah tertanam di berkas. Model (LeNet-5, AlexNet, Mini-VGG) dapat diganti lewat tombol di bilah atas; ukuran berkas sekitar 7,5 MB karena memuat bobot AlexNet (4,5 juta parameter, int8).

Setelah training ulang, perbarui halaman dengan:
```bash
python src/export_web.py --data "Dataset NEU-CLS"   # baca results/*/model.keras, tulis web/data/ (butuh h5py)
python web/build.py                                 # hasilkan dist/neu-steel-defect.html (membaca web/data/*.json)
node web/test.js                                    # cek inferensi JS sama dengan NumPy
```
`export_web.py` membaca bobot langsung dari berkas `.keras`, menjalankan ulang inferensi dengan NumPy, dan
menyimpan bobot int8 per kanal. Praproses di browser meniru training: skala abu-abu, resize Lanczos ke ukuran
masukan model, lalu dibagi 255. Model hanya dilatih pada citra NEU, jadi foto baja dari sumber lain bisa
menghasilkan prediksi yang tidak bermakna.

## Referensi
1. Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-based learning applied to document recognition," *Proc. IEEE*, 86(11), 1998.
2. A. Krizhevsky, I. Sutskever, G. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," NIPS 2012.
3. K. Simonyan, A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," arXiv:1409.1556, 2014.
4. S. Ioffe, C. Szegedy, "Batch Normalization," ICML 2015.
5. K. Song, Y. Yan, *Applied Surface Science* 285 (2013) 858–864.
