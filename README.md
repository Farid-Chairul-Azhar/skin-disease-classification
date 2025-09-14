📌 Skin Disease Classification using Transfer Learning (VGG16 & EfficientNet)
📖 Deskripsi Proyek

Penyakit kulit merupakan salah satu masalah kesehatan yang sering dijumpai, mulai dari penyakit autoimun seperti Psoriasis hingga kanker kulit berbahaya seperti Melanoma. Diagnosis dini sangat penting untuk mencegah komplikasi.

Proyek ini membangun model klasifikasi penyakit kulit berbasis Transfer Learning dengan dua arsitektur populer: VGG16 dan EfficientNetB0. Model dilatih menggunakan dataset gambar penyakit kulit dari Kaggle dan ISIC Archive, kemudian dievaluasi untuk mengukur performanya dalam membedakan kategori penyakit kulit.

📂 Dataset

Psoriasis Skin Dataset – Kaggle

Berisi kumpulan gambar kulit dengan indikasi psoriasis.

ISIC Archive – ISIC Dataset

Koleksi gambar lesi kulit terbesar secara global, mencakup berbagai penyakit termasuk melanoma.

Tantangan dataset:

Distribusi kelas yang tidak seimbang.

Variasi kualitas gambar (resolusi, pencahayaan, sudut).

Kemiripan visual antar kelas (misalnya psoriasis vs eczema).

⚙️ Metodologi

Preprocessing

Resize gambar ke 224x224 px.

Normalisasi piksel → [0,1].

Data augmentation: rotation, zoom, flip.

Split data: Train / Validation / Test.

Modeling

VGG16: pretrained pada ImageNet, digunakan tanpa fine-tuning.

EfficientNetB0: pretrained pada ImageNet, di-fine-tune pada beberapa lapisan akhir.

Evaluasi

Accuracy, Precision, Recall, F1-score.

Confusion Matrix.

Learning Curve (akurasi & loss).

📊 Hasil Eksperimen
🔹 VGG16 (Tanpa Fine-Tuning)

Accuracy: 96.58%

Loss: 0.1015

Precision: 96.92%

Recall: 96.18%

F1-Score: 96.55%

Confusion Matrix (VGG16):

TN = 128

FP = 4

FN = 5

TP = 126

🔹 EfficientNetB0 (Fine-Tuned)

Accuracy: 73.38%

Loss: 0.4719

Precision: 85.06%

Recall: 56.49%

F1-Score: 67.30%

Confusion Matrix (EfficientNetB0):

TN = 119

FP = 13

FN = 57

TP = 74

🔹 Perbandingan Model
Model	Akurasi	Loss	Precision	Recall	F1-Score
VGG16 (No Fine-Tune)	96.58%	0.1015	96.92%	96.18%	96.55%
EfficientNetB0 (FT)	73.38%	0.4719	85.06%	56.49%	67.30%


Visualisasi perbandingan hasil antara VGG16 dan EfficientNetB0

🚀 Kesimpulan

VGG16 tanpa fine-tuning menghasilkan performa yang sangat baik (akurasi 96.58%).

EfficientNetB0 dengan fine-tuning meningkat signifikan dibanding versi awal (50.19% → 73.38%), tapi masih di bawah VGG16.

Imbalance dataset dan kemiripan visual antar kelas menjadi faktor utama penurunan performa EfficientNet.
