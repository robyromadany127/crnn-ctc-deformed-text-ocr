# Kajian Ilmiah: Pengenalan Teks Visual & Evaluasi Keamanan CAPTCHA v1
### Analisis Komprehensif Visi Komputer, Pemodelan Sekuens NLP, Deep Learning, dan Sistem Verifikasi Web

Dokumen ini membedah secara mendalam aspek teoritis, formulasi matematis, dan rekayasa perangkat lunak dari pipeline pengenalan karakter visual (OCR) pada sistem CAPTCHA v1.

---

## Daftar Isi
1. [Landasan Visi Komputer (Computer Vision)](#1-landasan-visi-komputer-computer-vision)
2. [Advanced NLP & Pemodelan Sekuens Temporal (Sequence Modeling)](#2-advanced-nlp--pemodelan-sekuens-temporal-sequence-modeling)
3. [Arsitektur Deep Learning & Rekayasa Inferensi (CRNN + ONNX)](#3-arsitektur-deep-learning--rekayasa-inferensi-crnn--onnx)
4. [Analisis Keamanan Siber: Runtuhnya Paradigma CAPTCHA v1](#4-analisis-keamanan-siber-runtuhnya-paradigma-captcha-v1)
5. [Pembedahan Teknis Kode Notebook (`standalone_ocr_tutorial.ipynb`)](#5-pembedahan-teknis-kode-notebook)
6. [Referensi & Sitasi Akademik](#6-referensi--sitasi-akademik)

---

## 1. Landasan Visi Komputer (Computer Vision)

Pra-pemrosesan citra digital dalam pipeline ini bukan sekadar manipulasi format berkas, melainkan tahapan reduksi entropi untuk mengisolasi informasi morfologis teks dari derau (*noise*).

```
[Citra Masukan Mentah (RGBA/RGB)]
               │
               ▼
[1. Alpha Blending] ─────────> Netralkan transparansi: piksel transparan diubah ke putih solid (255)
               │
               ▼
[2. Bilinear Resizing] ──────> Normalisasi tinggi H=64 px, skala lebar W dinamis menjaga rasio aspek
               │
               ▼
[3. Grayscale ITU-R BT.601] ─> Reduksi 3-kanal RGB ke 1-kanal Luminance (L), hemat 66.7% memori
               │
               ▼
[4. Min-Max Normalization] ──> Transformasi intensitas [0, 255] uint8 ke rentang [0.0, 1.0] float32
               │
               ▼
[5. Topologi Tensor NCHW] ───> Pembentukan tensor 4D [1, 1, 64, W] siap inferensi ONNX
```

### 1.1 Penanganan Transparansi (Alpha Channel Blending)
Format citra modern (khususnya PNG 32-bit) sering kali menyimpan latar belakang dalam kanal alfa transparan ($A = 0$). Pada representasi matriks mentah, piksel transparan kerap diberi nilai RGB $(0, 0, 0)$ (hitam). Jika kanal alfa diabaikan saat konversi ke ruang warna RGB/Grayscale, latar belakang citra akan berubah menjadi hitam pekat dan menutupi teks.

Operasi normalisasi alfa dilakukan dengan mendeteksi koordinat piksel transparan dan menggantikannya dengan kanvas putih solid:

$$\mathbf{I}_{\text{RGB}}(x, y) = \begin{cases} (255, 255, 255), & \text{jika } A(x, y) = 0 \\ \mathbf{I}_{\text{RGBA}}(x, y)_{[:3]}, & \text{jika } A(x, y) > 0 \end{cases}$$

### 1.2 Penskalaan Spasial Aspek-Rasio Terjaga (Aspect-Ratio Preserved Resizing)
Jaringan konvolusional (CNN) mensyaratkan dimensi tinggi yang tetap agar sesuai dengan ukuran kernel vertikal model. Namun, lebar citra bervariasi sesuai jumlah karakter. Jika citra dipaksa (*stretched*) ke ukuran tetap (misal $64 \times 128$), rasio geometri huruf akan terdistorsi (huruf menjadi terlalu ramping atau terlalu gemuk), yang menurunkan akurasi pengenalan pola.

Penskalaan dilakukan dengan mengunci tinggi target $H_{\text{target}} = 64\text{ px}$ dan menghitung lebar target secara proporsional:

$$W_{\text{target}} = \max\left(1, \; \text{round}\left(W_{\text{asli}} \times \frac{H_{\text{target}}}{H_{\text{asli}}}\right)\right)$$

Proses *resampling* memanfaatkan **Interpolasi Bilinier (*Bilinear Interpolation*)**, yang mengestimasi nilai intensitas piksel baru berdasarkan rata-rata terbobot dari empat tetangga terdekat, sehingga transisi kontur huruf tetap halus tanpa artefak bergerigi (*aliasing*).

### 1.3 Reduksi Ruang Warna (Chrominance Reduction via ITU-R BT.601)
Karakteristik diskriminatif teks berada pada fitur gradien tepi dan geometri morfologi, bukan pada saluran warna (*chrominance*). Citra 3-kanal RGB memiliki redundansi informasi yang tinggi.

Konversi citra ke skala keabuan 1-kanal (*Luminance*) dilakukan menggunakan standar **ITU-R BT.601**:

$$Y(x, y) = 0.299 \cdot R(x, y) + 0.587 \cdot G(x, y) + 0.114 \cdot B(x, y)$$

*Dampak Efisiensi*: Matriks citra berkurang dari $(H, W, 3)$ menjadi $(H, W, 1)$, menghemat konsumsi memori kerja dan *bandwidth* komputasi hingga **66.7%** tanpa kehilangan ketajaman fitur teks.

### 1.4 Normalisasi Intensitas Numerik (Min-Max Scaling)
Representasi citra standar menggunakan bilangan bulat 8-bit `uint8` dengan rentang $[0, 255]$. Memberikan nilai skala besar ke dalam jaringan saraf tiruan dapat menyebabkan ledakan gradien (*gradient explosion*) dan kejenuhan fungsi aktivasi non-linear.

Penskalaan linier dilakukan dengan membagi seluruh elemen matriks dengan skalar $255.0$:

$$\mathbf{I}_{\text{norm}}(x, y) = \frac{\mathbf{I}_{\text{gray}}(x, y)}{255.0}, \quad \mathbf{I}_{\text{norm}} \in [0.0, 1.0] \subset \mathbb{R}$$

### 1.5 Penataan Topologi Memori Tensor (NCHW Layout)
Sebagian besar arsitektur inferensi *deep learning* yang teroptimasi (termasuk ONNX Runtime dan engine berbasis cuDNN) mensyaratkan urutan dimensi memori dalam format **NCHW**:
* $N = 1$ : Ukuran kelompok (*Batch size*) untuk inferensi instan.
* $C = 1$ : Jumlah saluran kanal (1 kanal skala keabuan).
* $H = 64$ : Dimensi spasial vertikal (tinggi).
* $W$ : Dimensi spasial horizontal proporsional (lebar dinamis).

Transformasi dilakukan dari matriks 2D $(64, W)$ menjadi tensor 4D `[1, 1, 64, W]` menggunakan ekspansi dimensi (*dimension expansion*).

---

## 2. Advanced NLP & Pemodelan Sekuens Temporal (Sequence Modeling)

Pengenalan teks berbasis citra (*Scene Text Recognition* / OCR) secara fundamental merupakan irisan antara Visi Komputer dan Pemrosesan Bahasa Alami (*Natural Language Processing* / NLP). Teks visual dipandang sebagai suatu urutan sekuensial (*temporal sequence*) dari kiri ke kanan.

### 2.1 Paradigma Alignment-Free Sequence Learning
Pendekatan OCR tradisional memerlukan tahapan *character segmentation*—memotong-motong citra menjadi kotak-kotak huruf individu sebelum diklasifikasikan. Pada citra dengan teks bersambung, terdistorsi, atau berimpitan (seperti CAPTCHA v1), segmentasi huruf hampir mustahil dilakukan secara sempurna.

Model CRNN menyelesaikan masalah ini secara *alignment-free*: seluruh kata diproses sebagai satu kesatuan sekuens utuh menggunakan kombinasi Recurrent Neural Network dan fungsi rugi **Connectionist Temporal Classification (CTC)**.

### 2.2 Algoritma CTC Greedy Decoding (Graves et al., 2006)
Model membagi citra masukan menjadi $T$ irisan vertikal waktu (*time-steps*). Pada setiap irisan waktu $t \in \{1, \dots, T\}$, model memprediksi distribusi probabilitas atas seluruh karakter dalam kamus ditambah sebuah token khusus: **Blank Token ($\epsilon$, indeks 0)**.

Karena satu karakter fisik membentang melewati beberapa irisan waktu, model menghasilkan prediksi token yang berulang pada frame yang berturutan (misal: `['b', 'b', 'b', 0, 'a', 'a']`).

Algoritma **CTC Greedy Decoding** ($\mathcal{B}$-mapping) meruntuhkan prediksi mentah menjadi teks final melalui dua aturan formal:
1. **Runtuhkan token berulang yang berturutan**: Jika token pada frame $t$ identik dengan frame $t-1$, maka token tersebut diabaikan.
2. **Eliminasi Blank Token ($\epsilon$)**: Token indeks 0 dihilangkan dari hasil akhir. Blank token berfungsi sebagai pemisah penting ketika sebuah kata memiliki huruf kembar yang memang berdampingan (contoh: kata `"free"` dipisahkan oleh blank token: `['e', 'e', 0, 'e', 'e']` $\rightarrow$ direduksi menjadi `"ee"`).

```
Prediksi Mentah Frame : [ - , 'b', 'b', 'b',  - , 'a', 'a',  - , 't', 't' ]
Aturan 1 (Deduplikasi): [ - , 'b',  - , 'a',  - , 't' ]
Aturan 2 (Buang Blank): [ 'b', 'a', 't' ] -> Teks Final: "bat"
```

### 2.3 Representasi Kamus Karakter (Charset / Vocabulary)
Model menggunakan kamus karakter terkompresi internal sebanyak **8.210 entri**:
* Indeks `0`: Blank Token ($\epsilon$).
* Indeks `1` s/d `8209`: Meliputi digit alfanumerik (`0-9`, `a-z`, `A-Z`), simbol tanda baca, serta karakter ortografi visual multibahasa.
* Untuk efisiensi distribusi, kamus karakter disimpan dalam format string Base64 yang didekompresi menggunakan algoritma `zlib` langsung ke memori RAM saat program dijalankan.

### 2.4 Perhitungan Metrik Tingkat Keyakinan (Confidence Score)
Tingkat keyakinan sistem dihitung dari rata-rata probabilitas maksimum kelas karakter yang lolos seleksi CTC:

$$\text{Confidence (\%)} = \left(\frac{1}{L} \sum_{i=1}^L P(c_i)\right) \times 100\%$$

Di mana $L$ adalah panjang string teks akhir, dan $P(c_i)$ adalah probabilitas Softmax pada frame puncak saat karakter $c_i$ diterima.

---

## 3. Arsitektur Deep Learning & Rekayasa Inferensi (CRNN + ONNX)

### 3.1 Arsitektur Jaringan CRNN (Shi et al., 2015)
Arsitektur CRNN (*Convolutional Recurrent Neural Network*) terdiri dari tiga komponen utama yang bekerja secara *end-to-end*:

```
Citra Input [1, 1, 64, W]
         │
         ▼
┌─────────────────────────────────┐
│ 1. Lapisan Konvolusional (CNN)  │  --> Ekstraksi representasi peta fitur spasial
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ 2. Map-to-Sequence Layer        │  --> Irisan fitur vertikal diubah jadi urutan sekuens waktu
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ 3. Lapisan Rekuren (BiLSTM)     │  --> Menangkap dependensi konteks maju & mundur (bidirectional)
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ 4. Lapisan Transkripsi (CTC)    │  --> Menghasilkan logits probabilitas tiap langkah waktu
└─────────────────────────────────┘
```

1. **Komponen Konvolusi**: Mereduksi dimensi spasial tinggi dan mengekstrak fitur garis, sudut, dan kurva huruf dari citra.
2. **Komponen Rekuren (Bidirectional LSTM)**: Memproses irisan fitur sebagai urutan waktu. BiLSTM mengevaluasi keterkaitan huruf dengan membaca urutan dari arah kiri-ke-kanan dan kanan-ke-kiri, meminimalkan kerancuan pada karakter yang mirip (misal: membedakan `'c'` dan `'e'`, atau `'l'` dan `'1'`).
3. **Komponen Transkripsi**: Lapisan linier padat (*dense layer*) yang menghasilkan skor aktivasi (*logits*) berdimensi $(T, N, |\mathcal{C}| + 1)$, di mana $|\mathcal{C}| = 8.209$ kelas karakter.

### 3.2 Stabilitas Numerik Fungsi Softmax
Untuk mengonversi matriks *logits* riil $(-\infty, +\infty)$ menjadi distribusi probabilitas $[0.0, 1.0]$, diterapkan fungsi aktivasi **Softmax**. Untuk menghindari kesalahan *floating point overflow* ketika nilai logits sangat besar, diterapkan trik stabilisasi numerik (*max-subtraction*):

$$P(c_t \mid \mathbf{x}) = \frac{e^{z_{t, c} - \max_k(z_{t, k})}}{\sum_{j} e^{z_{t, j} - \max_k(z_{t, k})}}$$

Pengurangan dengan nilai maksimum tiap baris menjamin nilai eksponensial maksimum bernilai $e^0 = 1.0$, sehingga komputasi terhindar dari nilai tak hingga (`inf` atau `NaN`).

### 3.3 Optimasi Inferensi Mandiri via ONNX Runtime
Model diekspor ke dalam format **ONNX (Open Neural Network Exchange)** berukuran ~13 MB. Keuntungan teknis pendekatan ini:
* **Zero-Heavy-Framework Dependency**: Tidak memerlukan instalasi PyTorch (~2.5 GB) atau TensorFlow (~500 MB) pada peladen produksi (*inference server*).
* **Efisiensi Komputasi CPU**: ONNX Runtime menerapkan teknik fusi operator (*operator fusion*) dan optimasi eksekusi instruksi CPU (seperti AVX/AVX2), memungkinkan inferensi citra selesai dalam durasi rata-rata **< 30 milidetik per citra**.

---

## 4. Analisis Keamanan Siber: Runtuhnya Paradigma CAPTCHA v1

### 4.1 Asumsi Klasik Uji Turing CAPTCHA v1 (von Ahn et al., 2003)
CAPTCHA (*Completely Automated Public Turing test to tell Computers and Humans Apart*) generasi pertama dirancang berdasarkan premis:
> *"Membaca teks yang terdistorsi secara visual, diberi derau titik/garis acak, serta saling berhimpitan adalah pekerjaan yang relatif mudah bagi kognisi manusia, namun merupakan persoalan AI yang sangat sulit dipecahkan oleh komputer (*Hard AI Problem*)."*

Pada era tahun 2000-an, sistem pertahanan ini efektif karena algoritma OCR tradisional (seperti Tesseract v3 berbasis pencocokan pola matriks atau ekstraksi fitur manual) membutuhkan pemisahan huruf yang bersih.

### 4.2 Faktor Penyebab Keruntuhan CAPTCHA Teks di Era Modern
Kombinasi teknik modern berhasil membongkar seluruh mekanisme pertahanan CAPTCHA v1:
1. **Derau Garis dan Titik (*Noise & Background Lines*) Teranulir**: Filter konvolusi (CNN) secara inheren mempelajari fitur invariansi translasi dan mampu mengabaikan garis derau yang tidak memiliki pola struktur ortografis teks.
2. **Segmentasi Tidak Lagi Diperlukan**: Adanya CTC Loss mengeliminasi kebutuhan memotong huruf. Huruf yang bersambung (*connected scripts*) atau tumpang tindih (*overlapping characters*) dapat dibaca secara sekuensial oleh BiLSTM.
3. **Tingkat Akurasi Melampaui Batas Toleransi Manusia**: Model Deep Learning mampu mencapai akurasi pembacaan di atas **95%**, bahkan sering kali membaca lebih teliti daripada manusia pada teks yang sangat buram.

### 4.3 Vektor Ancaman Terhadap Sistem Web Terbuka
Runtuhnya sistem verifikasi teks dasar menimbulkan celah keamanan siber yang signifikan:
* **Credential Stuffing & Brute Force**: Penyerang dapat menguji jutaan kombinasi nama pengguna dan kata sandi pada formulir masuk (*login page*) secara otomatis tanpa terhambat oleh verifikasi teks.
* **Automated Data Scraping**: Akses terhadap konten terproteksi (harga tiket, direktori publik, data lelang) dapat dikuras secara masif oleh bot.
* **Denial of Service (DoS) pada Lapisan Aplikasi**: Pembuatan akun palsu massal atau pemesanan tiket otomatis (*scalper bot*) yang membebani basis data aplikasi.

### 4.4 Evolusi Mekanisme Pertahanan Verifikasi Modern
Industri keamanan web global telah meninggalkan CAPTCHA v1 dan bermigrasi ke generasi baru:
* **CAPTCHA v2 / v3 (Behavioral Analysis)**: Menganalisis pola interaksi pengguna (pergerakan kursor *mouse*, riwayat penelusuran, waktu respons browser) tanpa menampilkan kuis teks (misal: Google reCAPTCHA v3 atau Cloudflare Turnstile).
* **Proof of Work (PoW) Kriptografis**: Memaksa peramban klien menyelesaikan teka-teki komputasi matematis sebelum formulir dikirim, meningkatkan biaya komputasi bagi penyerang massal.
* **Autentikasi Tanpa Kata Sandi (FIDO2 / WebAuthn)**: Menggantikan kombinasi kata sandi + CAPTCHA dengan verifikasi biometrik berbasis perangkat keras (*passkeys*).

---

## 5. Pembedahan Teknis Kode Notebook

Berikut adalah pemetaan operasional setiap tahapan kode di dalam `standalone_ocr_tutorial.ipynb`:

| Langkah | Fungsi / Komponen | Peran & Alasan Teknis |
| :--- | :--- | :--- |
| **0** | `ort.InferenceSession`, Charset Decompression | Memuat bobot jaringan ONNX teroptimasi dan mendekompresi 8.210 kamus token dari memori RAM via `zlib` tanpa membaca berkas teks eksternal. |
| **1** | Evaluasi Jalur `target_file` | Validasi keberadaan berkas masukan (`default.png`). Menyediakan mekanisme *fallback* ramah tanpa memicu *system crash*. |
| **2** | `apply_alpha_blending` | Menetralkan kanal transparansi PNG menjadi latar putih solid `(255, 255, 255)` untuk mencegah latar belakang menjadi hitam pekat. |
| **3** | `resize_keep_aspect_ratio` | Mengunci tinggi $H=64\text{ px}$ dan menghitung lebar $W$ proporsional menggunakan interpolasi bilinier guna mempertahankan rasio aspek huruf. |
| **4** | `img.convert("L")` | Mereduksi 3-kanal RGB menjadi 1-kanal skala keabuan berstandar ITU-R BT.601, menghemat memori hingga 66.7%. |
| **5** | Min-Max Normalization | Penskalaan matriks piksel `uint8` $[0, 255]$ ke `float32` $[0.0, 1.0]$ guna menjamin stabilitas gradien dan aktivasi model. |
| **6** | `np.expand_dims` (NCHW) | Membentuk tensor 4D `[1, 1, 64, W]` sesuai dengan konvensi memori antarmuka ONNX Runtime. |
| **7** | `ort_session.run()` | Mengeksekusi *forward pass* inferensi pada CPU, menghasilkan representasi matriks *logits* berdimensi $(T, 1, 8210)$. |
| **8** | Softmax Terstabilkan | Menghitung probabilitas kelas karakter menggunakan teknik pengurangan nilai maksimum (*max-subtraction*). |
| **9** | CTC Greedy Decoding | Mengeliminasi token kembar berurutan dan membuang *blank token* ($\epsilon$, indeks 0) serta mengkalkulasi metrik *confidence score* (%). |
| **10** | `shutil.copy2` ke `solved/` | Menyimpan berkas hasil ke direktori arsip dengan nama baru berbasis teks deteksi (`solved/{text}.png`). |
| **11** | `os.remove()` | Menghapus berkas mentah dari direktori kerja setelah validasi arsip berhasil, mencegah pemrosesan duplikat (*directory hygiene*). |
| **12** | `process_and_move_to_solved` | Mengenkapsulasi seluruh tahapan 1–11 ke dalam satu fungsi terpadu modular siap pakai (*drop-in function*). |

---

## 6. Referensi & Sitasi Akademik

1. **Shi, B., Bai, X., & Yao, C. (2015).** *An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition.* IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 39(11), 2298–2304.
2. **Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006).** *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks.* Proceedings of the 23rd International Conference on Machine Learning (ICML '06), 369–376.
3. **von Ahn, L., Blum, M., Hopper, N. J., & Langford, J. (2003).** *CAPTCHA: Using Hard AI Problems for Security.* In International Conference on the Theory and Applications of Cryptographic Techniques (EUROCRYPT 2003), Lecture Notes in Computer Science, vol 2656. Springer, Berlin, Heidelberg.
4. **Bursztein, E., Martin, M., & Mitchell, J. (2011).** *Text-based CAPTCHA Strengths and Weaknesses.* Proceedings of the 18th ACM Conference on Computer and Communications Security (CCS '11), 125–138.
5. **International Telecommunication Union (ITU-R). (2011).** *Recommendation ITU-R BT.601-7: Studio encoding parameters of digital television for standard 4:3 and wide screen 16:9 aspect ratios.*
