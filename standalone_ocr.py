"""
Pipeline Pengenal Karakter Citra Digital (Deep Learning OCR)
============================================================
Modul ini mengimplementasikan sistem pengenalan teks citra CAPTCHA secara
mandiri menggunakan jaringan saraf tiruan Convolutional Recurrent Neural Network
(CRNN) dan Connectionist Temporal Classification (CTC) Decoding dengan pustaka
ONNX Runtime, NumPy, dan Pillow.

Alur Pemrosesan dan Fungsi Setiap Komponen:
-------------------------------------------
1. Penanganan Transparansi (Alpha Channel Blending):
   - Alasan Digunakan: Citra berformat PNG sering kali menyimpan latar belakang transparan
     dengan nilai RGB (0, 0, 0) yang tampak hitam pekat jika dibaca tanpa kanal alfa.
   - Fungsi: Mengganti seluruh piksel transparan (Alpha = 0) menjadi warna putih solid (255, 255, 255)
     agar kontras antara teks dan latar belakang tetap terjaga sempurna.

2. Penskalaan Spasial Aspek-Rasio Terjaga (Aspect-Ratio Preserved Resizing):
   - Alasan Digunakan: Lapisan konvolusi pada model neural network mensyaratkan dimensi tinggi
     citra yang seragam (H = 64 piksel), namun lebar teks bervariasi sesuai jumlah karakter.
   - Fungsi: Menyesuaikan dimensi tinggi tepat 64 piksel dan menghitung lebar secara proporsional
     menggunakan interpolasi bilinier (Bilinear Interpolation) agar geometri morfologi huruf
     tidak mengalami distorsi peregangan atau pemampatan.

3. Reduksi Ruang Warna (Chrominance Reduction / Grayscale):
   - Alasan Digunakan: Informasi warna (RGB 3-kanal) pada citra CAPTCHA ini bersifat redundan
     dan hanya membebani komputasi memori serta memperlambat waktu inferensi.
   - Fungsi: Mengonversi citra menjadi 1-kanal skala keabuan (L-channel) berbasis standar ITU-R BT.601,
     menghemat beban komputasi memori hingga 66.7%.

4. Normalisasi Intensitas Numerik (Min-Max Scaling):
   - Alasan Digunakan: Nilai piksel nguyên 8-bit [0, 255] memiliki rentang dinamis yang terlalu besar
     dan menyebabkan ledakan gradien (gradient explosion) pada aktivasi jaringan saraf.
   - Fungsi: Mentransformasikan nilai matriks piksel ke rentang float32 [0.0, 1.0] dengan membagi 255.0.

5. Penataan Topologi Tensor (NCHW Tensor Structuring):
   - Alasan Digunakan: Antarmuka komputasi deep learning standar ONNX Runtime mensyaratkan
     urutan dimensi berformat (Batch, Channel, Height, Width).
   - Fungsi: Membentuk matriks 2D (64, W) menjadi tensor 4D berdimensi [1, 1, 64, W].

6. CTC Greedy Decoding & Perhitungan Keakuratan (Confidence Score):
   - Alasan Digunakan: Model memprediksi distribusi probabilitas karakter pada setiap irisan waktu (frame),
     sehingga menghasilkan token yang berulang dan token kosong (blank token).
   - Fungsi: Mengambil argmax probabilitas tertinggi, mereduksi karakter kembar berturutan,
     membuang token blank (epsilon), dan menghitung rata-rata probabilitas keyakinan model (confidence %).

7. Manajemen Pemindahan Berkas (Business Process):
   - Alasan Digunakan: Menjamin otomasi pipeline pemrosesan di mana berkas CAPTCHA yang selesai dipecahkan
     langsung diarsipkan ke folder tujuan dengan penamaan teks hasil pembacaan.
   - Fungsi: Membaca berkas (CLI atau default.png), mengekstrak teks beserta persentase keakuratan,
     dan memindahkan berkas fisik ke folder 'solved/' secara otomatis.
"""

import os
import sys
import shutil
import base64
import zlib
import subprocess
from typing import Optional, Union, List, Tuple

# Auto-verifikasi & instalasi dependensi jika dijalankan pada environment Python yang baru
required_packages = {
    'numpy': 'numpy',
    'PIL': 'Pillow',
    'onnxruntime': 'onnxruntime'
}
for mod_name, pip_name in required_packages.items():
    try:
        __import__(mod_name)
    except ImportError:
        print(f"[*] Menginstal dependensi otomatis '{pip_name}' ke environment ({sys.executable})...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

import numpy as np
from PIL import Image
import onnxruntime as ort

# Kamus karakter (Charset) terkompresi internal untuk decoding CTC
_CHARSET_DATA_B64 = "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"


class StandaloneOCR:
    """
    Engine OCR mandiri berbasis ONNX Runtime dan CTC Greedy Decoding.
    Mengeksekusi inferensi langsung pada arsitektur CRNN secara independen.
    """

    def __init__(self, model_path: Optional[str] = None, charset: Optional[List[str]] = None):
        """
        Inisialisasi engine OCR.
        Memuat model ONNX dan tabel karakter (charset) dari direktori lokal.
        """
        self.model_path = model_path or self._resolve_model_path()
        self.charset = charset or self._load_embedded_charset()

        # Konfigurasi sesi inferensi ONNX Runtime
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # Menekan log internal level rendah
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _resolve_model_path() -> str:
        """
        Mencari dan memvalidasi keberadaan berkas model ONNX (model.onnx) di folder lokal.
        Fungsi: Memastikan engine dapat memuat bobot jaringan saraf tiruan tanpa konfigurasi rumit.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(current_dir, "model.onnx"),
            os.path.join(current_dir, "common_old.onnx")
        ]
        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"Berkas model ONNX tidak ditemukan pada direktori: {current_dir}. "
            "Pastikan berkas 'model.onnx' berada di folder yang sama dengan skrip ini."
        )

    @staticmethod
    def _load_embedded_charset() -> List[str]:
        """
        Mendekompresi tabel karakter (charset) langsung dari memori internal.
        Fungsi: Mengeliminasi ketergantungan pada berkas konfigurasi eksternal.
        """
        raw_bytes = base64.b64decode(_CHARSET_DATA_B64.encode("ascii"))
        decompressed_text = zlib.decompress(raw_bytes).decode("utf-8")
        return decompressed_text.split("\n")

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Pipeline Pra-Pemrosesan Citra Digital Menuju Format Tensor Masukan Jaringan:
        1. Alpha Blending: Menetralkan transparansi PNG menjadi putih solid (255, 255, 255).
        2. Bilinear Resizing: Menyeragamkan tinggi H=64 px dan menghitung lebar proporsional.
        3. Grayscale Conversion: Reduksi ruang warna ke skala keabuan 1-kanal.
        4. Min-Max Normalization: Penskalaan nilai intensitas ke rentang [0.0, 1.0] float32.
        5. NCHW Reshaping: Pembentukan topologi memori tensor 4D [1, 1, 64, W].
        """
        # 1. Penanganan Transparansi (Alpha Channel Blending)
        if image.mode == "RGBA":
            arr_rgba = np.array(image)
            is_transparent = (arr_rgba[:, :, 3] == 0)
            arr_rgba[is_transparent, :3] = 255
            image = Image.fromarray(arr_rgba[:, :, :3], mode="RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # 2. Penskalaan Spasial Aspek-Rasio Terjaga (Tinggi konstan H = 64 piksel)
        orig_w, orig_h = image.size
        target_h = 64
        target_w = max(1, int(orig_w * (target_h / float(orig_h))))
        image_resized = image.resize((target_w, target_h), Image.Resampling.BILINEAR)

        # 3. Reduksi Ruang Warna Menuju Skala Keabuan (Grayscale L-Channel)
        image_gray = image_resized.convert("L")

        # 4. Normalisasi Intensitas Numerik ke Rentang [0.0, 1.0] float32
        pixel_matrix = np.array(image_gray, dtype=np.float32) / 255.0

        # 5. Penataan Dimensi Tensor NCHW [Batch=1, Channel=1, Height=64, Width=W]
        tensor_nchw = np.expand_dims(np.expand_dims(pixel_matrix, axis=0), axis=0)
        return tensor_nchw

    def predict(
        self,
        image_input: Union[str, bytes, Image.Image],
        return_confidence: bool = False
    ) -> Union[str, Tuple[str, float]]:
        """
        Mengeksekusi inferensi model dan dekoding sekuens karakter CTC.
        
        Args:
            image_input: Path berkas gambar (str), raw bytes gambar, atau objek PIL.Image.
            return_confidence: Jika True, mengembalikan tuple (teks_hasil, persentase_keakuratan).
            
        Returns:
            String hasil pembacaan teks, atau tuple (teks, confidence_pct).
        """
        # Validasi dan pemuatan citra masukan
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Berkas gambar tidak ditemukan: {image_input}")
            img = Image.open(image_input)
        elif isinstance(image_input, bytes):
            import io
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise TypeError("Format masukan harus berupa jalur berkas (str), bytes citra, atau PIL.Image.")

        # Eksekusi pipeline pra-pemrosesan citra
        input_tensor = self.preprocess_image(img)

        # Inferensi komputasi pada ONNX Runtime
        outputs = self.session.run(None, {self.input_name: input_tensor})
        raw_output = outputs[0]  # Dimensi output: (seq_len, batch_size=1, num_classes)

        # Ekstraksi representasi logits 2D (seq_len, num_classes)
        if raw_output.ndim == 3 and raw_output.shape[1] == 1:
            logits = raw_output[:, 0, :]
        elif raw_output.ndim == 3 and raw_output.shape[0] == 1:
            logits = raw_output[0, :, :]
        else:
            logits = raw_output

        # Penerapan fungsi aktivasi Softmax terstabilkan secara numerik
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        best_indices = np.argmax(probabilities, axis=-1)
        max_probs = np.max(probabilities, axis=-1)

        # Algoritma CTC Greedy Decoding: Reduksi duplikasi berurutan & eliminasi blank token (index 0)
        decoded_chars = []
        char_confidences = []
        prev_idx = None

        for idx, prob in zip(best_indices, max_probs):
            idx = int(idx)
            if idx != prev_idx:
                if idx != 0 and idx < len(self.charset):
                    decoded_chars.append(self.charset[idx])
                    char_confidences.append(float(prob))
            prev_idx = idx

        text_result = "".join(decoded_chars)

        # Perhitungan metrik keakuratan rata-rata (confidence score %)
        if char_confidences:
            confidence_pct = float(np.mean(char_confidences)) * 100.0
        else:
            confidence_pct = 0.0

        if return_confidence:
            return text_result, confidence_pct
        return text_result


# Instance singleton global untuk efisiensi komputasi (menghindari inisialisasi sesi berulang)
_global_ocr: Optional[StandaloneOCR] = None


def get_ocr_instance() -> StandaloneOCR:
    """
    Mengembalikan instance StandaloneOCR global (Pola Desain Singleton).
    Fungsi: Memastikan sesi ONNX Runtime hanya dimuat satu kali ke dalam memori kerja.
    """
    global _global_ocr
    if _global_ocr is None:
        _global_ocr = StandaloneOCR()
    return _global_ocr


def extract_text_from_image(
    image_path: str,
    return_confidence: bool = False
) -> Union[str, Tuple[str, float]]:
    """
    Membaca berkas gambar lokal dan mengekstrak teks karakter menggunakan engine OCR.
    
    Args:
        image_path: Jalur lengkap berkas gambar pada disk lokal.
        return_confidence: Jika True, mengembalikan tuple (teks, persentase_keakuratan).
        
    Returns:
        String teks hasil pengenalan karakter, atau tuple (teks, persentase_keakuratan).
    """
    ocr = get_ocr_instance()
    result = ocr.predict(image_path, return_confidence=return_confidence)
    if return_confidence:
        text, conf = result
        return text.strip().lower(), conf
    return result.strip().lower()


def process_and_move_to_solved(image_path: str, solved_dir: str) -> Tuple[str, str, float]:
    """
    Alur Pemrosesan Berkas:
    1. Mengekstrak teks dari gambar target beserta persentase keakuratannya.
    2. Memastikan direktori arsip 'solved/' telah tersedia.
    3. Memindahkan berkas fisik ke direktori 'solved/' dan merenamenya sesuai teks hasil pembacaan.
    
    Args:
        image_path: Jalur berkas gambar yang akan diproses.
        solved_dir: Jalur direktori arsip hasil pembacaan.
        
    Returns:
        Tuple: (teks_terdeteksi, jalur_berkas_baru, persentase_keakuratan).
    """
    detected_text, confidence = extract_text_from_image(image_path, return_confidence=True)
    if not detected_text:
        raise ValueError("Gagal mendeteksi teks atau hasil pengenalan karakter kosong.")

    # Pastikan folder tujuan tersedia
    os.makedirs(solved_dir, exist_ok=True)

    # Dapatkan ekstensi berkas asal (.png, .jpg, dll)
    _, ext = os.path.splitext(image_path)
    if not ext:
        ext = ".png"

    # Susun penamaan baru berbasis teks hasil deteksi
    new_filename = f"{detected_text}{ext}"
    dest_path = os.path.join(solved_dir, new_filename)

    # Pindahkan berkas fisik (mengosongkan file asal di direktori kerja)
    shutil.move(image_path, dest_path)

    return detected_text, dest_path, confidence


if __name__ == "__main__":
    # Penanganan kompatibilitas output UTF-8 pada terminal Windows
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    script_dir = os.path.dirname(os.path.abspath(__file__))
    solved_dir = os.path.join(script_dir, "solved")

    # Jamin ketersediaan folder arsip
    os.makedirs(solved_dir, exist_ok=True)

    # Evaluasi masukan berkas dari baris perintah CLI atau gunakan default.png
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = os.path.join(script_dir, "default.png")

    try:
        text, dest, accuracy = process_and_move_to_solved(target_file, solved_dir)
        print(f"Hasil Parser : {text}")
        print(f"Keakuratan   : {accuracy:.2f}%")
        print(f"Dipindahkan ke: {dest}")
        print(f"Status       : File asli '{target_file}' telah dipindahkan ke folder solved.")
    except Exception as err:
        print(f"Terjadi kesalahan: {err}")
        sys.exit(1)
