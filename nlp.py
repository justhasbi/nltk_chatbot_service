"""Preprocessing bersama: tokenisasi (NLTK) + stemming Bahasa Indonesia (Sastrawi)."""
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Pastikan resource tokenizer NLTK tersedia (aman lintas versi NLTK)
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

_stemmer = StemmerFactory().create_stemmer()


def preprocess(text: str):
    """Ubah kalimat menjadi daftar kata dasar (stem) huruf kecil."""
    text = text.lower()
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        tokens = re.findall(r"\w+", text)  # fallback bila punkt tak tersedia
    return [_stemmer.stem(t) for t in tokens if t.isalnum()]


def extract_body(message: str):
    """Ambil tinggi (cm) & berat (kg) dari kalimat, mis. 'tinggi 170 berat 65'."""
    msg = message.lower()
    tinggi = berat = None

    m = re.search(r"tinggi\D{0,6}(\d{2,3})", msg)
    if m:
        tinggi = int(m.group(1))
    m = re.search(r"berat\D{0,6}(\d{2,3})", msg)
    if m:
        berat = int(m.group(1))

    # Fallback heuristik: angka >= 100 dianggap tinggi, < 100 dianggap berat
    if tinggi is None or berat is None:
        for n in map(int, re.findall(r"\b(\d{2,3})\b", msg)):
            if n >= 100 and tinggi is None:
                tinggi = n
            elif n < 100 and berat is None:
                berat = n
    return tinggi, berat


def rekomendasi_ukuran(tinggi, berat):
    """Rekomendasi ukuran umum berdasarkan tinggi & berat badan."""
    if tinggi is None or berat is None:
        return "Boleh info tinggi (cm) dan berat (kg) badan Anda? Contoh: tinggi 165 berat 60."

    if berat <= 55 and tinggi <= 165:
        size = "S"
    elif berat <= 68 and tinggi <= 172:
        size = "M"
    elif berat <= 82 and tinggi <= 178:
        size = "L"
    elif berat <= 95:
        size = "XL"
    else:
        size = "XXL"

    return (
        f"Untuk tinggi {tinggi} cm dan berat {berat} kg, ukuran yang kami rekomendasikan adalah {size}. "
        "Setiap model bisa sedikit berbeda—cek tabel ukuran di halaman produk untuk memastikan ya."
    )
