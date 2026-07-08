# Batik TrustYou — Chatbot (Python NLTK + Sastrawi)

## Menjalankan service Python

```bash
cd chatbot_service
python3 -m venv venv && source venv/bin/activate   # opsional tapi disarankan
pip install -r requirements.txt

# Download tokenizer NLTK (sekali saja)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Train model dengan dataset intents.js
python train.py

# Running service chatbot
uvicorn main:app --port 8001 --reload
```

Test service:
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"message":"tinggi 170 berat 65 ukuran apa"}'
```

> Note: Jalankan `python train.py` untuk train model ulang.

## Kemampuan bot (tamu, tanpa login)

| Pertanyaan | Intent | Jawaban |
|------------|--------|---------|
| "produk/motif/bahan apa saja" | info_produk | info koleksi & arahan ke halaman Produk |
| "jam buka / cara pesan / bayar" | operasional | info operasional |
| "ongkir / ekspedisi / estimasi" | pengiriman | info pengiriman |
| "retur / barang rusak" | retur | kebijakan retur |
| "tinggi X berat Y ukuran apa" | rekomendasi_ukuran | rekomendasi ukuran (rule-based) |
| "mau bicara admin" | hubungi_admin | tawaran alih ke admin |

## Kemampuan tambahan (pelanggan login + sudah checkout)

| Pertanyaan | Intent | Jawaban |
|------------|--------|---------|
| "status pesanan / lacak / resi" | status_pengiriman | status pengiriman + no. resi dari pesanan terakhir |

Jika pelanggan belum login menanyakan status, bot meminta login dulu.