"""Service chatbot: klasifikasi intent + jawaban. Dipanggil oleh Laravel via HTTP."""
import pickle
import random
from fastapi import FastAPI
from pydantic import BaseModel
from nlp import preprocess, extract_body, rekomendasi_ukuran

CONFIDENCE_THRESHOLD = 0.35

with open("model.pkl", "rb") as f:
    _model = pickle.load(f)

classifier = _model["classifier"]
all_words = _model["all_words"]
responses = _model["responses"]

app = FastAPI(title="Batik TrustYou Chatbot")


class ChatIn(BaseModel):
    message: str
    is_authenticated: bool = False
    tinggi: float | None = None
    berat: float | None = None


def classify(message: str):
    words = set(preprocess(message))
    features = {w: (w in words) for w in all_words}
    dist = classifier.prob_classify(features)
    tag = dist.max()
    return tag, dist.prob(tag)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(inp: ChatIn):
    intent, confidence = classify(inp.message)

    # Confidence rendah -> tawarkan admin
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "intent": "fallback",
            "reply": "Maaf, saya kurang menangkap maksud Anda. Mau saya hubungkan ke admin?",
            "suggest_admin": True,
        }

    # Rekomendasi ukuran (ditangani penuh di Python, rule-based)
    if intent == "rekomendasi_ukuran":
        t, b = inp.tinggi, inp.berat
        if t is None or b is None:
            et, eb = extract_body(inp.message)
            t = t if t is not None else et
            b = b if b is not None else eb
        return {"intent": intent, "reply": rekomendasi_ukuran(t, b)}

    # Status pengiriman: butuh data order -> biarkan Laravel yang mengisi
    if intent == "status_pengiriman":
        if not inp.is_authenticated:
            return {
                "intent": intent,
                "reply": "Untuk mengecek status pengiriman, silakan login terlebih dahulu ya.",
                "require_login": True,
            }
        # reply None menandakan Laravel harus mengambil data pesanan
        return {"intent": intent, "reply": None, "needs_order": True}

    # Minta dihubungkan ke admin
    if intent == "hubungi_admin":
        return {"intent": intent, "reply": random.choice(responses[intent]), "suggest_admin": True}

    # Jawaban umum dari intents.json
    reply = random.choice(responses.get(intent, ["Maaf, saya belum paham."]))
    return {"intent": intent, "reply": reply}
