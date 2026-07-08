import json
import pickle
import os

import nltk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from nlp import preprocess

REPORT_DIR = "reports"
INDIGO = "#2e2a5e"


def build_features(doc_words, all_words):
    doc_set = set(doc_words)
    return {w: (w in doc_set) for w in all_words}


def load_data():
    with open("intents.json", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels, responses = [], [], {}
    for intent in data["intents"]:
        tag = intent["tag"]
        responses[tag] = intent.get("responses", [])
        for pattern in intent["patterns"]:
            texts.append(preprocess(pattern))
            labels.append(tag)
    return texts, labels, responses

def plot_distribution(labels):
    tags = sorted(set(labels))
    counts = [labels.count(t) for t in tags]
    plt.figure(figsize=(9, 4.5))
    plt.bar(tags, counts, color=INDIGO)
    plt.title("Distribusi Data Latih per Intent")
    plt.ylabel("Jumlah pola")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/01_distribusi_data.png", dpi=130)
    plt.close()


def plot_accuracy(train_acc, test_acc):
    plt.figure(figsize=(5, 4.5))
    bars = plt.bar(["Data Latih", "Data Uji"], [train_acc, test_acc],
                   color=[INDIGO, "#8a86b8"])
    plt.title("Akurasi Model")
    plt.ylabel("Akurasi")
    plt.ylim(0, 1.05)
    for b, v in zip(bars, [train_acc, test_acc]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/02_akurasi.png", dpi=130)
    plt.close()


def plot_classification_report(report):
    tags = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
    metrics = ["precision", "recall", "f1-score"]
    colors = [INDIGO, "#6f6ba8", "#b3b0d4"]

    x = range(len(tags))
    width = 0.25
    plt.figure(figsize=(10, 5))
    for i, m in enumerate(metrics):
        vals = [report[t][m] for t in tags]
        plt.bar([p + i * width for p in x], vals, width, label=m, color=colors[i])
    plt.title("Metrik per Intent (Data Uji)")
    plt.ylabel("Skor")
    plt.ylim(0, 1.1)
    plt.xticks([p + width for p in x], tags, rotation=40, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/03_classification_report.png", dpi=130)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, tags):
    cm = confusion_matrix(y_true, y_pred, labels=tags)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(len(tags)))
    ax.set_yticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_yticklabels(tags)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix (Data Uji)")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(len(tags)):
        for j in range(len(tags)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/04_confusion_matrix.png", dpi=130)
    plt.close()


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    texts, labels, responses = load_data()
    all_words = sorted({w for doc in texts for w in doc})

    # split data set menjadi train/test
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    train_set = [(build_features(t, all_words), y) for t, y in zip(X_train, y_train)]
    test_set = [(build_features(t, all_words), y) for t, y in zip(X_test, y_test)]

    # train model menggunakan naive bayes
    eval_clf = nltk.NaiveBayesClassifier.train(train_set)

    train_pred = [eval_clf.classify(f) for f, _ in train_set]
    test_pred = [eval_clf.classify(f) for f, _ in test_set]
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    tags = sorted(set(labels))
    report = classification_report(y_test, test_pred, labels=tags,
                                   output_dict=True, zero_division=0)

    print(f"Akurasi data latih : {train_acc:.2%}")
    print(f"Akurasi data uji   : {test_acc:.2%}\n")
    print(classification_report(y_test, test_pred, labels=tags, zero_division=0))

    # Visualisasi
    plot_distribution(labels)
    plot_accuracy(train_acc, test_acc)
    plot_classification_report(report)
    plot_confusion_matrix(y_test, test_pred, tags)
    print(f"Grafik tersimpan di folder '{REPORT_DIR}/'")

    # Artifact model, hasil akhir
    full_set = [(build_features(t, all_words), y) for t, y in zip(texts, labels)]
    final_clf = nltk.NaiveBayesClassifier.train(full_set)
    with open("model.pkl", "wb") as f:
        pickle.dump({"classifier": final_clf, "all_words": all_words, "responses": responses}, f)
    print("Model final (seluruh data) tersimpan di model.pkl")


if __name__ == "__main__":
    main()