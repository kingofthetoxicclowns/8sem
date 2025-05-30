import time
import psutil
import csv
import os
import numpy as np
import textdistance
import tensorflow_hub as hub
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sentence_transformers import SentenceTransformer, util

# === Загрузка данных SNLI ===
def load_snli_pairs(sample_size=100, only_long_words=False):
    dataset = load_dataset("snli", split="train")
    pairs = []
    for item in dataset:
        if item["label"] != 0:
            continue
        s1, s2 = item["premise"], item["hypothesis"]
        if s1 and s2:
            if only_long_words:
                if all(len(w) > 7 for w in (s1 + " " + s2).split()):
                    pairs.append((s1, s2))
            else:
                pairs.append((s1, s2))
        if len(pairs) >= sample_size:
            break
    return pairs

# === Метрики ===

# Строковые метрики через textdistance
def damerau_levenshtein(s1, s2): return textdistance.damerau_levenshtein.distance(s1, s2)
def levenshtein(s1, s2): return textdistance.levenshtein.distance(s1, s2)
def hamming(s1, s2): return textdistance.hamming.distance(s1, s2) if len(s1) == len(s2) else None
def jaro_winkler(s1, s2): return textdistance.jaro_winkler.similarity(s1, s2)
def lcs(s1, s2): return textdistance.lcsseq.similarity(s1, s2)
def jaccard_q3(s1, s2): return textdistance.Jaccard(qval=3).similarity(s1, s2)
def cosine_q3(s1, s2): return textdistance.Cosine(qval=3).similarity(s1, s2)

# Корпусные метрики
def tfidf_metrics(pairs):
    texts = [x for p in pairs for x in p]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    tfidf = []
    eucl = []
    manh = []
    for i in range(0, len(texts), 2):
        tfidf.append(cosine_similarity(X[i], X[i+1])[0][0])
        eucl.append(euclidean_distances(X[i], X[i+1])[0][0])
        manh.append(manhattan_distances(X[i], X[i+1])[0][0])
    return tfidf, eucl, manh

def lsa_metrics(pairs):
    from sklearn.decomposition import TruncatedSVD
    texts = [x for p in pairs for x in p]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=100)
    X_reduced = svd.fit_transform(X)
    scores = []
    for i in range(0, len(X_reduced), 2):
        scores.append(cosine_similarity([X_reduced[i]], [X_reduced[i+1]])[0][0])
    return scores

# Семантические метрики
def sbert_model_scores(pairs, model_name):
    model = SentenceTransformer(model_name)
    scores = []
    for s1, s2 in pairs:
        emb1 = model.encode(s1, convert_to_tensor=True)
        emb2 = model.encode(s2, convert_to_tensor=True)
        scores.append(util.pytorch_cos_sim(emb1, emb2).item())
    return scores

def use_model_scores(pairs):
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    scores = []
    for s1, s2 in pairs:
        emb = model([s1, s2])
        sim = cosine_similarity(emb[0:1].numpy(), emb[1:2].numpy())[0][0]
        scores.append(sim)
    return scores

# === Основной процесс ===
def evaluate_all(pairs):
    tfidf_cos, tfidf_euc, tfidf_man = tfidf_metrics(pairs)
    lsa = lsa_metrics(pairs)
    sbert = sbert_model_scores(pairs, "all-MiniLM-L6-v2")
    use = use_model_scores(pairs)
    glove = sbert_model_scores(pairs, "sentence-transformers/paraphrase-distilroberta-base-v1")
    word2vec = sbert_model_scores(pairs, "sentence-transformers/paraphrase-MiniLM-L6-v2")
    elmo = [0.0]*len(pairs)  # заглушка
    infsent = [0.0]*len(pairs)  # заглушка

    results = []
    for i, (s1, s2) in enumerate(pairs):
        start_time = time.time()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)

        row = [
            i, s1, s2,
            levenshtein(s1, s2),
            damerau_levenshtein(s1, s2),
            hamming(s1, s2),
            jaro_winkler(s1, s2),
            lcs(s1, s2),
            jaccard_q3(s1, s2),
            cosine_q3(s1, s2),
            tfidf_cos[i], tfidf_euc[i], tfidf_man[i],
            lsa[i],
            word2vec[i], glove[i], elmo[i], use[i], infsent[i], sbert[i]
        ]

        mem_after = process.memory_info().rss / (1024 * 1024)
        end_time = time.time()

        row += [end_time - start_time, mem_after - mem_before]
        results.append(row)
    return results

# === Сохранение в CSV ===
def save_to_csv(results, filepath):
    headers = [
        "pair_index", "text1", "text2",
        "Levenshtein", "Damerau_levenshtein", "Hamming", "Jaro_winkler", "Lcs", "Jaccard_q3", "Cosine_q3",
        "Tfidf", "Tfidf_euclidean", "Tfidf_manhattan", "Lsa",
        "Word2vec", "Glove", "Elmo", "Use", "Infsent", "Sbert",
        "Time_seconds", "Memory_MB"
    ]
    with open(filepath, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)

if __name__ == "__main__":
    sample_pairs = load_snli_pairs(sample_size=50)
    results = evaluate_all(sample_pairs)
    save_to_csv(results, "similarity_metrics_results.csv")
