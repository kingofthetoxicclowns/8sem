
import os
import time
import numpy as np
import textdistance
import tensorflow_hub as hub
import tensorflow as tf
import torch
from datasets import load_dataset
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import TruncatedSVD

# === Загрузка датасета STS-B ===
def load_stsb_data(sample_size=100):
    dataset = load_dataset("glue", "stsb", split="validation")
    pairs = []
    labels = []
    for item in dataset:
        s1, s2 = item["sentence1"], item["sentence2"]
        label = item["label"] / 5.0
        pairs.append((s1, s2))
        labels.append(label)
        if len(pairs) >= sample_size:
            break
    return pairs, labels

# === Строковые метрики ===
def damerau_levenshtein(s1, s2): return textdistance.damerau_levenshtein.distance(s1, s2)
def levenshtein(s1, s2): return textdistance.levenshtein.distance(s1, s2)
def hamming(s1, s2): return textdistance.hamming.distance(s1, s2) if len(s1) == len(s2) else None
def jaro_winkler(s1, s2): return textdistance.jaro_winkler.similarity(s1, s2)
def lcs(s1, s2): return textdistance.lcsseq.similarity(s1, s2)
def jaccard_q3(s1, s2): return textdistance.Jaccard(qval=3).similarity(s1, s2)
def cosine_q3(s1, s2): return textdistance.Cosine(qval=3).similarity(s1, s2)

# === Корпусные метрики ===
def tfidf_metrics(pairs):
    texts = [x for p in pairs for x in p]
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(texts)
    tfidf, eucl, manh = [], [], []
    for i in range(0, len(texts), 2):
        tfidf.append(cosine_similarity(x[i], x[i+1])[0][0])
        eucl.append(euclidean_distances(x[i], x[i+1])[0][0])
        manh.append(manhattan_distances(x[i], x[i+1])[0][0])
    return tfidf, eucl, manh

def lsa_metrics(pairs):
    texts = [x for p in pairs for x in p]
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=100)
    x_reduced = svd.fit_transform(x)
    scores = []
    for i in range(0, len(x_reduced), 2):
        scores.append(cosine_similarity([x_reduced[i]], [x_reduced[i+1]])[0][0])
    return scores

# === Семантические метрики ===
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

def elmo_model_scores(pairs):
    elmo = hub.load("https://tfhub.dev/google/elmo/3")
    scores = []
    for s1, s2 in pairs:
        emb1 = elmo.signatures['default'](tf.constant([s1]))['elmo']
        emb2 = elmo.signatures['default'](tf.constant([s2]))['elmo']
        emb1_avg = tf.reduce_mean(emb1, axis=1)
        emb2_avg = tf.reduce_mean(emb2, axis=1)
        sim = cosine_similarity(emb1_avg.numpy(), emb2_avg.numpy())[0][0]
        scores.append(sim)
    return scores

# === InferSent интеграция ===
from models import InferSent

def load_infersent_model():
    params_model = {
        'bsize': 64,
        'word_emb_dim': 300,
        'enc_lstm_dim': 2048,
        'pool_type': 'max',
        'dpout_model': 0.0,
        'version': 2
    }
    model = InferSent(params_model)
    model.load_state_dict(torch.load("infersent2.pkl"))
    model.set_w2v_path("glove.840B.300d.txt")
    model.build_vocab_k_words(K=100000)
    return model

def infersent_model_scores(pairs, model):
    scores = []
    for s1, s2 in pairs:
        emb1 = model.encode([s1])[0]
        emb2 = model.encode([s2])[0]
        sim = cosine_similarity([emb1], [emb2])[0][0]
        scores.append(sim)
    return scores

# === Корреляции ===
def compute_correlations(metric_scores, ground_truth):
    spearman_corr, _ = spearmanr(metric_scores, ground_truth)
    pearson_corr, _ = pearsonr(metric_scores, ground_truth)
    return spearman_corr, pearson_corr

# === Основной блок ===
if __name__ == "__main__":
    data_pairs, true_scores = load_stsb_data(sample_size=100)

    tfidf_cos, tfidf_euc, tfidf_man = tfidf_metrics(data_pairs)
    lsa = lsa_metrics(data_pairs)
    sbert = sbert_model_scores(data_pairs, "all-MiniLM-L6-v2")
    use = use_model_scores(data_pairs)
    glove = sbert_model_scores(data_pairs, "sentence-transformers/paraphrase-distilroberta-base-v1")
    word2vec = sbert_model_scores(data_pairs, "sentence-transformers/paraphrase-MiniLM-L6-v2")
    elmo = elmo_model_scores(data_pairs)
    infersent_model = load_infersent_model()
    infsent = infersent_model_scores(data_pairs, infersent_model)

    lev, dam, ham, jar, lcs_, jac, cos_q3 = [], [], [], [], [], [], []
    for text1, text2 in data_pairs:
        lev.append(levenshtein(text1, text2))
        dam.append(damerau_levenshtein(text1, text2))
        ham.append(hamming(text1, text2) or 0.0)
        jar.append(jaro_winkler(text1, text2))
        lcs_.append(lcs(text1, text2))
        jac.append(jaccard_q3(text1, text2))
        cos_q3.append(cosine_q3(text1, text2))

    all_metrics = {
        "Levenshtein": lev,
        "Damerau-Levenshtein": dam,
        "Hamming": ham,
        "Jaro-Winkler": jar,
        "LCS": lcs_,
        "Jaccard_q3": jac,
        "Cosine_q3": cos_q3,
        "TFIDF_Cosine": tfidf_cos,
        "TFIDF_Euclidean": tfidf_euc,
        "TFIDF_Manhattan": tfidf_man,
        "LSA": lsa,
        "Word2Vec": word2vec,
        "Glove": glove,
        "Elmo": elmo,
        "USE": use,
        "InferSent": infsent,
        "SBERT": sbert
    }

    print("=== Информативность (корреляция с ground-truth) ===")
    print(f"{'Метрика':<20} | {'Spearman':>9} | {'Pearson':>9}")
    print("-"*44)
    for name, scores in all_metrics.items():
        s, p = compute_correlations(scores, true_scores)
        print(f"{name:<20} | {s:>9.4f} | {p:>9.4f}")
