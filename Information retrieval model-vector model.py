#!/usr/bin/env python
# coding: utf-8

# In[ ]:


documents = [
    "cat runs behind rat",     
    "dog runs behind cat",                 
    "dog , cat & rat runs behind cheese"                   
]

query = "cat cheese"

vocab = ["cat", "runs", "behind", "rat", "dog", "cheese"]


# In[ ]:


documents = [
    "binary inormation retrieval model",     
    "vector retrieval model",                 
    "extended binary model"                   
]

query = "extended binary"

vocab = ["binary", "inormation", "retrieval", "model", "vector", "extended"]


# In[ ]:


documents = [
    "health observance for march awareness",     
    "health oriented calendar",                  
    "awareness news for march awareness"                   
]

query = "march calendar"

vocab = ["health", "observance", "march", "awareness", "oriented", "calendar", "news"]


# In[ ]:


import numpy as np
import pandas as pd
from math import log10, sqrt

documents = [
    "cat runs behind rat",     
    "dog runs behind cat",                 
    "dog , cat & rat runs behind cheese"                   
]

query = "sport"

vocab = ["cat", "runs", "behind", "rat", "dog", "cheese"]


def compute_tf(text, vocab):
    words = text.split()
    return [words.count(word) for word in vocab]

def normalize_tf(tf_vector):
    max_freq = max(tf_vector) if max(tf_vector) != 0 else 1
    return [round(freq / max_freq, 2) for freq in tf_vector]

tf_docs_raw = [compute_tf(doc, vocab) for doc in documents]
tf_docs = np.array([normalize_tf(tf) for tf in tf_docs_raw])
tf_query_raw = compute_tf(query, vocab)
tf_query = normalize_tf(tf_query_raw)

tf_df0 = pd.DataFrame(tf_docs_raw, columns=vocab, index=["D1", "D2", "D3"])
print("=== Term Frequency (TF) Table ===")
print(tf_df0.round(2))
print()

df = np.sum(np.array(tf_docs_raw) > 0, axis=0)

N = len(documents)
idf = [log10(1 + N / df_i) if df_i != 0 else 0 for df_i in df]

tfidf_docs = tf_docs * idf
tfidf_query = (np.array(tf_query)) * idf

tf_df = pd.DataFrame(tf_docs, columns=vocab, index=["D1", "D2", "D3"])
print("=== Normalized Term Frequency (TF) Table ===")
print(tf_df.round(2))
print()

idf_df = pd.DataFrame({
    "Keyword": vocab,
    "DF": df,
    "IDF": [f"log(1 + {N}/{df_i}) = {round(idf_i, 2)}" for df_i, idf_i in zip(df, idf)]
})
print("=== Inverse Document Frequency (IDF) Table ===")
print(idf_df)
print()


tfidf_df = pd.DataFrame(tfidf_docs, columns=vocab, index=["D1", "D2", "D3"])
print("=== Normalized TF-IDF Table ===")
print(tfidf_df.round(2))
print()


query_df = pd.DataFrame([tfidf_query], columns=vocab, index=["Query"])
print("=== Normalized TF-IDF Query Vector ===")
print(query_df.round(2))
print()


def cosine_similarity(doc_vector, query_vector):
    dot = sum(d * q for d, q in zip(doc_vector, query_vector))
    norm_doc = sqrt(sum(d**2 for d in doc_vector))
    norm_query = sqrt(sum(q**2 for q in query_vector))
    similarity = dot / (norm_doc * norm_query) if norm_doc and norm_query else 0
    return dot, norm_doc, norm_query, similarity

similarities = {}

for i, doc_vector in enumerate(tfidf_docs):
    doc_name = f"D{i+1}"
    dot, norm_d, norm_q, sim = cosine_similarity(doc_vector, tfidf_query)
    similarities[doc_name] = sim

    df = pd.DataFrame({
        "Keyword": vocab,
        doc_name: np.round(doc_vector, 2),
        "Query": np.round(tfidf_query, 2),
        f"{doc_name} x Q": np.round([d * q for d, q in zip(doc_vector, tfidf_query)], 2)
    })

    print(f"\n=== TF-IDF Table for {doc_name} and Query ===")
    print(df.to_string(index=False))
    print(f"\nDot Product (Σ {doc_name} x Q) = {round(dot, 2)}")
    print(f"||{doc_name}|| = {round(norm_d, 2)}")
    print(f"||Query|| = {round(norm_q, 2)}")
    print(f"Cosine Similarity ({doc_name}, Query) = {round(sim, 2)}")
    print("-" * 50)

best_doc = max(similarities, key=similarities.get)
print(f"\nMost similar document to the query is **{best_doc}** with similarity {round(similarities[best_doc], 2)}")

