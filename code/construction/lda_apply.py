# -*- coding: utf-8 -*-
"""
lda_apply.py

Applies the fitted nine-topic LDA model (data/lda_model/, estimated by lda_fit.py)
to the 2,330 documents of the analysis corpus and writes the document x topic
matrix used throughout the article (data/02_topic_model_document_topics.csv).

As the article explains (Section 5.1), the model was estimated before
near-duplicate documents were removed and then applied to the 2,330 documents
retained after deduplication. Text preprocessing is identical to lda_fit.py,
which this script imports. Documents with fewer than ten content tokens after
preprocessing are excluded.

The composite ideology score is
    ideology_score = T1 + T3 + T9 - T5 - T8,
the probability mass on the social engagement topics minus the mass on the
spiritual/institutional topics; higher values denote more social engagement.

Inputs (PROJECT_ROOT):
  corpus_final/master_corpus_analysis.csv   full text, not distributed
  data/lda_model/lda_model_full             fitted model, distributed
  data/lda_model/dictionary_full.gensim     dictionary, distributed
Outputs (current folder):
  02_topic_model_document_topics.csv
  02d_topic_summary.csv

Requires: Python 3.9+, pandas, numpy, gensim, spaCy (Spanish stopwords)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from gensim import corpora, models

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lda_fit as T  # noqa: E402  (shared preprocessing)

# PROJECT_ROOT is the folder that holds corpus_final/, the full-text corpus.
# The full text is not distributed (press copyright); see README.
PROJECT_ROOT = "."
CORPUS = os.path.join(PROJECT_ROOT, "corpus_final", "master_corpus_analysis.csv")
MODEL_DIR = os.path.join(HERE, "..", "..", "data", "lda_model")


def main():
    lda = models.LdaModel.load(os.path.join(MODEL_DIR, "lda_model_full"))
    dictionary = corpora.Dictionary.load(os.path.join(MODEL_DIR, "dictionary_full.gensim"))
    k = lda.num_topics

    df = pd.read_csv(CORPUS, encoding="utf-8-sig", low_memory=False)
    df["tokens"] = df.text.fillna("").apply(T.preprocess)
    df = df[df.tokens.map(len) >= 10].reset_index(drop=True)

    dist = np.zeros((len(df), k))
    for i, toks in enumerate(df.tokens):
        for t, p in lda.get_document_topics(dictionary.doc2bow(toks), minimum_probability=0.0):
            dist[i, t] = p
    probs = pd.DataFrame(dist, columns=[f"topic_{i + 1}_prob" for i in range(k)]).round(6)

    doc = pd.concat([
        df[["doc_id", "source", "subsource", "bishop", "name_normalized", "author_type",
            "territory", "date", "year", "title", "token_count"]].reset_index(drop=True),
        probs], axis=1)
    doc["primary_topic"] = probs.values.argmax(axis=1) + 1
    doc["primary_topic_prob"] = probs.values.max(axis=1).round(6)
    doc["ideology_score"] = (doc[["topic_1_prob", "topic_3_prob", "topic_9_prob"]].sum(axis=1)
                             - doc[["topic_5_prob", "topic_8_prob"]].sum(axis=1))
    doc.to_csv("02_topic_model_document_topics.csv", index=False, encoding="utf-8-sig")

    summary = (doc.groupby("primary_topic")
               .agg(n_documents=("doc_id", "size"), mean_prob=("primary_topic_prob", "mean"))
               .reset_index().rename(columns={"primary_topic": "topic"}))
    summary["percent"] = (summary.n_documents / len(doc) * 100).round(1)
    summary["mean_prob"] = summary.mean_prob.round(3)
    summary.to_csv("02d_topic_summary.csv", index=False, encoding="utf-8-sig")
    print(f"{len(doc)} documents, {k} topics")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
