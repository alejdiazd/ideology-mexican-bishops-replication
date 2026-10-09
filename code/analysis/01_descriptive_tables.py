# -*- coding: utf-8 -*-
"""
01_descriptive_tables.py

Reproduces the descriptive and test tables of the article and its supplement
from the files in ../../data:

  Table 1   corpus composition by source
  Table 2   the nine LDA topics (documents, share, top terms, mean probability)
  Table 3   principal consecrators (direct and two-generation counts)
  Table 4   tests on the full corpus (H4, H5, validity)
  Table 5   mean document ideology by region
  Table 6   mean document ideology by decade
  Table S1  LDA coherence by number of topics
  Table S2  primary topic by decade
  Table S5  regional document ideology by source type
  Table S6  institutional vs. individual voice by source type

Tables S3 and S4 are the data files 03_wordfish_ideal_points_N39.csv and
03c_wordfish_church_vs_press_N39.csv. Tables 7-9, S7 and S8 come from
02_regression_tables.R.

Run from this folder:  python 01_descriptive_tables.py
Output: ../../output/tables/*.csv
Requires: Python 3.9+, pandas, numpy, scipy
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..", "..", "output", "tables")
os.makedirs(OUT, exist_ok=True)


def rd(name):
    return pd.read_csv(os.path.join(DATA, name), encoding="utf-8-sig", low_memory=False)


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False, encoding="utf-8-sig")
    print(f"\n=== {name}\n{df.to_string(index=False)}")


corp = rd("01_corpus_metadata_N2330.csv")
topics = rd("02_topic_model_document_topics.csv")
panel = rd("06_document_panel_FINAL.csv")
wf = rd("03_wordfish_ideal_points_N39.csv")
nodes = rd("04_consecration_network_nodes.csv")
edges = rd("04b_consecration_edges_clean.csv")

# Church-generated sources; every other source is press coverage.
CHURCH = {"diocesis", "CEM", "mensajero", "voices", "messages2018"}
POS, NEG = ["topic_1_prob", "topic_3_prob", "topic_9_prob"], ["topic_5_prob", "topic_8_prob"]
topics["ideology_score"] = topics[POS].sum(axis=1) - topics[NEG].sum(axis=1)
topics["bucket"] = np.where(topics.source.isin(CHURCH), "church", "press")
REGION = {"Norte": "North", "Sur": "South", "Bajio-Occidente": "West (Bajío-Occidente)",
          "Golfo-Sureste": "Gulf-Southeast", "Centro": "Central"}


def pval(p):
    return "<0.0001" if p < 1e-4 else f"{p:.3f}"


def cohen_d(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


# ---------------------------------------------------------------- Table 1
t1 = corp.source.value_counts().rename_axis("source").reset_index(name="documents")
save(t1, "table1_corpus_by_source.csv")

# ---------------------------------------------------------------- Table 2
terms = rd("02c_topic_terms_top20.csv")
top8 = (terms[terms["rank"] <= 8].sort_values(["topic", "rank"])
        .groupby("topic").term.apply(lambda s: ", ".join(s)))
t2 = rd("02d_topic_summary.csv").sort_values("topic")
t2["top_terms"] = t2.topic.map(top8)
save(t2, "table2_topics.csv")

# ---------------------------------------------------------------- Table 3
prin = edges[edges.role == "principal_consecrator"]
kids = prin.groupby("from_key").to_key.apply(set).to_dict()


def two_generations(key):
    d1 = kids.get(key, set())
    d2 = set().union(*[kids.get(c, set()) for c in d1]) if d1 else set()
    return len((d1 | d2) - {key})


t3 = nodes.sort_values("n_principal_consecrations", ascending=False).head(7)
t3 = pd.DataFrame({"consecrator": t3.node,
                   "direct": t3.n_principal_consecrations.astype(int),
                   "two_generation_nodes": t3.node_key.map(two_generations)})
save(t3, "table3_principal_consecrators.csv")

# ---------------------------------------------------------------- Table 4
F, pF = stats.f_oneway(*[g.ideology_index.values for _, g in panel.groupby("region5")])
size = panel.groupby("territory").agg(n=("doc_id", "size"), m=("ideology_index", "mean"))
size = size[size.n >= 5]
r_size, p_size = stats.pearsonr(size.n, size.m)
inst = topics[topics.author_type == "institutional"].ideology_score
indiv = topics[topics.author_type == "individual"].ideology_score
_, p_h5 = stats.ttest_ind(inst, indiv, equal_var=False)
ch = topics[topics.bucket == "church"]
inst_c = ch[ch.author_type == "institutional"].ideology_score
indiv_c = ch[ch.author_type == "individual"].ideology_score
_, p_h5c = stats.ttest_ind(inst_c, indiv_c, equal_var=False)
# Validity: actor-level mean of the topic score against Wordfish, actors with >= 10 documents
actor = topics.groupby("name_normalized").ideology_score.agg(["mean", "size"])
actor = actor[actor["size"] >= 10]
val = wf.set_index("bishop").join(actor, how="inner").dropna(subset=["mean", "theta_oriented"])
r_val, p_val = stats.pearsonr(val["mean"], val["theta_oriented"])
t4 = pd.DataFrame([
    ["H4 Regional variation predicts ideology", "ANOVA (5 regions)", f"F = {F:.1f}", pval(pF)],
    ["H4 Diocese size predicts ideology", f"Pearson r ({len(size)} dioceses)", f"r = {r_size:.2f}", pval(p_size)],
    ["H5 Institutional vs. individual voice", f"t-test (N = {len(inst) + len(indiv)})",
     f"d = {cohen_d(inst, indiv):.2f}", pval(p_h5)],
    ["H5 (Church only sources)", f"t-test (N = {len(inst_c) + len(indiv_c)})",
     f"d = {cohen_d(inst_c, indiv_c):.2f}", pval(p_h5c)],
    ["Validity: topic ideology vs. Wordfish", f"Pearson r ({len(val)} actors)",
     f"r = {r_val:.2f}", pval(p_val)],
], columns=["hypothesis", "test", "effect", "p_value"])
save(t4, "table4_full_corpus_tests.csv")

# ---------------------------------------------------------------- Table 5
t5 = (panel.groupby("region5").ideology_index.agg(n_docs="size", mean_ideology="mean")
      .sort_values("mean_ideology", ascending=False).round(2).reset_index())
t5["region5"] = t5.region5.map(REGION)
save(t5, "table5_ideology_by_region.csv")

# ---------------------------------------------------------------- Table 6
topics["decade"] = pd.to_numeric(topics.year, errors="coerce") // 10 * 10
t6 = (topics.dropna(subset=["decade"]).groupby("decade").ideology_score
      .agg(n_docs="size", mean_ideology="mean").round(2).reset_index())
t6["decade"] = t6.decade.astype(int).astype(str) + "s"
save(t6, "table6_ideology_by_decade.csv")

# ---------------------------------------------------------------- Table S1
save(rd("02b_lda_coherence_by_k.csv").round(3), "tableS1_lda_coherence.csv")

# ---------------------------------------------------------------- Table S2
s2 = (topics.dropna(subset=["decade"])
      .pivot_table(index="decade", columns="primary_topic", values="doc_id", aggfunc="count")
      .reindex(columns=range(1, 10)).fillna(0).astype(int))
s2.columns = [f"T{c}" for c in s2.columns]
s2.index = s2.index.astype(int).astype(str) + "s"
save(s2.reset_index(), "tableS2_primary_topic_by_decade.csv")

# ---------------------------------------------------------------- Table S5
g = panel.groupby(["region5", "bucket"]).ideology_index.agg(n="size", mean="mean").round(2)
s5 = g.unstack("bucket")
s5.columns = [f"{stat}_{bucket}" for stat, bucket in s5.columns]
s5 = s5.loc[panel.groupby("region5").ideology_index.mean().sort_values(ascending=False).index]
s5.index = s5.index.map(REGION)
save(s5.reset_index(), "tableS5_region_by_source.csv")


# ---------------------------------------------------------------- Table S6
def h5(sub, label):
    a = sub[sub.author_type == "institutional"].ideology_score
    b = sub[sub.author_type == "individual"].ideology_score
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return [label, round(a.mean(), 3), len(a), round(b.mean(), 3), len(b),
            round(t, 2), pval(p), round(cohen_d(a, b), 2)]


s6 = pd.DataFrame([h5(topics, "Full corpus"), h5(ch, "Church only"),
                   h5(topics[topics.bucket == "press"], "Press only")],
                  columns=["subset", "institutional_mean", "institutional_n", "individual_mean",
                           "individual_n", "t", "p", "cohens_d"])
save(s6, "tableS6_institutional_vs_individual_by_source.csv")
