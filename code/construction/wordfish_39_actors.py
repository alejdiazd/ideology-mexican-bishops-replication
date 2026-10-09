# -*- coding: utf-8 -*-
"""
Wordfish Ideology Estimation – Mexican Bishops (full corpus, 2,460 docs)
=========================================================================
Inputs
------
  corpus_final/master_corpus_final.csv              – texts + bishop + territory
  topic_model_results_full/document_topics_full.csv – 9-topic probabilities

Pipeline
--------
  1. Load & merge on doc_id
  2. Assign Unknown / obispo → CEM (institutional voice)
  3. Drop noise names (iglesia católica, etc.)
  4. Filter Topic 8 (homilética y liturgia) documents from CEM source;
     individual bishops retain all documents
  5. Require ≥ MIN_DOCS per author after filtering
  6. Aggregate texts per author
  7. Wordfish (Poisson coordinate descent, Spanish stopwords)
  8. Topic-space PCA (complementary view)
  9. Outputs: CSVs + 5 plots in wordfish_results_full/

Reference: Lauderdale & Herzog (2016). Political Analysis 24(4):528–545.
"""

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler

try:
    from spacy.lang.es.stop_words import STOP_WORDS as SPACY_ES
except ImportError:
    SPACY_ES = set()

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# ── paths ──────────────────────────────────────────────────────────────────────
# PROJECT_ROOT is the folder that holds corpus_final/, the full-text corpus.
# The full text is not distributed (press copyright); see README.
BASE      = Path(".")  # PROJECT_ROOT
CORPUS    = BASE / "corpus_final" / "master_corpus_final.csv"
TOPICS    = BASE / "topic_model_results_full" / "document_topics_full.csv"
OUT_DIR   = BASE / "wordfish_results_full"
OUT_DIR.mkdir(exist_ok=True)

MIN_DOCS  = 10      # minimum docs per author to enter Wordfish (main list)
MAX_VOCAB = 3000    # vocabulary ceiling
MAX_ITER  = 200     # coordinate-descent iterations

# ── stopwords ──────────────────────────────────────────────────────────────────
EXTRA = {
    "así","aún","aquí","allá","allí","ahí","sino","pues","mas","ya","sí",
    "ni","oh","ah","eh","hay","hoy","haber","ser","estar","tener","hacer",
    "ir","ver","dar","saber","querer","poder","deber",
    "enero","febrero","marzo","abril","mayo","junio","julio","agosto",
    "septiembre","octubre","noviembre","diciembre",
    "lunes","martes","miércoles","jueves","viernes","sábado","domingo",
    "uno","dos","tres","cuatro","cinco","seis","siete","ocho","nueve","diez",
    "cien","mil","primero","segundo","tercero","primera","segunda","tercera",
    "fue","han","son","será","están","tenemos","puede","pueden",
    "tiene","tienen","hace","hacen","hecho","decir","dicho",
    "escrito","escrita","http","www","com","flickr","photos","video",
    "secretaría","secretario","comunicación","prensa",
    "mons","monseñor","obispo","arzobispo","cardenal",
    "señor","señora","señores",
}
STOPS = {w.lower() for w in SPACY_ES} | {w.lower() for w in EXTRA}
print(f"Stopword list: {len(STOPS)} words")

# ── 9-topic labels (full corpus model) ────────────────────────────────────────
TOPIC_LABELS = {
    1: "Iglesia y sociedad",
    2: "Jerarquía eclesiástica",
    3: "Paz, violencia y seguridad",
    4: "Política electoral",
    5: "Vida pastoral y diocesana",
    6: "Organización episcopal",
    7: "Iglesia en medios",
    8: "Homilética y liturgia",   # filtered from CEM docs
    9: "Chiapas e indigenismo",
}

HOMILETIC_TOPIC = 8   # excluded from CEM-sourced documents

# names that are not individual bishops
NOISE_NAMES = {
    "iglesia católica", "iglesia cat�lica",
    "desconocido", "desconocida",
    "episcopado", "vaticano", "juan pablo ii",
    "iglesia", "cat�lica",
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & MERGE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/8] Loading and merging data...")
master = pd.read_csv(CORPUS,  encoding="latin-1")
topics = pd.read_csv(TOPICS,  encoding="utf-8-sig")

# Merge text into topic-model output (topic model is the authoritative doc list)
# master also has 'source', so only pull 'text' to avoid duplication
df = topics.merge(
    master[["doc_id","text"]],
    on="doc_id", how="left"
)
df["text"] = df["text"].fillna("").astype(str)
print(f"  Loaded {len(df)} documents after merge")

# ── clean name_normalized ──────────────────────────────────────────────────────
df["name_normalized"] = (
    df["name_normalized"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
    .str.rstrip(",")
    .str.strip()
)

# ── assign Unknown / obispo → CEM ─────────────────────────────────────────────
reassign = {"Unknown", "unknown", "obispo", "OBISPO"}
df.loc[df["name_normalized"].isin(reassign), "name_normalized"] = "CEM"
print(f"  After Unknown→CEM reassignment: {df['name_normalized'].nunique()} unique names")

# ── drop noise names ──────────────────────────────────────────────────────────
df = df[~df["name_normalized"].str.lower().isin(
    {n.lower() for n in NOISE_NAMES}
)].reset_index(drop=True)
print(f"  After noise removal: {len(df)} documents")

# ══════════════════════════════════════════════════════════════════════════════
# 2. FILTER HOMILETIC/LITURGICAL CEM DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/8] Filtering homiletic/liturgical CEM documents (Topic 8)...")

is_cem_source = df["source"].str.upper() == "CEM"
is_homiletic  = df["primary_topic"] == HOMILETIC_TOPIC

before = len(df)
df = df[~(is_cem_source & is_homiletic)].reset_index(drop=True)
removed = before - len(df)
print(f"  Removed {removed} CEM-source homiletic documents (Topic 8)")
print(f"  Remaining: {len(df)} documents")

# ══════════════════════════════════════════════════════════════════════════════
# 3. TOPIC PROBABILITY SETUP
# ══════════════════════════════════════════════════════════════════════════════
prob_cols = [f"topic_{t}_prob" for t in range(1, 10)]
for c in prob_cols:
    df[c] = df[c].fillna(0)

# ══════════════════════════════════════════════════════════════════════════════
# 4. BISHOP-LEVEL AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/8] Aggregating texts and topic profiles per bishop/actor...")

grp = df.groupby("name_normalized")

def mode_val(s):
    m = s.mode()
    return m.iloc[0] if len(m) > 0 else s.iloc[0]

bishop_df = pd.DataFrame({
    "bishop":      list(grp.groups.keys()),
    "n_docs":      grp["text"].count().values,
    "text_concat": grp["text"].apply(lambda s: " ".join(s.dropna().astype(str))).values,
    "territory":   grp["territory"].apply(mode_val).values,
    "year_min":    grp["year"].min().values,
    "year_max":    grp["year"].max().values,
})
for c in prob_cols:
    bishop_df[c] = grp[c].mean().values

bishop_df["dominant_topic"] = (
    bishop_df[prob_cols].idxmax(axis=1)
    .str.replace("topic_","").str.replace("_prob","").astype(int)
)
bishop_df["dominant_topic_label"] = bishop_df["dominant_topic"].map(TOPIC_LABELS)

bishop_df["is_cem"] = bishop_df["bishop"] == "CEM"

# ── "Main list" filter ────────────────────────────────────────────────────────
# Keep bishops who: (a) have a named territory, OR (b) are CEM
# This filters out the ~80 small-corpus bishops from the Nuevo Casas Grandes
# diocesan records who lack territory metadata and would contaminate the scale.
has_named_territory = (
    bishop_df["territory"].notna()
    & (bishop_df["territory"].astype(str).str.strip().str.lower() != "nan")
    & (bishop_df["territory"].astype(str).str.strip() != "")
)
main_list_mask = has_named_territory | bishop_df["is_cem"]
bishop_df = bishop_df[main_list_mask].copy().reset_index(drop=True)
print(f"  Main-list filter: {len(bishop_df)} actors with named territory or CEM")

# Filter to actors with enough documents
bishop_enough = (
    bishop_df[bishop_df["n_docs"] >= MIN_DOCS]
    .copy()
    .reset_index(drop=True)
)
print(f"  Actors with ≥{MIN_DOCS} docs: {len(bishop_enough)} "
      f"(of {len(bishop_df)} unique names)")
print(f"  Covers {bishop_enough['n_docs'].sum()} documents "
      f"({100*bishop_enough['n_docs'].sum()/len(df):.1f}% of filtered corpus)")
print(f"  CEM is included: {'CEM' in bishop_enough['bishop'].values}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. WORDFISH  (Poisson coordinate descent on aggregated texts)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[4/8] Building document-term matrix ({len(bishop_enough)} actors)...")

vectorizer = CountVectorizer(
    max_features  = MAX_VOCAB,
    min_df        = 6,          # word must appear in ≥6 of 114 actor corpora
    max_df        = 0.90,
    stop_words    = list(STOPS),
    token_pattern = r"(?u)\b[a-záéíóúüñ]{5,}\b",  # min 5 chars removes fragments
    lowercase     = True,
)
DTM  = vectorizer.fit_transform(bishop_enough["text_concat"]).toarray().astype(float)
vocab = vectorizer.get_feature_names_out()
n_doc, n_word = DTM.shape
print(f"  DTM shape: {n_doc} actors × {n_word} words")


def _ca_init(dtm):
    grand = dtm.sum()
    P   = dtm / grand
    r   = P.sum(axis=1, keepdims=True)
    c   = P.sum(axis=0, keepdims=True)
    S   = (P - r @ c) / np.sqrt(r @ c + 1e-12)
    svd = TruncatedSVD(n_components=1, random_state=42)
    theta = svd.fit_transform(S).flatten()
    return (theta - theta.mean()) / (theta.std() + 1e-12)


def wordfish(dtm, max_iter=MAX_ITER, lr=0.005, tol=1e-4):
    """
    Poisson Wordfish:  log E[count_{dw}] = alpha_w + beta_w * theta_d
    Estimated by coordinate descent.
    Returns theta (ideal points), alpha (word fixed effects),
    beta (word discrimination weights), log-likelihood history.
    """
    n_d, n_w = dtm.shape
    theta = _ca_init(dtm)
    alpha = np.log(dtm.mean(axis=0).clip(min=0.01))
    beta  = np.zeros(n_w)
    ll_hist = []

    for it in range(max_iter):
        # update word params
        for w in range(n_w):
            counts = dtm[:, w]
            if counts.sum() < 1:
                continue
            mu  = np.exp(alpha[w] + beta[w] * theta).clip(1e-10)
            W   = mu
            X   = np.column_stack([np.ones(n_d), theta])
            lhs = X.T @ (W[:, None] * X) + np.eye(2) * 1e-6
            z   = np.log(mu) + (counts - mu) / mu
            rhs = X.T @ (W * z)
            try:
                params = np.linalg.solve(lhs, rhs)
                alpha[w] = params[0]
                beta[w]  = params[1]
            except np.linalg.LinAlgError:
                pass

        # update document positions
        for d in range(n_d):
            counts = dtm[d, :]
            mu     = np.exp(alpha + beta * theta[d]).clip(1e-10)
            grad   = np.sum(beta * (counts - mu))
            hess   = -np.sum(beta**2 * mu)
            if abs(hess) > 1e-10:
                theta[d] -= lr * grad / hess

        theta = (theta - theta.mean()) / (theta.std() + 1e-12)

        mu_all = np.exp(alpha[None, :] + beta[None, :] * theta[:, None]).clip(1e-10)
        ll = (dtm * np.log(mu_all) - mu_all - dtm * np.log(dtm.clip(1e-10))).sum()
        ll_hist.append(ll)

        if it > 5 and abs(ll_hist[-1] - ll_hist[-2]) < tol:
            print(f"    Converged at iteration {it}  (ΔLL={ll_hist[-1]-ll_hist[-2]:.6f})")
            break
        if it % 25 == 0:
            print(f"    iter {it:4d}  LL={ll:.1f}")

    return theta, alpha, beta, ll_hist


print(f"\n[5/8] Running Wordfish (max {MAX_ITER} iterations)...")
theta, alpha_w, beta_w, ll_hist = wordfish(DTM)
bishop_enough["wordfish_score"] = theta

# orient: progressive (liberation/Chiapas) → negative pole
corr_orient = np.corrcoef(theta, bishop_enough["topic_9_prob"].values)[0, 1]
if corr_orient < 0:
    bishop_enough["wordfish_score"] = -theta
    beta_w = -beta_w
    print("  (Scale flipped: Chiapas/liberation theology → negative/progressive)")

# Standard errors via document-level projection
doc_level_scores = {}
for bname, bgrp in df[df["name_normalized"].isin(bishop_enough["bishop"])].groupby("name_normalized"):
    texts_b  = bgrp["text"].fillna("").astype(str).tolist()
    dtm_b    = vectorizer.transform(texts_b).toarray().astype(float)
    num      = (dtm_b * beta_w[None, :]).sum(axis=1)
    den      = (dtm_b * beta_w[None, :] ** 2).sum(axis=1).clip(1e-12)
    doc_level_scores[bname] = num / den

bishop_enough["wordfish_se"] = bishop_enough["bishop"].apply(
    lambda b: doc_level_scores.get(b, np.array([0])).std()
)

# ══════════════════════════════════════════════════════════════════════════════
# 6. TOPIC-SPACE PCA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/8] Topic-space PCA...")

topic_matrix = bishop_enough[prob_cols].values
scaler       = StandardScaler()
pca          = PCA(n_components=2, random_state=42)
pca_scores   = pca.fit_transform(scaler.fit_transform(topic_matrix))

bishop_enough["topic_pca_dim1"] = pca_scores[:, 0]
bishop_enough["topic_pca_dim2"] = pca_scores[:, 1]
print(f"  PCA variance: Dim1={pca.explained_variance_ratio_[0]:.1%}  "
      f"Dim2={pca.explained_variance_ratio_[1]:.1%}")

loadings_df = pd.DataFrame(
    pca.components_.T,
    index=[TOPIC_LABELS[i] for i in range(1, 10)],
    columns=["Dim1", "Dim2"],
).round(3)
loadings_df.to_csv(OUT_DIR / "topic_pca_loadings.csv", encoding="utf-8-sig")
print(loadings_df.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE CSVs
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/8] Saving results...")

rename_map = {f"topic_{t}_prob": f"topic_{t}_pct" for t in range(1, 10)}
cols_out = (
    ["bishop","territory","n_docs","year_min","year_max","is_cem",
     "wordfish_score","wordfish_se","topic_pca_dim1","topic_pca_dim2",
     "dominant_topic","dominant_topic_label"]
    + prob_cols
)
scores_out = (
    bishop_enough[cols_out]
    .sort_values("wordfish_score")
    .rename(columns=rename_map)
    .copy()
)
scores_out.to_csv(OUT_DIR / "wordfish_scores_full.csv", index=False, encoding="utf-8-sig")
print(f"  [saved] wordfish_scores_full.csv  ({len(scores_out)} actors)")

words_df = pd.DataFrame({
    "word":    vocab,
    "alpha":   alpha_w.round(4),
    "beta":    beta_w.round(4),
    "abs_beta": np.abs(beta_w).round(4),
}).sort_values("abs_beta", ascending=False)
words_df.to_csv(OUT_DIR / "wordfish_word_parameters.csv", index=False, encoding="utf-8-sig")
print(f"  [saved] wordfish_word_parameters.csv  ({len(words_df)} words)")

pd.DataFrame({"iteration": range(len(ll_hist)), "log_likelihood": ll_hist}).to_csv(
    OUT_DIR / "wordfish_convergence.csv", index=False, encoding="utf-8-sig"
)

# print table
print("\n  WORDFISH IDEAL POINTS (sorted progressive → conservative):")
print(f"  {'Actor':<45s}  {'Score':>7s}  {'SE':>5s}  {'N':>4s}  {'Territory'}")
print(f"  {'-'*45}  {'-'*7}  {'-'*5}  {'-'*4}  {'-'*22}")
for _, row in scores_out.iterrows():
    flag = " [CEM]" if row["is_cem"] else ""
    print(f"  {row['bishop']:<45s}  {row['wordfish_score']:+7.3f}  "
          f"{row['wordfish_se']:5.3f}  {int(row['n_docs']):4d}  "
          f"{row['territory']}{flag}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8/8] Generating plots...")

palette_terr = dict(zip(
    scores_out["territory"].unique(),
    sns.color_palette("tab20", scores_out["territory"].nunique())
))

# — PLOT 1: Wordfish ideology scale ————————————————————————————————————————————
n_actors  = len(scores_out)
fig, ax   = plt.subplots(figsize=(15, max(10, n_actors * 0.38)))

y_pos  = np.arange(n_actors)
scores = scores_out["wordfish_score"].values
ses    = scores_out["wordfish_se"].values
names  = scores_out["bishop"].values
terrs  = scores_out["territory"].values
ndocs  = scores_out["n_docs"].values
is_cem = scores_out["is_cem"].values

colors = []
for t, cem_flag in zip(terrs, is_cem):
    if cem_flag:
        colors.append("#444444")
    else:
        colors.append(palette_terr.get(t, "grey"))

ax.barh(y_pos, scores, xerr=ses, color=colors, alpha=0.75,
        edgecolor="white", linewidth=0.5,
        error_kw=dict(elinewidth=1.2, capsize=3, ecolor="black", alpha=0.6))
ax.axvline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.5)

for i, (name, sc, n, cem_flag) in enumerate(zip(names, scores, ndocs, is_cem)):
    label = f"{name} (n={n})" + (" ●" if cem_flag else "")
    offset = 0.04 if sc >= 0 else -0.04
    ha     = "left" if sc >= 0 else "right"
    ax.text(sc + offset, i, label, va="center", ha=ha,
            fontsize=8, color="black", fontweight="bold" if cem_flag else "normal")

ax.set_yticks([])
ax.set_xlabel("Wordfish Score  ← más progresista  |  más conservador →",
              fontsize=11, fontweight="bold")
ax.set_title("Posiciones Ideológicas — Obispos Mexicanos (corpus completo, 2,460 docs)\n"
             "Método: Wordfish (Poisson, Lauderdale & Herzog 2016)", fontsize=12,
             fontweight="bold", pad=10)
legend_patches = [
    mpatches.Patch(color="#444444", label="CEM (institución)", alpha=0.85)
] + [
    mpatches.Patch(color=c, label=t, alpha=0.8)
    for t, c in list(palette_terr.items())[:18]
]
ax.legend(handles=legend_patches, title="Territorio", fontsize=7.5,
          loc="lower right", framealpha=0.9, ncol=2)
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_wordfish_ideology_scale.png", dpi=200, bbox_inches="tight")
plt.close()
print("  [saved] 01_wordfish_ideology_scale.png")

# — PLOT 2: Word discrimination (efish plot: α vs β) ——————————————————————————
n_label   = 35
top_words = words_df.head(n_label)

fig, ax = plt.subplots(figsize=(12, 7))
sc = ax.scatter(top_words["beta"], top_words["alpha"],
                c=top_words["beta"], cmap="RdBu_r",
                s=60, alpha=0.8, edgecolors="grey", linewidths=0.4)
plt.colorbar(sc, ax=ax, label="β (discrimination weight)")
for _, row in top_words.iterrows():
    ax.annotate(row["word"], xy=(row["beta"], row["alpha"]),
                xytext=(3, 3), textcoords="offset points", fontsize=8, alpha=0.85)
ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.4)
ax.set_xlabel("β  (− = progresista  |  + = conservador)", fontsize=11, fontweight="bold")
ax.set_ylabel("α  (log frecuencia de base)", fontsize=11, fontweight="bold")
ax.set_title(f"Parámetros Wordfish — Top {n_label} palabras discriminantes",
             fontsize=12, fontweight="bold", pad=10)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUT_DIR / "02_wordfish_words.png", dpi=200, bbox_inches="tight")
plt.close()
print("  [saved] 02_wordfish_words.png")

# — PLOT 3: Topic-space PCA biplot —————————————————————————————————————————————
fig, ax = plt.subplots(figsize=(13, 9))
for _, row in bishop_enough.iterrows():
    col = "#444444" if row["is_cem"] else palette_terr.get(row["territory"], "grey")
    ax.scatter(row["topic_pca_dim1"], row["topic_pca_dim2"],
               color=col, s=max(20, row["n_docs"] * 3),
               alpha=0.75, edgecolors="white", linewidths=0.4)
    label = "CEM" if row["is_cem"] else row["bishop"].split()[-1]
    ax.annotate(label, xy=(row["topic_pca_dim1"], row["topic_pca_dim2"]),
                xytext=(4, 2), textcoords="offset points", fontsize=7.5, alpha=0.9)

scale_f = 3.0
for topic_name, (d1, d2) in zip(loadings_df.index, loadings_df.values):
    ax.annotate("", xy=(d1*scale_f, d2*scale_f), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="navy", lw=1.5, alpha=0.6))
    ax.text(d1*scale_f*1.1, d2*scale_f*1.1, topic_name,
            fontsize=7, color="navy", alpha=0.75, ha="center")

ax.axhline(0, color="grey", lw=0.8, alpha=0.5)
ax.axvline(0, color="grey", lw=0.8, alpha=0.5)
ax.set_xlabel(f"PCA Dim 1 ({pca.explained_variance_ratio_[0]:.1%} var.)",
              fontsize=11, fontweight="bold")
ax.set_ylabel(f"PCA Dim 2 ({pca.explained_variance_ratio_[1]:.1%} var.)",
              fontsize=11, fontweight="bold")
ax.set_title("Espacio de tópicos — Posición de obispos (PCA, 9 tópicos)",
             fontsize=12, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "03_topic_pca_biplot.png", dpi=200, bbox_inches="tight")
plt.close()
print("  [saved] 03_topic_pca_biplot.png")

# — PLOT 4: Wordfish vs. Topic-PCA Dim 1 (validation) —————————————————————————
corr_val = np.corrcoef(bishop_enough["wordfish_score"], bishop_enough["topic_pca_dim1"])[0, 1]
fig, ax = plt.subplots(figsize=(11, 8))
for _, row in bishop_enough.iterrows():
    col = "#444444" if row["is_cem"] else palette_terr.get(row["territory"], "grey")
    ax.scatter(row["topic_pca_dim1"], row["wordfish_score"],
               color=col, s=max(25, row["n_docs"] * 3),
               alpha=0.75, edgecolors="white", linewidths=0.4)
    label = "CEM" if row["is_cem"] else row["bishop"].split()[-1]
    ax.annotate(label, xy=(row["topic_pca_dim1"], row["wordfish_score"]),
                xytext=(4, 2), textcoords="offset points", fontsize=8, alpha=0.9)
x = bishop_enough["topic_pca_dim1"].values
y = bishop_enough["wordfish_score"].values
m, b_reg = np.polyfit(x, y, 1)
xline = np.linspace(x.min(), x.max(), 100)
ax.plot(xline, m*xline + b_reg, "k--", lw=1.5, alpha=0.5)
ax.axhline(0, color="grey", lw=0.8, alpha=0.4)
ax.axvline(0, color="grey", lw=0.8, alpha=0.4)
ax.set_xlabel("Topic-PCA Dim 1", fontsize=11, fontweight="bold")
ax.set_ylabel("Wordfish Score", fontsize=11, fontweight="bold")
ax.set_title(f"Validación: Wordfish vs. PCA de tópicos  (r = {corr_val:+.3f})",
             fontsize=12, fontweight="bold", pad=10)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_DIR / "04_wordfish_vs_pca.png", dpi=200, bbox_inches="tight")
plt.close()
print("  [saved] 04_wordfish_vs_pca.png")

# — PLOT 5: Most discriminating words (left vs. right) ————————————————————————
n_side = 15
left_w  = words_df[words_df["beta"] < 0].head(n_side)
right_w = words_df[words_df["beta"] > 0].head(n_side)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax_s, wdf, title, color in [
    (axes[0], left_w,  "Palabras más progresistas (β < 0)", "#2166ac"),
    (axes[1], right_w, "Palabras más conservadoras (β > 0)", "#d6604d"),
]:
    ax_s.barh(wdf["word"][::-1], np.abs(wdf["beta"][::-1]),
              color=color, alpha=0.75, edgecolor="white")
    ax_s.set_xlabel("|β|  (peso discriminante)", fontsize=10)
    ax_s.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax_s.grid(axis="x", alpha=0.3)
plt.suptitle("Vocabulario más discriminante — corpus completo",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "05_discriminating_words.png", dpi=200, bbox_inches="tight")
plt.close()
print("  [saved] 05_discriminating_words.png")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("DONE.  Outputs in:")
print(f"  {OUT_DIR}")
print("=" * 72)
for f in sorted(OUT_DIR.glob("*")):
    print(f"  {f.name:<55s}  {f.stat().st_size/1024:6.1f} KB")
