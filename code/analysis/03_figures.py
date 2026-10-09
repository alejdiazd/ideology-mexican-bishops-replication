# -*- coding: utf-8 -*-
"""
03_figures.py

Draws every figure of the article and its supplement from the files in
../../data (Figure S7 is drawn by 02_regression_tables.R):

  Figure 1   topic prevalence over time (5-year bins)
  Figure 2   primary topic by author, top 20 authors
  Figure 3   Wordfish ideal points, 39 actors, 95 percent error bands
  Figure 4   Wordfish scores vs. topic-space PCA, 39 actors
  Figure S1  most discriminating words along the Wordfish scale
  Figure S2  centrality of the principal consecrators
  Figure S3  the episcopal consecration network
  Figure S4  documents per primary topic
  Figure S5  LDA coherence (c_v) by number of topics
  Figure S6  Wordfish: full corpus vs. Church only vs. press only
  Figure S8  mean document ideology by decade

Run from this folder:  python 03_figures.py
Output: ../../output/figures/*.png (600 dpi)
Requires: Python 3.9+, pandas, numpy, matplotlib, seaborn, networkx, adjustText
"""
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from adjustText import adjust_text
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..", "..", "output", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"savefig.dpi": 600, "figure.dpi": 110, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold",
                     "font.family": "DejaVu Sans"})
RED, BLUE = "#c44e52", "#4c72b0"
TL = {1: "Social pastoral / development", 2: "Hierarchy & papal relations",
      3: "Episcopal peace calls", 4: "Politics & Church–State",
      5: "Diocesan pastoral life", 6: "Appointments & formation",
      7: "Church in public sphere (news)", 8: "Theology & homiletics",
      9: "Chiapas & indigenous rights"}
TPROB = [f"topic_{i}_prob" for i in range(1, 10)]


def rd(name):
    return pd.read_csv(os.path.join(DATA, name), encoding="utf-8-sig", low_memory=False)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


dt = rd("02_topic_model_document_topics.csv")
wf = rd("03_wordfish_ideal_points_N39.csv")
wf["oriented"] = wf["theta_oriented"]  # negative = progressive, positive = institutional

# ---------------------------------------------------------------- Figure 1
d = dt[dt.year.notna()].copy()
d["bin"] = (d.year // 5 * 5).astype(int)
grp = d.groupby("bin")[TPROB].mean()
grp = grp.div(grp.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(8, 4.2))
cmap = plt.get_cmap("tab10")
ax.stackplot(grp.index, [grp[c].values for c in TPROB],
             labels=[TL[i] for i in range(1, 10)], colors=[cmap(i) for i in range(9)])
ax.set_xlim(grp.index.min(), grp.index.max())
ax.set_ylim(0, 1); ax.set_xlabel("Year (5-year bins)"); ax.set_ylabel("Share of topic emphasis")
ax.set_title("Topic prevalence over time, 1976–2026")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
save(fig, "Figure_1.png")

# ---------------------------------------------------------------- Figure 2
top = dt["name_normalized"].value_counts().head(20).index
sub = dt[dt["name_normalized"].isin(top)].copy()
sub["name_normalized"] = sub["name_normalized"].replace(
    {"Provincia Eclesiastica de Tlalnepantla": "Provincia Eclesiástica de Tlalnepantla"})
ct = pd.crosstab(sub["name_normalized"], sub["primary_topic"])
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
            cbar_kws={"label": "Documents"}, ax=ax)
ax.set_title("Topic Profile — Top 20 Authors / Institutions", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Topic", fontsize=11, fontweight="bold")
ax.set_ylabel("Author", fontsize=11, fontweight="bold")
plt.tight_layout()
save(fig, "Figure_2.png")

# ---------------------------------------------------------------- Figure 3
d = wf.sort_values("oriented").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7, 9))
ax.errorbar(d.oriented, range(len(d)), xerr=1.96 * d.wordfish_se, fmt="o", ms=4,
            color="#222", ecolor="#999", elinewidth=.8, capsize=2, zorder=3)
ax.scatter(d.oriented, range(len(d)), c=[RED if v < 0 else BLUE for v in d.oriented], s=28, zorder=4)
ax.axvline(0, color="gray", ls="--", lw=.7)
ax.set_yticks(range(len(d))); ax.set_yticklabels(d.bishop, fontsize=7)
ax.set_xlabel("Wordfish ideal point\n(negative = progressive / liberation;  "
              "positive = institutional / conservative)")
ax.set_title("Wordfish ideal points of Mexican bishops and the CEM")
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", color=BLUE, label="Positive score (institutional / conservative)"),
    Line2D([], [], marker="o", ls="", color=RED, label="Negative score (progressive / liberation)"),
    Line2D([], [], marker="|", ls="-", color="#999", label="95 percent error band")],
    loc="lower right", fontsize=7, frameon=True)
save(fig, "Figure_3.png")

# ---------------------------------------------------------------- Figure 4
# Labels are placed by hand (data coordinates) so that no name covers a point or
# another name; a thin leader line joins each label to its point. Labels give the
# first name and first surname; Méndez Arceo keeps both surnames.
POS = {
    "CEM": (-1.05, 4.55, "left"),
    "Ramón Castro Castro": (-0.25, 1.77, "left"),
    "Rogelio Cabrera López": (2.41, 1.56, "center"),
    "Fernando Mario Chávez Ruvalcaba": (0.20, 1.13, "left"),
    "Mario De Gasperín Gasperín": (-0.62, 1.04, "right"),
    "Carlos Aguiar Retes": (-0.78, 0.72, "right"),
    "Enrique Díaz Díaz": (0.64, 0.66, "left"),
    "Rodrigo Aguilar Martínez": (0.12, 0.80, "center"),
    "Juan Manuel Mancilla Sánchez": (-1.95, 0.36, "center"),
    "Felipe Arizmendi Esquivel": (1.80, 0.40, "left"),
    "Constancio Miranda Weckmann": (-2.15, 0.10, "center"),
    "Héctor González Martínez": (1.76, -0.04, "left"),
    "Francisco Robles Ortega": (1.85, -0.21, "left"),
    "Raúl Gómez González": (-2.74, -0.37, "center"),
    "Christophe Pierre": (-2.10, -0.53, "center"),
    "Arturo Lona Reyes": (3.24, -0.37, "left"),
    "Sergio Méndez Arceo": (1.52, -0.79, "left"),
    "Raúl Vera López": (2.67, -0.62, "left"),
    "José Guadalupe Torres Campos": (-1.62, -0.71, "right"),
    "Jesús José Herrera Quiñonez": (-1.62, -0.85, "right"),
    "Víctor Sánchez Espinosa": (-0.45, -0.64, "left"),
    "Gustavo Rodríguez Vega": (-0.45, -0.77, "left"),
    "Rafael Romo Muñoz": (-0.45, -0.90, "left"),
    "Norberto Rivera Carrera": (-0.45, -1.03, "left"),
    "Felipe Aguirre Franco": (2.19, -0.98, "left"),
    "Samuel Ruiz García": (4.33, -1.32, "center"),
}


def short(name):
    parts = name.split()
    if name == "Sergio Méndez Arceo" or len(parts) < 3:
        return name
    return " ".join(parts[:-1])


fig, ax = plt.subplots(figsize=(8.5, 7))
ax.scatter(wf.topic_pca_dim1, wf.oriented, s=30, color=BLUE, zorder=3)
for _, r in wf.iterrows():
    if r.bishop in POS:
        xt, yt, ha = POS[r.bishop]
        ax.annotate(short(r.bishop), (r.topic_pca_dim1, r.oriented), xytext=(xt, yt),
                    ha=ha, va="center", fontsize=7, zorder=4,
                    arrowprops=dict(arrowstyle="-", color="#888", lw=0.5, shrinkA=0, shrinkB=3))
ax.axhline(0, color="gray", lw=.6, ls="--"); ax.axvline(0, color="gray", lw=.6, ls="--")
ax.set_xlabel("Topic-space PCA Dimension 1 (political engagement → )")
ax.set_ylabel("Wordfish ideal point (oriented)")
ax.set_title("Wordfish scores vs. topic-space PCA (39 actors)")
ax.set_xlim(-3.6, 5.3); ax.set_ylim(-1.75, 4.85)
save(fig, "Figure_4.png")

# ---------------------------------------------------------------- Figure S1
# Word weights are oriented like the actor scale (beta_oriented = -beta), so the
# lowest weights mark the progressive end and the highest the institutional end.
wp = rd("03b_wordfish_word_parameters_N39.csv")
wp["beta_oriented"] = -wp["beta"]
lo, hi = wp.nsmallest(15, "beta_oriented"), wp.nlargest(15, "beta_oriented")
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
axes[0].barh(lo.word[::-1], lo.beta_oriented[::-1], color=RED)
axes[0].set_title("Progressive / liberation end (lowest β)")
axes[1].barh(hi.word[::-1], hi.beta_oriented[::-1], color=BLUE)
axes[1].set_title("Institutional / conservative end (highest β)")
for a in axes:
    a.tick_params(labelsize=8)
    a.set_xlabel("Discrimination weight β (oriented as in Figure 3)")
fig.suptitle("Most discriminating words along the Wordfish scale", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "Figure_S1.png")

# ---------------------------------------------------------------- Figures S2, S3
nodes = rd("04_consecration_network_nodes.csv")
edges = rd("04b_consecration_edges_clean.csv")


def strip_title(s):
    s = re.sub(r"\b(Archbishop|Bishop|Cardinal)\b", "", str(s)).replace("†", "")
    return re.sub(r"\s+", " ", s).strip()


top = nodes.sort_values("n_principal_consecrations", ascending=False).head(15)
cols = ["n_principal_consecrations", "betweenness", "pagerank", "hits_hub"]
M = top[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
Mn = (M - M.min(0)) / (np.ptp(M, axis=0) + 1e-9)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(Mn, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(["Bishops\nconsecrated\n(principal)", "Betweenness", "PageRank", "HITS hub"], fontsize=9)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.node.map(strip_title), fontsize=8)
ax.set_title("Centrality of principal consecrators (min–max normalized)")
fig.colorbar(im, ax=ax, shrink=.7, label="Normalized centrality")
save(fig, "Figure_S2.png")

name = dict(zip(nodes.node_key, nodes.node.map(strip_title)))
SHORT = {"girolamo prigione": "Prigione", "norberto rivera carrera": "Rivera Carrera",
         "juan sandoval iniguez": "Sandoval", "jose francisco robles ortega": "Robles Ortega",
         "carlos aguiar retes": "Aguiar", "rogelio cabrera lopez": "Cabrera",
         "alberto suarez inda": "Suárez Inda", "adolfo antonio suarez rivera": "Suárez Rivera",
         "ernesto corripio y ahumada": "Corripio"}
G = nx.DiGraph()
for _, r in edges[edges.role == "principal_consecrator"].iterrows():
    G.add_edge(r.from_key, r.to_key)
deg = dict(zip(nodes.node_key, nodes.n_principal_consecrations.fillna(0)))
clus = dict(zip(nodes.node_key, nodes.cluster.fillna("Other").astype(str)))
pos = nx.spring_layout(G, k=0.45, seed=7, iterations=60)
clusters = sorted(set(clus.get(n, "Other") for n in G.nodes()))
ccol = {c: plt.get_cmap("tab20")(i % 20) for i, c in enumerate(clusters)}
fig, ax = plt.subplots(figsize=(11, 9))
nx.draw_networkx_edges(G, pos, ax=ax, alpha=.2, arrows=False, width=.5)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[30 + 40 * np.sqrt(deg.get(n, 0)) for n in G.nodes()],
                       node_color=[ccol[clus.get(n, "Other")] for n in G.nodes()],
                       linewidths=.3, edgecolors="white")
texts = [ax.text(pos[n][0], pos[n][1], SHORT.get(n, name.get(n, n)), fontsize=8, fontweight="bold",
                 ha="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
         for n in G.nodes() if deg.get(n, 0) >= 7]
adjust_text(texts, ax=ax, expand=(1.2, 1.5), force_text=(0.5, 0.8),
            arrowprops=dict(arrowstyle="-", color="#444", lw=0.6))
ax.set_title("Episcopal consecration network (node size = bishops consecrated; color = lineage cluster)")
ax.axis("off")
ax.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc=ccol[c], mec="none", label=c) for c in clusters],
          fontsize=8, loc="lower left", frameon=False, title="Cluster")
save(fig, "Figure_S3.png")

# ---------------------------------------------------------------- Figure S4
counts = dt.primary_topic.value_counts().reindex(range(1, 10)).fillna(0)
fig, ax = plt.subplots(figsize=(7, 4))
y = range(1, 10)
ax.barh(list(y), counts.values, color=BLUE)
ax.set_yticks(list(y)); ax.set_yticklabels([f"{i}. {TL[i]}" for i in y]); ax.invert_yaxis()
for i, v in zip(y, counts.values):
    ax.text(v + 4, i, f"{int(v)} ({v / len(dt) * 100:.1f}%)", va="center", fontsize=8)
ax.set_xlim(0, counts.max() * 1.18)
ax.set_xlabel("Number of documents (primary topic)")
ax.set_title("Distribution of documents across the nine topics")
save(fig, "Figure_S4.png")

# ---------------------------------------------------------------- Figure S5
c = rd("02b_lda_coherence_by_k.csv")
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(c.k, c.coherence_cv, "-o", color="#333", ms=4)
kbest = int(c.loc[c.coherence_cv.idxmax(), "k"])
ax.axvline(kbest, color="crimson", ls="--", lw=1)
ax.annotate(f"selected K = {kbest}", xy=(kbest, c.coherence_cv.max()),
            xytext=(kbest + 1.2, c.coherence_cv.max()), color="crimson")
ax.set_xlabel("Number of topics (K)"); ax.set_ylabel("Coherence (c_v)")
ax.set_title("Topic-model coherence by number of topics")
save(fig, "Figure_S5.png")

# ---------------------------------------------------------------- Figure S6
BENCH = ["CEM", "Ramón Castro Castro", "Rogelio Cabrera López", "Carlos Aguiar Retes",
         "Arturo Lona Reyes", "Juan Sandoval Íñiguez", "Sergio Méndez Arceo", "Raúl Vera López",
         "Rafael Romo Muñoz", "Norberto Rivera Carrera", "Samuel Ruiz García"]
cp = rd("03c_wordfish_church_vs_press_N39.csv").set_index("bishop").loc[BENCH[::-1]]
fig, ax = plt.subplots(figsize=(10, 6.8))
yy = np.arange(len(cp))
for i, (_, r) in enumerate(cp.iterrows()):
    vals = r[["wordfish_full", "wordfish_church", "wordfish_press"]].dropna()
    ax.plot([vals.min(), vals.max()], [i, i], color="#bbb", lw=1, zorder=1)
ax.scatter(cp.wordfish_full, yy, marker="D", s=70, color="black", label="Full corpus", zorder=3)
ax.scatter(cp.wordfish_church, yy, marker="o", s=70, color="#2166ac", label="Church-only", zorder=3)
ax.scatter(cp.wordfish_press, yy, marker="s", s=70, color="#b2182b", label="Press-only", zorder=3)
ax.axvline(0, color="gray", ls="--", lw=.7)
ax.set_yticks(yy); ax.set_yticklabels(cp.index)
ax.grid(axis="x", alpha=.3)
ax.set_xlabel("Wordfish score  (negative = progressive/liberationist  |  positive = institutional/conservative)")
ax.set_title("Wordfish ideal points: full corpus vs. Church-only vs. press-only re-estimation",
             fontweight="normal")
ax.legend(loc="lower right")
save(fig, "Figure_S6.png")

# ---------------------------------------------------------------- Figure S8
dt["decade"] = dt.year // 10 * 10
dec = dt.dropna(subset=["decade"]).groupby("decade").ideology_score.mean()
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar([f"{int(x)}s" for x in dec.index], dec.values, color=[BLUE if v < 0 else RED for v in dec.values])
ax.axhline(0, color="gray", lw=.6)
ax.set_ylabel("Mean document ideology (social-engagement +)")
ax.set_title("Mean episcopal ideology by decade, 1976–2026")
save(fig, "Figure_S8.png")
