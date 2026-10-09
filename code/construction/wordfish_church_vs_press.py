# -*- coding: utf-8 -*-
"""
Church-only vs. press-only Wordfish robustness check.
Re-estimates the Wordfish scale (same algorithm/settings as
run_wordfish_full_corpus.py) restricted first to Church-generated documents
(CEM, diocesan bulletins, Voices, El Mensajero) and then to press documents
(news, La Jornada, Proceso, El Universal, Reforma), to test whether benchmark
actors (Ruiz, Vera, Lona, Aguiar, Sandoval, Rivera, CEM, Cabrera, Castro)
retain comparable relative positions once the mixed-source corpus is split.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

try:
    from spacy.lang.es.stop_words import STOP_WORDS as SPACY_ES
except ImportError:
    SPACY_ES = set()

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# PROJECT_ROOT is the folder that holds corpus_final/, the full-text corpus.
# The full text is not distributed (press copyright); see README.
BASE   = Path(".")  # PROJECT_ROOT
CORPUS = BASE / "corpus_final" / "master_corpus_final.csv"
TOPICS = BASE / "topic_model_results_full" / "document_topics_full.csv"
OUT_DIR = BASE / "replication" / "wordfish_source_split"
OUT_DIR.mkdir(exist_ok=True, parents=True)

MIN_DOCS  = 8       # 10 in the full corpus; 8 retains more actors in the smaller subsets
MAX_VOCAB = 3000
MAX_ITER  = 200

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

HOMILETIC_TOPIC = 8
NOISE_NAMES = {
    "iglesia católica", "iglesia cat�lica", "desconocido", "desconocida",
    "episcopado", "vaticano", "juan pablo ii", "iglesia", "cat�lica",
}
CHURCH_SOURCES = {"CEM","diocesis","voices","mensajero"}
PRESS_SOURCES  = {"news","jornada","proceso","universal","reforma"}

# ══ 1. LOAD & MERGE (identical to run_wordfish_full_corpus.py) ══
master = pd.read_csv(CORPUS, encoding="latin-1")
topics = pd.read_csv(TOPICS, encoding="utf-8-sig")
df_full = topics.merge(master[["doc_id","text"]], on="doc_id", how="left")
df_full["text"] = df_full["text"].fillna("").astype(str)

df_full["name_normalized"] = (
    df_full["name_normalized"].fillna("Unknown").astype(str).str.strip().str.rstrip(",").str.strip()
)
reassign = {"Unknown","unknown","obispo","OBISPO"}
df_full.loc[df_full["name_normalized"].isin(reassign), "name_normalized"] = "CEM"
df_full = df_full[~df_full["name_normalized"].str.lower().isin({n.lower() for n in NOISE_NAMES})].reset_index(drop=True)

is_cem_source = df_full["source"].str.upper() == "CEM"
is_homiletic  = df_full["primary_topic"] == HOMILETIC_TOPIC
df_full = df_full[~(is_cem_source & is_homiletic)].reset_index(drop=True)

prob_cols = [f"topic_{t}_prob" for t in range(1, 10)]
for c in prob_cols:
    df_full[c] = df_full[c].fillna(0)

print(f"Full preprocessed corpus: {len(df_full)} docs, {df_full['name_normalized'].nunique()} actors")
print(df_full["source"].value_counts())

# ══ wordfish core (identical algorithm to run_wordfish_full_corpus.py) ══
def _ca_init(dtm):
    grand = dtm.sum()
    P = dtm / grand
    r = P.sum(axis=1, keepdims=True)
    c = P.sum(axis=0, keepdims=True)
    S = (P - r @ c) / np.sqrt(r @ c + 1e-12)
    svd = TruncatedSVD(n_components=1, random_state=42)
    theta = svd.fit_transform(S).flatten()
    return (theta - theta.mean()) / (theta.std() + 1e-12)

def wordfish(dtm, max_iter=MAX_ITER, lr=0.005, tol=1e-4):
    n_d, n_w = dtm.shape
    theta = _ca_init(dtm)
    alpha = np.log(dtm.mean(axis=0).clip(min=0.01))
    beta  = np.zeros(n_w)
    ll_hist = []
    for it in range(max_iter):
        for w in range(n_w):
            counts = dtm[:, w]
            if counts.sum() < 1:
                continue
            mu = np.exp(alpha[w] + beta[w] * theta).clip(1e-10)
            X = np.column_stack([np.ones(n_d), theta])
            lhs = X.T @ (mu[:, None] * X) + np.eye(2) * 1e-6
            z = np.log(mu) + (counts - mu) / mu
            rhs = X.T @ (mu * z)
            try:
                params = np.linalg.solve(lhs, rhs)
                alpha[w], beta[w] = params[0], params[1]
            except np.linalg.LinAlgError:
                pass
        for d in range(n_d):
            counts = dtm[d, :]
            mu = np.exp(alpha + beta * theta[d]).clip(1e-10)
            grad = np.sum(beta * (counts - mu))
            hess = -np.sum(beta**2 * mu)
            if abs(hess) > 1e-10:
                theta[d] -= lr * grad / hess
        theta = (theta - theta.mean()) / (theta.std() + 1e-12)
        mu_all = np.exp(alpha[None, :] + beta[None, :] * theta[:, None]).clip(1e-10)
        ll = (dtm * np.log(mu_all) - mu_all - dtm * np.log(dtm.clip(1e-10))).sum()
        ll_hist.append(ll)
        if it > 5 and abs(ll_hist[-1] - ll_hist[-2]) < tol:
            break
    return theta, alpha, beta, ll_hist


def run_wordfish_on_subset(df, label, min_docs=MIN_DOCS):
    print(f"\n{'='*70}\nSUBSET: {label}  ({len(df)} docs)\n{'='*70}")
    grp = df.groupby("name_normalized")

    def mode_val(s):
        m = s.mode()
        return m.iloc[0] if len(m) > 0 else s.iloc[0]

    bishop_df = pd.DataFrame({
        "bishop": list(grp.groups.keys()),
        "n_docs": grp["text"].count().values,
        "text_concat": grp["text"].apply(lambda s: " ".join(s.dropna().astype(str))).values,
        "territory": grp["territory"].apply(mode_val).values,
    })
    for c in prob_cols:
        bishop_df[c] = grp[c].mean().values
    bishop_df["is_cem"] = bishop_df["bishop"] == "CEM"

    has_named_territory = (
        bishop_df["territory"].notna()
        & (bishop_df["territory"].astype(str).str.strip().str.lower() != "nan")
        & (bishop_df["territory"].astype(str).str.strip() != "")
    )
    bishop_df = bishop_df[has_named_territory | bishop_df["is_cem"]].copy().reset_index(drop=True)
    bishop_enough = bishop_df[bishop_df["n_docs"] >= min_docs].copy().reset_index(drop=True)
    print(f"  Actors with >= {min_docs} docs: {len(bishop_enough)} of {len(bishop_df)}")
    print(f"  Docs covered: {bishop_enough['n_docs'].sum()} ({100*bishop_enough['n_docs'].sum()/len(df):.1f}%)")

    vectorizer = CountVectorizer(
        max_features=MAX_VOCAB, min_df=max(3, len(bishop_enough)//8), max_df=0.90,
        stop_words=list(STOPS), token_pattern=r"(?u)\b[a-záéíóúüñ]{5,}\b", lowercase=True,
    )
    DTM = vectorizer.fit_transform(bishop_enough["text_concat"]).toarray().astype(float)
    print(f"  DTM shape: {DTM.shape}")

    theta, alpha_w, beta_w, ll_hist = wordfish(DTM)
    bishop_enough["wordfish_score"] = theta

    corr_orient = np.corrcoef(theta, bishop_enough["topic_9_prob"].values)[0, 1]
    if corr_orient < 0:
        bishop_enough["wordfish_score"] = -theta
    print(f"  Orientation corr with Topic9: {corr_orient:+.3f}  (final LL={ll_hist[-1]:.1f}, {len(ll_hist)} iters)")

    out = bishop_enough[["bishop","territory","n_docs","is_cem","wordfish_score"]].sort_values("wordfish_score")
    out.to_csv(OUT_DIR / f"wordfish_{label}.csv", index=False, encoding="utf-8-sig")
    return out


df_church = df_full[df_full["source"].isin(CHURCH_SOURCES)].reset_index(drop=True)
df_press  = df_full[df_full["source"].isin(PRESS_SOURCES)].reset_index(drop=True)

res_church = run_wordfish_on_subset(df_church, "church_only")
res_press  = run_wordfish_on_subset(df_press,  "press_only")

# ══ full-corpus reference (reuse existing saved result for comparability) ══
full = pd.read_csv(BASE / "wordfish_results_full" / "wordfish_scores_full.csv", encoding="utf-8-sig")
full = full[["bishop","n_docs","wordfish_score"]].rename(
    columns={"n_docs":"n_docs_full","wordfish_score":"wordfish_full"})

print("\n\n" + "="*70)
print("COMPARISON TABLE (full-corpus vs Church-only vs Press-only)")
print("="*70)
m = full.merge(res_church[["bishop","n_docs","wordfish_score"]].rename(
        columns={"n_docs":"n_docs_church","wordfish_score":"wordfish_church"}), on="bishop", how="left")
m = m.merge(res_press[["bishop","n_docs","wordfish_score"]].rename(
        columns={"n_docs":"n_docs_press","wordfish_score":"wordfish_press"}), on="bishop", how="left")
m = m.sort_values("wordfish_full")
pd.set_option("display.width", 160)
print(m.to_string(index=False))
m.to_csv(OUT_DIR / "comparison_full_church_press.csv", index=False, encoding="utf-8-sig")

from scipy.stats import spearmanr, pearsonr
both_church = m.dropna(subset=["wordfish_church"])
both_press  = m.dropna(subset=["wordfish_press"])
print(f"\nN actors retained in Church-only: {len(both_church)} of {len(m)}")
print(f"N actors retained in Press-only:  {len(both_press)} of {len(m)}")
if len(both_church) > 2:
    r_s, p_s = spearmanr(both_church["wordfish_full"], both_church["wordfish_church"])
    r_p, p_p = pearsonr(both_church["wordfish_full"], both_church["wordfish_church"])
    print(f"Full vs Church-only: Spearman rho={r_s:.3f} (p={p_s:.4f})  Pearson r={r_p:.3f} (p={p_p:.4f})")
if len(both_press) > 2:
    r_s, p_s = spearmanr(both_press["wordfish_full"], both_press["wordfish_press"])
    r_p, p_p = pearsonr(both_press["wordfish_full"], both_press["wordfish_press"])
    print(f"Full vs Press-only:  Spearman rho={r_s:.3f} (p={p_s:.4f})  Pearson r={r_p:.3f} (p={p_p:.4f})")

print("\nSaved outputs to", OUT_DIR)
