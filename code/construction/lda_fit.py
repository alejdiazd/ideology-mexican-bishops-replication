# -*- coding: utf-8 -*-
"""
Topic Model – Full Mexico Bishops Corpus
==========================================
Pipeline:
  1. Load master_corpus_final_clean.csv (columns: doc_id, source, bishop, territory,
     date, year, title, text, token_count, ...)
  2. Normalize bishop names (consolidate variants of the same person)
  3. Preprocess text with comprehensive Spanish stopwords (spaCy + custom)
  4. Find optimal number of topics via coherence score (c_v)
  5. Fit final LDA model (gensim)
  6. Output:
       - document_topics_full.csv        : one row per document with topic assignments
       - topics_terms_full.csv           : top words per topic
       - topics_summary_full.csv         : topic label + top words + doc count
       - coherence_scores_full.csv       : coherence by number of topics
"""

import sys, re, warnings, os
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from gensim import corpora, models
from gensim.models import CoherenceModel

from spacy.lang.es.stop_words import STOP_WORDS as SPACY_ES_STOPS

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# ── paths ──────────────────────────────────────────────────────────────────────
# PROJECT_ROOT is the folder that holds corpus_final/, the full-text corpus.
# The full text is not distributed (press copyright); see README.
BASE_DIR  = Path(".")  # PROJECT_ROOT
OUT_DIR   = BASE_DIR / "topic_model_results_full"
CSV_IN    = BASE_DIR / "corpus_final" / "master_corpus_final_clean.csv"
OUT_DIR.mkdir(exist_ok=True)

# Windows multiprocessing requires the entry-point guard
print("=" * 72)
print("TOPIC MODEL — FULL MEXICO BISHOPS CORPUS (n≈2,476)")
print("=" * 72)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  STOPWORDS
# ══════════════════════════════════════════════════════════════════════════════

# spaCy built-in Spanish stopwords (521 words, already handles accents)
STOPS = set(SPACY_ES_STOPS)

# Additional high-frequency Spanish words not always in spaCy
EXTRA_COMMON = {
    "así","aún","aquí","allá","allí","ahí","sino","pues","mas","ya","sí","no",
    "ni","si","oh","ah","eh","hay","hoy","haber","ser","estar","tener","hacer",
    "ir","ver","dar","saber","querer","poder","deber","dijo","dice","señor",
    "señora","señores","misma","mismo","dicho","dicha","dichos","dichas",
    "ante","bajo","según","durante","mediante","circa","versus",
    # Numbers written out
    "uno","dos","tres","cuatro","cinco","seis","siete","ocho","nueve","diez",
    "cien","mil","millones","primero","segundo","tercero","primera","segunda",
    # Temporal connectors
    "enero","febrero","marzo","abril","mayo","junio","julio","agosto",
    "septiembre","octubre","noviembre","diciembre","lunes","martes","miércoles",
    "jueves","viernes","sábado","domingo","año","mes","día","días","años","meses",
    # Common prepositions / conjunctions not always in spaCy
    "sobre","entre","hacia","desde","hasta","tras","contra","mediante","respecto",
    # Very frequent verb forms
    "fue","han","son","será","están","tenemos","puede","pueden","debe","deben",
    "tiene","tienen","hace","hacen","hacer","hecho","hablar","decir","dicho",
    # PDF/OCR artifacts and metadata words
    "escrito","escrita","página","pág","http","www","flickr","com","photos",
    "video","foto","fotografía","fotografías","sets","disposición",
    # Ecclesiastical boilerplate that appears in nearly every document
    "mons","monseñor","ilustrísimo","excelentísimo","reverendísimo",
    "secretaría","secretario","secretaria","prensa","comunicación","comunicación",
    # Generic document words
    "informe","carta","mensaje","documento","texto","artículo","nota","boletín",
    "publicado","publicada","publicadas","publicados","versión","edición",
    # Wikipedia citation boilerplate ("Consultado el 18 de junio de 2014")
    "consultado","archivado","original","recuperado","accedido","enlace",
    # Wikipedia structural words
    "referencias","véase","también","categoría","categorías","editar","fuente",
    # English function words (for the handful of English-language docs)
    "the","and","for","from","with","that","this","was","are","were",
    "its","has","had","have","been","not","but","they","their","his",
    "her","our","who","which","when","also","into","more","than","one",
    # Honorific / title words (no thematic content)
    "don","doña","fray","pbro","padre","hermano","hermana",
    # Generic news vocabulary
    "caso","horas","hora","grupo","grupos","general","generales",
    # Numeric / percentage fragments
    "ciento","tanto","cuanto","num","número",
    # Document header abbreviations and OCR garbage (Mensajero)
    "ntlcaial","prodvei","necelpaes","tclaial","encelpeasniatlsatica","prodvein",
    # News site section headers / low-content words
    "noticias","noticia","semana","semanal","edición",
}
STOPS.update(EXTRA_COMMON)

# Lowercase everything so matching works regardless of capitalisation
STOPS = {w.lower() for w in STOPS}

print(f"Stopword list: {len(STOPS)} words")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  NAME NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

# Manual canonical mapping: raw_lowercase_stripped → canonical_name
#   Keys are produced AFTER:
#     - lowercasing
#     - stripping prefixes (mons., monseñor, cardenal, card., don, fr., fray, dr.)
#     - stripping suffixes (o.p., op, l.c., lc, m.j., mj, m.n.m., c.m.g., etc.)
#     - stripping leading/trailing spaces
#
# Value is the canonical full name.

NAME_MAP = {
    # ── Carlos Aguiar Retes ──────────────────────────────────────────────────
    "carlos aguiar retes": "Carlos Aguiar Retes",
    "carlos aguiar": "Carlos Aguiar Retes",
    "aguiar retes": "Carlos Aguiar Retes",
    "carlos aguiar retes  jos": "Carlos Aguiar Retes",
    # ── Norberto Rivera Carrera ──────────────────────────────────────────────
    "norberto rivera carrera": "Norberto Rivera Carrera",
    "norberto rivera": "Norberto Rivera Carrera",
    "norberto card. rivera carrera": "Norberto Rivera Carrera",
    "norberto card  rivera carrera": "Norberto Rivera Carrera",
    # ── Francisco Robles Ortega ──────────────────────────────────────────────
    "francisco robles ortega": "Francisco Robles Ortega",
    "francisco robles": "Francisco Robles Ortega",
    "j. francisco cardenal robles ortega": "Francisco Robles Ortega",
    "francisco cardenal robles ortega": "Francisco Robles Ortega",
    # ── Felipe Arizmendi Esquivel ────────────────────────────────────────────
    "felipe arizmendi esquivel": "Felipe Arizmendi Esquivel",
    "felipe arizmendi": "Felipe Arizmendi Esquivel",
    # ── Juan Sandoval Íñiguez ────────────────────────────────────────────────
    "juan sandoval iñiguez": "Juan Sandoval Íñiguez",
    "juan sandoval ñiguez": "Juan Sandoval Íñiguez",
    "juan sandoval": "Juan Sandoval Íñiguez",
    "cardenal juan sandoval iñiguez": "Juan Sandoval Íñiguez",
    # ── Christophe Pierre ───────────────────────────────────────────────────
    "christophe pierre": "Christophe Pierre",
    # ── Hipólito Reyes Larios ────────────────────────────────────────────────
    "hipólito reyes larios": "Hipólito Reyes Larios",
    # ── Héctor González Martínez ─────────────────────────────────────────────
    "héctor gonzález martínez": "Héctor González Martínez",
    "héctor gonzález": "Héctor González Martínez",
    "héctor gonzález martínez aseveró": "Héctor González Martínez",
    # ── Javier Navarro Rodríguez ─────────────────────────────────────────────
    "javier navarro rodríguez": "Javier Navarro Rodríguez",
    # ── Víctor Sánchez Espinosa ──────────────────────────────────────────────
    "víctor sánchez espinosa": "Víctor Sánchez Espinosa",
    # ── Víctor René Rodríguez Gómez ─────────────────────────────────────────
    "víctor rené rodríguez gómez": "Víctor René Rodríguez Gómez",
    "víctor rené rodrí": "Víctor René Rodríguez Gómez",
    "víctor rené rodríguez": "Víctor René Rodríguez Gómez",
    # ── Mario De Gasperín Gasperín ───────────────────────────────────────────
    "mario de gasperín gasperín": "Mario De Gasperín Gasperín",
    "mario de gasperín": "Mario De Gasperín Gasperín",
    "gasperín gasperín tomó\nposesión": "Mario De Gasperín Gasperín",
    # ── Faustino Armendáriz Jiménez ──────────────────────────────────────────
    "faustino armendáriz jiménez": "Faustino Armendáriz Jiménez",
    "faustiano armendáriz jiménez": "Faustino Armendáriz Jiménez",
    # ── Ramón Castro Castro ──────────────────────────────────────────────────
    "ramón castro castro": "Ramón Castro Castro",
    # ── Rogelio Cabrera López ────────────────────────────────────────────────
    "rogelio cabrera lópez": "Rogelio Cabrera López",
    "rogelio cabrera": "Rogelio Cabrera López",
    "rogelio cabrera arzobispo de tuxtla gutiérrez": "Rogelio Cabrera López",
    # ── Samuel Ruiz García ───────────────────────────────────────────────────
    "samuel ruiz garcía": "Samuel Ruiz García",
    "samuel ruiz": "Samuel Ruiz García",
    "ruiz garcía": "Samuel Ruiz García",
    # ── Sergio Méndez Arceo ──────────────────────────────────────────────────
    "sergio méndez arceo": "Sergio Méndez Arceo",
    "méndez arceo": "Sergio Méndez Arceo",
    # ── Raúl Vera López ──────────────────────────────────────────────────────
    "raúl vera lópez": "Raúl Vera López",
    "raúl vera": "Raúl Vera López",
    "vera lópez": "Raúl Vera López",
    "fr. raúl vera lópez": "Raúl Vera López",
    "fray raúl vera lópez": "Raúl Vera López",
    # ── Felipe Aguirre Franco ────────────────────────────────────────────────
    "felipe aguirre franco": "Felipe Aguirre Franco",
    "felipe aguirre": "Felipe Aguirre Franco",
    # ── Carlos Garfias Merlos ────────────────────────────────────────────────
    "carlos garfias merlos": "Carlos Garfias Merlos",
    "carlos garfias": "Carlos Garfias Merlos",
    # ── Ernesto Corripio Ahumada ─────────────────────────────────────────────
    "ernesto corripio ahumada": "Ernesto Corripio Ahumada",
    # ── José Guadalupe Martín Rábago ─────────────────────────────────────────
    "josé guadalupe martín rábago": "José Guadalupe Martín Rábago",
    "josé g. martín rábago": "José Guadalupe Martín Rábago",
    "josé. g. martín rábago": "José Guadalupe Martín Rábago",
    "josé martín rábago": "José Guadalupe Martín Rábago",
    "josé guadalupe martín": "José Guadalupe Martín Rábago",
    # ── Juan Jesús Posadas Ocampo ────────────────────────────────────────────
    "juan jesús posadas ocampo": "Juan Jesús Posadas Ocampo",
    "cardenal juan jesús posadas": "Juan Jesús Posadas Ocampo",
    "posadas ocampo": "Juan Jesús Posadas Ocampo",
    # ── Florencio Olvera Ochoa ───────────────────────────────────────────────
    "florencio olvera ochoa": "Florencio Olvera Ochoa",
    "florencio olvera": "Florencio Olvera Ochoa",
    "florencio olvera ochoa": "Florencio Olvera Ochoa",
    # ── Adalberto Almeida Merino ─────────────────────────────────────────────
    "adalberto almeida merino": "Adalberto Almeida Merino",
    "adalberto almeida": "Adalberto Almeida Merino",
    # ── Bartolomé Carrasco Briseño ───────────────────────────────────────────
    "bartolomé carrasco briseño": "Bartolomé Carrasco Briseño",
    "bartolomé carrasco": "Bartolomé Carrasco Briseño",
    # ── Constancio Miranda Weckmann ──────────────────────────────────────────
    "constancio miranda weckmann": "Constancio Miranda Weckmann",
    "constancio miranda weckman": "Constancio Miranda Weckmann",
    "constancio miranda weckmann": "Constancio Miranda Weckmann",
    # ── Ricardo Watty Urquidi ────────────────────────────────────────────────
    "ricardo watty urquidi": "Ricardo Watty Urquidi",
    "ricardo watty": "Ricardo Watty Urquidi",
    # ── Emilio Berlie Belaunzarín ────────────────────────────────────────────
    "emilio berlie belaunzarín": "Emilio Berlie Belaunzarín",
    "emilio carlos berlie belaunzarín": "Emilio Berlie Belaunzarín",
    # ── Enrique Díaz Díaz ────────────────────────────────────────────────────
    "enrique díaz díaz": "Enrique Díaz Díaz",
    "enrique díaz diaz": "Enrique Díaz Díaz",
    # ── José Fernández Arteaga ───────────────────────────────────────────────
    "josé fernández arteaga": "José Fernández Arteaga",
    "josé fernández": "José Fernández Arteaga",
    # ── Alberto Suárez Inda ──────────────────────────────────────────────────
    "alberto suárez inda": "Alberto Suárez Inda",
    # ── Leopoldo González González ───────────────────────────────────────────
    "josé leopoldo gonzález gonzález": "Leopoldo González González",
    "leopoldo gonzález gonzález": "Leopoldo González González",
    # ── José Luis Dibildox Martínez ──────────────────────────────────────────
    "josé luis dibildox martínez": "José Luis Dibildox Martínez",
    "josé luis dibildox": "José Luis Dibildox Martínez",
    # ── Rodrigo Aguilar Martínez ─────────────────────────────────────────────
    "rodrigo aguilar martínez": "Rodrigo Aguilar Martínez",
    # ── Sergio Obeso Rivera ──────────────────────────────────────────────────
    "sergio obeso rivera": "Sergio Obeso Rivera",
    # ── Luis Reynoso Cervantes ───────────────────────────────────────────────
    "luis reynoso cervantes": "Luis Reynoso Cervantes",
    "luis reynoso": "Luis Reynoso Cervantes",
    # ── Rafael Romo Muñoz ────────────────────────────────────────────────────
    "rafael romo muñoz": "Rafael Romo Muñoz",
    "rafael romo": "Rafael Romo Muñoz",
    "romo muñoz": "Rafael Romo Muñoz",
    # ── Manuel Pérez Gil ─────────────────────────────────────────────────────
    "manuel pérez gil díaz": "Manuel Pérez Gil Díaz",
    "manuel pérez gil": "Manuel Pérez Gil Díaz",
    "manuel pérez": "Manuel Pérez Gil Díaz",
    # ── Renato Ascencio León ─────────────────────────────────────────────────
    "renato ascencio león": "Renato Ascencio León",
    # ── José Leopoldo González González ─────────────────────────────────────
    "josé leopoldo gonzález": "Leopoldo González González",
    # ── institutions ─────────────────────────────────────────────────────────
    "cem": "CEM",
    "segecem": "CEM",
    "secretaría general de la cem": "CEM",
}


def _strip_prefix(s: str) -> str:
    """Remove honorific prefixes."""
    prefixes = [
        r"^mons\.\s*", r"^monseñor\s+", r"^excelent[íi]simo\s+(e\s+ilustr[íi]simo\s+)?señor\s+",
        r"^ilustr[íi]simo\s+señor\s+", r"^reverendsimo\s+", r"^cardenal\s+",
        r"^card\.\s*", r"^don\s+", r"^fr\.\s*", r"^fray\s+", r"^dr\.\s*",
        r"^arzobispo\s+(don\s+)?", r"^obispo\s+", r"^p\.\s*", r"^pbro\.\s*",
    ]
    for p in prefixes:
        s = re.sub(p, "", s, flags=re.IGNORECASE)
    return s.strip()


def _strip_suffix(s: str) -> str:
    """Remove order/title suffixes."""
    suffixes = [
        r"\s+o\.p\.?\s*$", r"\s+op\s*$", r"\s+l\.c\.?\s*$", r"\s+lc\s*$",
        r"\s+m\.j\.?\s*$", r"\s+mj\s*$", r"\s+m\.n\.m\.?\s*$",
        r"\s+c\.m\.g\.?\s*$", r"\s+s\.j\.?\s*$", r"\s+sj\s*$",
        r"\s+curriculum.*$", r"\s+escrito\s*$", r"\s+vitae\s*$",
    ]
    for sx in suffixes:
        s = re.sub(sx, "", s, flags=re.IGNORECASE)
    return s.strip()


def normalize_name(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "Unknown"

    # Take only the first name if multiple are separated by semicolons
    # (joint pastoral letters: keep first signer as primary)
    primary = raw.split(";")[0].strip()
    primary = primary.strip(".")

    # Normalise whitespace / newlines
    primary = re.sub(r"\s+", " ", primary).strip()

    key = _strip_suffix(_strip_prefix(primary.lower())).strip()
    key = re.sub(r"\s+", " ", key)

    if key in NAME_MAP:
        return NAME_MAP[key]

    # Fuzzy fallback: partial key matching on known canonical names
    # (catches things like leading spaces, slight OCR errors)
    for map_key, canon in NAME_MAP.items():
        if len(map_key) > 8 and (map_key in key or key in map_key):
            return canon

    # Return cleaned version if no match found
    cleaned = _strip_suffix(_strip_prefix(primary)).strip()
    # Capitalise properly
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned if cleaned else "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
# 3.  NAME CONSOLIDATION  (runs on raw text before tokenisation)
# ══════════════════════════════════════════════════════════════════════════════
# Replace multi-word bishop names with a single underscore-joined token so that
# LDA sees e.g. "mendez_arceo" as one meaningful unit, not three fragments.
# Order matters: longer / more specific patterns first.

_NAME_CONSOLIDATION = [
    # Sergio Méndez Arceo — "the Red Bishop" of Cuernavaca
    (re.compile(r"\bsergio\s+m[eé]ndez\s+arceo\b",   re.I | re.U), "mendez_arceo"),
    (re.compile(r"\bm[eé]ndez\s+arceo\b",             re.I | re.U), "mendez_arceo"),
    # Carlos Aguiar Retes — current Archbishop of Mexico City / Cardinal
    (re.compile(r"\bcarlos\s+aguiar\s+retes\b",        re.I | re.U), "aguiar_retes"),
    (re.compile(r"\baguiar\s+retes\b",                 re.I | re.U), "aguiar_retes"),
    # Samuel Ruiz García — Bishop of San Cristóbal, EZLN mediator
    (re.compile(r"\bsamuel\s+ruiz\s+garc[íi]a\b",     re.I | re.U), "samuel_ruiz"),
    (re.compile(r"\bsamuel\s+ruiz\b",                  re.I | re.U), "samuel_ruiz"),
    (re.compile(r"\bruiz\s+garc[íi]a\b",               re.I | re.U), "samuel_ruiz"),
    # Ernesto Corripio Ahumada — Cardinal Archbishop of Mexico City (pre-Rivera)
    (re.compile(r"\bcorripio\s+ahumada\b",             re.I | re.U), "corripio_ahumada"),
    # Norberto Rivera Carrera — Cardinal Archbishop of Mexico City
    (re.compile(r"\brivera\s+carrera\b",               re.I | re.U), "rivera_carrera"),
    (re.compile(r"\bnorberto\s+rivera\b",              re.I | re.U), "rivera_carrera"),
    # Francisco Robles Ortega — Cardinal Archbishop of Guadalajara
    (re.compile(r"\brobles\s+ortega\b",                re.I | re.U), "robles_ortega"),
    # Juan Sandoval Íñiguez — Cardinal Archbishop of Guadalajara (emeritus)
    (re.compile(r"\bsandoval\s+[íi]ñ?iguez\b",        re.I | re.U), "sandoval_iniguez"),
    # Raúl Vera López — Bishop of Saltillo, human rights advocate
    (re.compile(r"\bvera\s+l[oó]pez\b",                re.I | re.U), "vera_lopez"),
    (re.compile(r"\bra[úu]l\s+vera\b",                 re.I | re.U), "vera_lopez"),
]


def consolidate_names(text: str) -> str:
    """Replace multi-word bishop names with single canonical tokens."""
    if not isinstance(text, str):
        return text
    for pattern, replacement in _NAME_CONSOLIDATION:
        text = pattern.sub(replacement, text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

# Patterns to remove from text before tokenisation
_BOILERPLATE_RE = re.compile(
    r"("
    r"escrito\s+por\s+[^\n]*"               # "Escrito por ..."
    r"|secretar[íi]a\s+\w+(\s+\w+){0,3}"   # "Secretaría de ..."
    r"|\d{1,2}\s*/\s*\d{1,2}"              # page numbers "1 / 9"
    r"|(lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)"
       r",\s*\d{1,2}\s+de\s+\w+\s+de\s+\d{4}(\s+\d{2}:\d{2})?"  # date lines
    r"|https?://\S+"                        # URLs
    r"|www\.\S+"                            # www links
    r"|\S+@\S+\.\S+"                        # emails
    r"|\d{4}-\d{2}-\d{2}"                  # ISO dates
    r"|consultado\s+el\s+\d{1,2}\s+de\s+\w+\s+de\s+\d{4}"  # Wikipedia "Consultado el..."
    r"|↑\s*[^\n]+"                          # Wikipedia footnote markers
    r"|ISBN\s+[\d\-X]+"                     # ISBN strings
    r"|\bPRODVEI?N?\s+\w+\s+\w+"           # Mensajero OCR header garbage (PRODVEIN TCLAIAL...)
    r"|ENCELPEASNIATLSA\w*"               # OCR fragment of "ECLESIÁSTICA"
    r"|TCLAIAL|NTLCAIAL|NECELPAES\w*"    # more OCR fragments
    r"|\b(xiv|xv|xvi|xvii|xviii|xix|xxi?v?|xxv|xxx)\b"  # standalone Roman numerals
    r"|\d{1,2}:\d{2}\s*horas?"             # clock times "18:00 horas"
    r")",
    re.IGNORECASE | re.UNICODE,
)

_TOKEN_RE = re.compile(r"\b[a-záéíóúüñ]{3,}\b", re.IGNORECASE | re.UNICODE)


def preprocess(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    # Step 1: consolidate multi-word bishop names into single tokens
    text = consolidate_names(text)
    # Step 2: remove boilerplate patterns
    text = _BOILERPLATE_RE.sub(" ", text)
    # Step 3: tokenise — alphabetic runs ≥ 3 chars; underscore-joined names preserved
    tokens = re.findall(r"\b[a-záéíóúüñ_]{3,}\b", text.lower(), re.UNICODE)
    # Step 4: remove stopwords (canonical name tokens are never in STOPS)
    tokens = [t for t in tokens if t not in STOPS]
    return tokens


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # 4.  LOAD & PREPARE DATA
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n[1/6] Loading data...")
    df = pd.read_csv(CSV_IN, encoding="utf-8-sig")

    # Rename new-corpus columns to internal names used throughout this script
    df = df.rename(columns={"bishop": "name", "date": "date_final"})
    df = df[["doc_id", "source", "name", "text", "territory", "date_final", "year"]].copy()
    df["date_final"] = pd.to_datetime(df["date_final"], errors="coerce")
    # Use pre-computed year where date parse fails (e.g. year-only entries)
    df["year"] = df["date_final"].dt.year.where(df["date_final"].notna(), df["year"])

    print(f"  Loaded {len(df)} documents from {df['source'].nunique()} sources")
    print(f"  Source counts: {df['source'].value_counts().to_dict()}")

    # Normalize names
    print("[2/6] Normalising bishop names...")
    df["name_normalized"] = df["name"].apply(normalize_name)

    # Show normalisation summary
    raw_counts   = df["name"].dropna().nunique()
    canon_counts = df["name_normalized"].nunique()
    print(f"  Raw unique names : {raw_counts}")
    print(f"  Canonical names  : {canon_counts}  (reduced by {raw_counts - canon_counts})")

    # Top 15 after normalisation
    top_bishops = df["name_normalized"].value_counts().head(15)
    print("\n  Top 15 authors (canonical):")
    for nm, cnt in top_bishops.items():
        print(f"    {cnt:4d}  {nm}")

    # ── tokenise ──────────────────────────────────────────────────────────────────
    print("\n[3/6] Preprocessing texts...")
    df["tokens"] = df["text"].apply(preprocess)
    df["token_count"] = df["tokens"].apply(len)

    print(f"  Docs with ≥ 10 tokens : {(df['token_count'] >= 10).sum()} / {len(df)}")
    print(f"  Median token count    : {df['token_count'].median():.0f}")
    print(f"  Mean   token count    : {df['token_count'].mean():.0f}")

    # Drop docs that are too short after stopword removal
    df_model = df[df["token_count"] >= 10].copy().reset_index(drop=True)
    print(f"  Kept {len(df_model)} documents for modelling ({len(df)-len(df_model)} dropped < 10 tokens)")

    # ── build dictionary & corpus ─────────────────────────────────────────────────
    print("\n[4/6] Building dictionary & corpus...")

    # Remove tokens that appear in < 5 docs or > 60% of docs
    dictionary = corpora.Dictionary(df_model["tokens"])
    n_before = len(dictionary)
    dictionary.filter_extremes(no_below=5, no_above=0.60, keep_n=50_000)
    print(f"  Dictionary: {n_before} → {len(dictionary)} terms after filtering")

    corpus = [dictionary.doc2bow(tokens) for tokens in df_model["tokens"]]

    # ══════════════════════════════════════════════════════════════════════════════
    # 5.  FIND OPTIMAL NUMBER OF TOPICS  (coherence sweep 5 – 20)
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n[5/6] Coherence sweep (k = 5 … 20) — this may take a few minutes...")

    k_range = range(5, 21)
    coherence_scores = []

    for k in k_range:
        lda = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=42,
            passes=10,
            alpha="auto",
            eta="auto",
            per_word_topics=False,
        )
        cm = CoherenceModel(
            model=lda, texts=df_model["tokens"], dictionary=dictionary,
            coherence="c_v", processes=1,   # single-process avoids Windows spawn issues
        )
        score = cm.get_coherence()
        coherence_scores.append({"k": k, "coherence_cv": round(score, 4)})
        print(f"    k={k:2d}  c_v={score:.4f}")

    coh_df = pd.DataFrame(coherence_scores)
    coh_df.to_csv(OUT_DIR / "coherence_scores_full.csv", index=False, encoding="utf-8-sig")

    best_k = int(coh_df.loc[coh_df["coherence_cv"].idxmax(), "k"])
    print(f"\n  Best k = {best_k}  (c_v = {coh_df['coherence_cv'].max():.4f})")

    # Plot coherence
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(coh_df["k"], coh_df["coherence_cv"], marker="o", linewidth=2.5,
            color="steelblue", markersize=8)
    ax.axvline(best_k, color="crimson", linestyle="--", linewidth=1.8,
               label=f"Best k={best_k}")
    ax.set_xlabel("Number of Topics (k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Coherence Score (c_v)", fontsize=12, fontweight="bold")
    ax.set_title("LDA Coherence Score by Number of Topics", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "00_coherence_by_k.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] 00_coherence_by_k.png")

    # ══════════════════════════════════════════════════════════════════════════════
    # 6.  FIT FINAL LDA MODEL
    # ══════════════════════════════════════════════════════════════════════════════
    print(f"\n[6/6] Fitting final LDA model with k={best_k}...")

    lda_final = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=best_k,
        random_state=42,
        passes=20,          # more passes for final model
        alpha="auto",
        eta="auto",
        per_word_topics=False,
    )

    # ── topics × terms ────────────────────────────────────────────────────────────
    TOP_N_WORDS = 20
    rows_terms = []
    for topic_id in range(best_k):
        for rank, (term, weight) in enumerate(
            lda_final.show_topic(topic_id, topn=TOP_N_WORDS), start=1
        ):
            rows_terms.append({
                "topic": topic_id + 1,       # 1-indexed
                "rank": rank,
                "term": term,
                "weight": round(weight, 6),
            })

    terms_df = pd.DataFrame(rows_terms)
    terms_df.to_csv(OUT_DIR / "topics_terms_full.csv", index=False, encoding="utf-8-sig")
    print(f"  [saved] topics_terms_full.csv  ({len(terms_df)} rows)")

    # Print top-10 words per topic
    print("\n  TOP 10 WORDS PER TOPIC:")
    for tid in range(1, best_k + 1):
        words = terms_df[terms_df["topic"] == tid].head(10)["term"].tolist()
        print(f"  Topic {tid:2d}: {', '.join(words)}")

    # ── document × topic assignments ──────────────────────────────────────────────
    topic_prob_cols = [f"topic_{i+1}_prob" for i in range(best_k)]

    doc_topic_rows = []
    for i, (bow, doc_topics) in enumerate(
        zip(corpus, lda_final.get_document_topics(corpus, minimum_probability=0))
    ):
        probs = {f"topic_{tid+1}_prob": round(p, 4) for tid, p in doc_topics}
        primary = max(doc_topics, key=lambda x: x[1])
        row = {
            "doc_idx": i,
            "doc_id": df_model.loc[i, "doc_id"],
            "source": df_model.loc[i, "source"],
            "name_raw": df_model.loc[i, "name"],
            "name_normalized": df_model.loc[i, "name_normalized"],
            "territory": df_model.loc[i, "territory"],
            "date": df_model.loc[i, "date_final"],
            "year": int(df_model.loc[i, "year"]) if pd.notna(df_model.loc[i, "year"]) else None,
            "token_count": df_model.loc[i, "token_count"],
            "primary_topic": primary[0] + 1,
            "primary_topic_prob": round(primary[1], 4),
        }
        row.update(probs)
        doc_topic_rows.append(row)

    doc_topics_df = pd.DataFrame(doc_topic_rows)
    doc_topics_df.to_csv(OUT_DIR / "document_topics_full.csv", index=False, encoding="utf-8-sig")
    print(f"\n  [saved] document_topics_full.csv  ({len(doc_topics_df)} rows)")

    # ── topic summary ─────────────────────────────────────────────────────────────
    summary_rows = []
    for tid in range(1, best_k + 1):
        subset = doc_topics_df[doc_topics_df["primary_topic"] == tid]
        top_words = terms_df[terms_df["topic"] == tid].head(8)["term"].tolist()
        top_territories = subset["territory"].dropna().value_counts().head(3).index.tolist()
        top_authors    = subset["name_normalized"].value_counts().head(3).index.tolist()
        top_sources    = subset["source"].value_counts().head(3).index.tolist()
        summary_rows.append({
            "topic": tid,
            "n_docs": len(subset),
            "pct_docs": f"{100*len(subset)/len(doc_topics_df):.1f}%",
            "top_8_words": ", ".join(top_words),
            "top_territories": " | ".join(top_territories),
            "top_authors": " | ".join(top_authors),
            "top_sources": " | ".join(top_sources),
            "mean_prob": round(subset["primary_topic_prob"].mean(), 3),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "topics_summary_full.csv", index=False, encoding="utf-8-sig")
    print(f"  [saved] topics_summary_full.csv")

    print("\n  TOPIC SUMMARY:")
    print(summary_df.to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════════
    # VISUALISATIONS
    # ══════════════════════════════════════════════════════════════════════════════

    palette = sns.color_palette("tab20", best_k)

    # ── A: topic distribution bar chart ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    topic_counts = doc_topics_df["primary_topic"].value_counts().sort_index()
    bars = ax.bar(topic_counts.index, topic_counts.values,
                  color=palette[:len(topic_counts)], edgecolor="black", linewidth=1)
    for bar, cnt in zip(bars, topic_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(cnt), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("Topic", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Documents", fontsize=12, fontweight="bold")
    ax.set_title(f"Document Distribution Across {best_k} Topics\n(n={len(doc_topics_df)})",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(topic_counts.index)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_topic_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  [saved] 01_topic_distribution.png")

    # ── B: heatmap — topic × territory ───────────────────────────────────────────
    top_terr = doc_topics_df["territory"].value_counts().head(15).index
    ct = pd.crosstab(
        doc_topics_df[doc_topics_df["territory"].isin(top_terr)]["territory"],
        doc_topics_df[doc_topics_df["territory"].isin(top_terr)]["primary_topic"],
    )
    fig, ax = plt.subplots(figsize=(max(10, best_k + 2), 8))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, cbar_kws={"label": "Documents"}, ax=ax)
    ax.set_title("Topic Distribution by Territory (Top 15)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Topic", fontsize=11, fontweight="bold")
    ax.set_ylabel("Territory", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_topic_by_territory.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  [saved] 02_topic_by_territory.png")

    # ── C: topic evolution over time ──────────────────────────────────────────────
    yearly = doc_topics_df[doc_topics_df["year"].notna()].copy()
    yearly["year"] = yearly["year"].astype(float).astype(int)
    if yearly["year"].nunique() > 1:
        pivot = yearly.groupby(["year", "primary_topic"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(13, 6))
        for tid in pivot.columns:
            ax.plot(pivot.index, pivot[tid], marker="o", linewidth=2,
                    label=f"Topic {tid}", color=palette[tid - 1], markersize=6)
        ax.set_xlabel("Year", fontsize=12, fontweight="bold")
        ax.set_ylabel("Documents", fontsize=12, fontweight="bold")
        ax.set_title("Topic Volume Over Time", fontsize=13, fontweight="bold", pad=12)
        ax.legend(ncol=3, fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "03_topic_over_time.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("  [saved] 03_topic_over_time.png")

    # ── D: top-10 word weights per topic (bar charts) ─────────────────────────────
    ncols = 3
    nrows = -(-best_k // ncols)   # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
    axes = axes.flatten()
    for i, tid in enumerate(range(1, best_k + 1)):
        sub = terms_df[terms_df["topic"] == tid].head(10)
        axes[i].barh(sub["term"][::-1], sub["weight"][::-1],
                     color=palette[i], edgecolor="grey", linewidth=0.6)
        axes[i].set_title(f"Topic {tid}", fontsize=11, fontweight="bold")
        axes[i].tick_params(axis="y", labelsize=9)
        axes[i].grid(axis="x", alpha=0.3)
    for j in range(best_k, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Top 10 Words per Topic (LDA Weight)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_top_words_per_topic.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  [saved] 04_top_words_per_topic.png")

    # ── E: heatmap — topic × bishop (top 20 authors) ─────────────────────────────
    top_authors = doc_topics_df["name_normalized"].value_counts().head(20).index
    ct_auth = pd.crosstab(
        doc_topics_df[doc_topics_df["name_normalized"].isin(top_authors)]["name_normalized"],
        doc_topics_df[doc_topics_df["name_normalized"].isin(top_authors)]["primary_topic"],
    )
    fig, ax = plt.subplots(figsize=(max(10, best_k + 2), 9))
    sns.heatmap(ct_auth, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, cbar_kws={"label": "Documents"}, ax=ax)
    ax.set_title("Topic Profile — Top 20 Authors / Institutions", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Topic", fontsize=11, fontweight="bold")
    ax.set_ylabel("Author", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_topic_by_bishop.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  [saved] 05_topic_by_bishop.png")

    # ══════════════════════════════════════════════════════════════════════════════
    # SAVE MODEL
    # ══════════════════════════════════════════════════════════════════════════════
    model_path = OUT_DIR / "lda_model_full"
    lda_final.save(str(model_path))
    dictionary.save(str(OUT_DIR / "dictionary_full.gensim"))
    print(f"\n  [saved] LDA model → {model_path}")

    # ══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print(f"DONE.  All outputs in:\n  {OUT_DIR}")
    print("=" * 72)
    files = sorted(OUT_DIR.glob("*"))
    for f in files:
        size = f.stat().st_size / 1024
        print(f"  {f.name:<45s}  {size:6.1f} KB")
