================================================================================
REPLICATION PACKAGE
The Ideology of Mexico's Catholic Bishops: Topic Modeling, Text Scaling,
and Episcopal Networks, 1976-2026
Alejandro Díaz-Domínguez, Religions (MDPI), 2026
================================================================================

CONTENTS
--------
  data/                 the datasets (each .csv has a .json twin)
  data/lda_model/       the fitted nine-topic LDA model (gensim)
  code/analysis/        reproduce every table and figure from data/
  code/construction/    the text-model estimation scripts (need the full text)
  output/               tables and figures as produced by code/analysis/

QUICK START
-----------
Every table and figure in the article and its Supplementary Materials can be
reproduced from data/ alone. Run, from code/analysis/:

  python 01_descriptive_tables.py   Tables 1-6, S1, S2, S5, S6
  Rscript 02_regression_tables.R    Tables 7-9, S7, S8 and Figure S7
  python 03_figures.py              Figures 1-4, S1-S6, S8

Results are written to output/tables/ and output/figures/. The copies already
in output/ were produced by these scripts.

Software used: Python 3.11.9 (pandas 3.0.2, numpy 2.4.4, scipy 1.17.1,
matplotlib 3.10.8, seaborn 0.13.2, networkx 3.6.1, adjustText 1.4.0) and
R 4.6.0 (lme4 2.0.1, performance 0.16.0, sandwich 3.1.1, lmtest 0.9.40).
The construction scripts also use gensim 4.4.0, scikit-learn 1.9.0 and
spaCy 3.8.14.

FILE FORMATS
------------
Every .csv is UTF-8 with a byte-order mark, so accents display correctly in
Excel on Windows. Every .csv has a .json twin with the same rows, written in
pure 7-bit ASCII (accented characters escaped as \uXXXX), which cannot be
mis-decoded by any reader or locale.

================================================================================
PART 1 -- DATA
================================================================================
 01_corpus_metadata_N2330.csv        the corpus: one row per document (2,330),
                                     with source, author, territory, date,
                                     title and length; the text is not included
 01b_duplicates_removed_N135.csv     near-duplicate clusters found during corpus
                                     assembly, and which copy was kept
 01c_diocese_state_map.csv           territory (see) -> state key, 75 sees

 02_topic_model_document_topics.csv  document x topic matrix (nine topics),
                                     primary topic and ideology score
 02b_lda_coherence_by_k.csv          c_v coherence, K = 5 to 20 (Table S1)
 02c_topic_terms_top20.csv           top 20 terms per topic
 02d_topic_summary.csv               documents and share per topic (Table 2)
 02e-02k (ALT)                       alternative topic solutions, for
                                     sensitivity checks: a six-topic solution
                                     (02f, 02g), a nine-topic re-estimation on
                                     the deduplicated corpus (02h-02j), its
                                     correspondence to the reported topics (02e)
                                     and its coherence sweep (02k)

 03_wordfish_ideal_points_N39.csv    Wordfish ideal points and standard errors,
                                     39 actors (Table S3, Figures 3-4)
 03b_wordfish_word_parameters_N39.csv  word parameters (alpha, beta) of the same
                                     estimation, 3,000 words (Figure S1)
 03c_wordfish_church_vs_press_N39.csv  the same actors re-estimated on Church
                                     documents only and press documents only
                                     (Table S4, Figure S6)

 04_consecration_network_nodes.csv   381 people: consecrations, centrality,
                                     lineage cluster
 04b_consecration_edges_clean.csv    628 consecration ties (238 principal
                                     consecrator, 390 co-consecrator)
 04c_edges_dropped_not_people.csv    source rows whose "consecrator" is a see
                                     rather than a person, excluded from 04b

 05_state_year_covariates_1970_2026.csv  32 states x 57 years
 06_document_panel_FINAL.csv         the analysis panel: 1,682 documents with
                                     state-year covariates attached (Tables 5,
                                     7-9, S5, S7, S8)

 lda_model/                          the fitted LDA model and its dictionary

KEY VARIABLES
  ideology_score / ideology_index    T1 + T3 + T9 - T5 - T8 (same quantity; the
                                     first name is used in 02, the second in 06)
  theta_oriented                     Wordfish ideal point, negative = progressive
  author_type                        individual (named bishop), institutional
                                     (issued in the name of the CEM, a diocese or
                                     another Church body), or unattributed
  source / bucket                    outlet; "church" = CEM, diocesan bulletins,
                                     Voces, El Mensajero and the 2018 messages,
                                     "press" = every news source
  pct_protestant, pct_indigenous     state shares (0-1 and percent, respectively)
  hdi, hdi_z                         Human Development Index, in levels and as a
                                     within-year z score
  region4, region5                   macro regions (Table 8 uses region4,
                                     Table 5 uses region5)
  *_extrapolated                     1 when a 2021-2026 value carries the 2020
                                     census value forward

================================================================================
PART 2 -- TABLE AND FIGURE MAP
================================================================================
 OUTPUT       CONTENT                               DATA                SCRIPT
 Table 1      corpus by source                      01                  01
 Table 2      the nine topics                       02c, 02d            01
 Table 3      principal consecrators                04, 04b             01
 Table 4      tests on the full corpus              02, 06, 03 (N39)    01
 Table 5      ideology by region                    06                  01
 Table 6      ideology by decade                    02                  01
 Table 7      descriptive statistics                06                  02
 Table 8      multilevel models                     06                  02
 Table 9      decomposition by topic                06                  02
 Table S1     LDA coherence by K                    02b                 01
 Table S2     primary topic by decade               02                  01
 Table S3     Wordfish ideal points                 03 (N39)            (data)
 Table S4     Wordfish by source type               03c (N39)           (data)
 Table S5     region by source type                 06                  01
 Table S6     institutional vs. individual          02                  01
 Table S7     linear vs. multilevel                 06                  02
 Table S8     HDI under two scalings                06                  02
 Figure 1     topic prevalence over time            02                  03
 Figure 2     topics by author                      02                  03
 Figure 3     Wordfish ideal points                 03 (N39)            03
 Figure 4     Wordfish vs. topic-space PCA          03 (N39)            03
 Figure S1    discriminating words                  03b (N39)           03
 Figure S2    centrality of consecrators            04                  03
 Figure S3    consecration network                  04, 04b             03
 Figure S4    documents per topic                   02                  03
 Figure S5    LDA coherence by K                    02b                 03
 Figure S6    Wordfish by source type               03c (N39)           03
 Figure S7    linear vs. multilevel estimates       06                  02
 Figure S8    ideology by decade                    02                  03

 Scripts: 01 = 01_descriptive_tables.py, 02 = 02_regression_tables.R,
          03 = 03_figures.py (all in code/analysis/).

================================================================================
PART 3 -- HOW THE DATA WERE BUILT
================================================================================
CORPUS
  2,701 documents were collected from the sources listed in Table 1. Cleaning
  removed off-topic items, exact and near-duplicate copies and items below ten
  content tokens, and five pastoral messages from 2018 were added, leaving 2,330
  documents. Near-duplicates are identified by MD5 on normalized text for
  identical documents and by tf-idf cosine similarity above 0.99, with
  connected-components clustering, for the rest; the clusters are in 01b.

  Of the 2,330 documents, 190 are national-level documents with no diocese,
  457 carry neither a territory nor a recoverable bishop-diocese timeline, and
  1 has no year, leaving 1,682 documents in the state-level panel (06).

TOPIC MODEL
  code/construction/lda_fit.py estimates LDA models for K = 5 to 20 with gensim
  and selects K = 9 by c_v coherence. As stated in Section 5.1 of the article,
  the model was estimated before near-duplicate documents were removed;
  code/construction/lda_apply.py applies the fitted model (data/lda_model/) to
  the 2,330 retained documents and produces 02 and 02d exactly. The six-topic
  and re-estimated nine-topic alternatives are in 02e-02k.

THE IDEOLOGY SCORE
  ideology = T1 + T3 + T9 - T5 - T8: the probability mass on the social
  engagement topics (social pastoral and human development; episcopal peace
  calls; Chiapas, EZLN and indigenous rights) minus the mass on the
  spiritual/institutional topics (diocesan pastoral life; theology and
  homiletics). It is bounded in [-1, 1]; higher values denote more social
  engagement discourse. Table 9 decomposes it topic by topic.

THE WORDFISH SCALE
  code/construction/wordfish_39_actors.py estimates Wordfish at the actor level
  (a named territory or the CEM, at least ten documents), giving 39 actors and
  1,435 documents; the CEM actor includes the documents coded "Unknown".
  theta_oriented = -wordfish_score, so that negative values mark the
  progressive pole; word parameters are oriented the same way in Figure S1.
  code/construction/wordfish_church_vs_press.py re-estimates the scale within
  Church and press documents with a minimum of eight documents per actor
  (24 and 16 actors; Table S4).

COVARIATES
  Religion shares come from INEGI census counts for 1970, 1980, 1990, 2000,
  2010 and 2020, interpolated annually and taken over the religiously
  classified population. ENR is the inverse Herfindahl index over Catholic,
  non-Catholic and no-religion shares.

  The Human Development Index runs annually from 1970 to 2023 and is
  chain-linked at the 2010 overlap, because the post-2010 index uses a
  different UNDP methodology. It enters the models as a within-year z score;
  in levels it is collinear with the year term (correlation with year 0.74).
  Table S8 reports both scalings.

  Indigenous-language share runs annually from 1970 to 2020 and is chain-linked
  at 2010, because the 2020 tabulation uses a different denominator. The 1980
  census anchor is not used: 31 of 32 states sit above the 1970-1990 line at
  1980 by a near-constant 1.24 points, a coverage artifact rather than a real
  movement. Values for 2021-2026 carry the 2020 census values forward and are
  flagged by the *_extrapolated variables.

THE NETWORK
  Nodes are people, merged on a canonical key that strips titles, the deceased
  dagger and accents. Rows whose consecrator field holds a see rather than a
  person are excluded and listed in 04c. Centrality is computed on the whole
  graph; betweenness_main_component and pagerank_main_component restrict it to
  the largest connected component. n_principal_consecrations, reported in
  Table 3, counts principal consecrations only.

================================================================================
PART 4 -- WHAT IS NOT DISTRIBUTED
================================================================================
  The full text of the documents is not redistributed, because most items are
  copyrighted press articles. 01_corpus_metadata_N2330.csv gives each
  document's source, outlet, date and title, so that every item can be located
  in its original archive. The construction scripts in code/construction/
  (topic model estimation and application, Wordfish) need that text; everything
  else runs from data/. The consecration records come from the public
  Catholic-Hierarchy database (Cheney 2026), and the covariates from public
  INEGI and UNDP sources.
