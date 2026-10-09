# =============================================================================
# 02_regression_tables.R
#
# Reproduces the regression results of the article and its supplement from
# ../../data/06_document_panel_FINAL.csv (1,682 documents):
#
#   Table 7   descriptive statistics for the religious competition test
#   Table 8   multilevel models, crossed random intercepts for bishop and state
#   Table 9   decomposition of the composite score by component topic
#   Table S7  Table 8 compared with bishop-clustered linear models
#   Table S8  Human Development Index coefficient under two scalings
#   Figure S7 linear vs. multilevel estimates of the Protestant/Evangelical share
#
# Run from this folder:  Rscript 02_regression_tables.R
# Output: ../../output/tables/*.csv and ../../output/figures/Figure_S7.png
# Requires: R >= 4.2, lme4, performance, sandwich, lmtest
# =============================================================================

suppressPackageStartupMessages({
  library(lme4); library(performance); library(sandwich); library(lmtest)
})

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
DATA <- file.path(here, "..", "..", "data")
OUT  <- file.path(here, "..", "..", "output", "tables")
FIG  <- file.path(here, "..", "..", "output", "figures")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG, showWarnings = FALSE, recursive = TRUE)

d <- read.csv(file.path(DATA, "06_document_panel_FINAL.csv"),
              stringsAsFactors = FALSE, fileEncoding = "UTF-8-BOM")
d$prot_pct  <- d$pct_protestant * 100
d$indig_pct <- d$pct_indigenous
d$bishop    <- factor(d$name_normalized)
d$state     <- factor(d$state_final)
d$region    <- factor(d$region4,
                      levels = c("Centro", "Norte", "Bajio-Occidente", "Sur-Sureste"))
d$year_c    <- d$year_int - mean(d$year_int)
cat("documents:", nrow(d), " bishops:", nlevels(d$bishop), " states:", nlevels(d$state), "\n\n")

ctrl  <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
stars <- function(p) ifelse(is.na(p), "", ifelse(p < 0.001, "***", ifelse(p < 0.01, "**",
                     ifelse(p < 0.05, "*", ifelse(p < 0.10, "+", "")))))
wald_p <- function(est, se) 2 * pnorm(-abs(est / se))
save <- function(x, name) {
  write.csv(x, file.path(OUT, name), row.names = FALSE, fileEncoding = "UTF-8")
  cat("===", name, "\n"); print(x, row.names = FALSE); cat("\n")
}

# ---------------------------------------------------------------- Table 7
vars <- c(ideology_index = "Document ideology score",
          prot_pct       = "Protestant/Evangelical population share (%)",
          indig_pct      = "Indigenous population share (%)",
          hdi            = "Human Development Index",
          ENR            = "Effective number of religions")
t7 <- do.call(rbind, lapply(names(vars), function(v) {
  x <- d[[v]]
  data.frame(variable = vars[[v]], n = sum(!is.na(x)), mean = round(mean(x, na.rm = TRUE), 3),
             sd = round(sd(x, na.rm = TRUE), 3), min = round(min(x, na.rm = TRUE), 3),
             max = round(max(x, na.rm = TRUE), 3))
}))
save(t7, "table7_descriptives.csv")

# ---------------------------------------------------------------- Table 8
rhs <- c("Model 1" = "prot_pct",
         "Model 2" = "prot_pct + indig_pct",
         "Model 3" = "prot_pct + indig_pct + year_c",
         "Model 4" = "prot_pct + indig_pct + year_c + hdi_z",
         "Model 5" = "prot_pct + indig_pct + year_c + hdi_z + region")
ml <- lapply(rhs, function(r)
  lmer(as.formula(paste("ideology_index ~", r, "+ (1|bishop) + (1|state)")),
       data = d, REML = TRUE, control = ctrl))

pretty <- c(prot_pct = "Protestant/Evangelical share (%)",
            indig_pct = "Indigenous population share (%)",
            year_c = "Year (centered)",
            hdi_z = "Human Development Index (within-year z)",
            regionNorte = "Region: North",
            "regionBajio-Occidente" = "Region: West (Bajio-Occidente)",
            "regionSur-Sureste" = "Region: South-Southeast",
            "(Intercept)" = "Intercept")
cells <- sapply(ml, function(m) {
  cf <- summary(m)$coefficients
  sapply(names(pretty), function(tr) {
    if (!tr %in% rownames(cf)) return("")
    sprintf("%.3f%s (%.3f)", cf[tr, 1], stars(wald_p(cf[tr, 1], cf[tr, 2])), cf[tr, 2])
  })
})
r2 <- lapply(ml, r2_nakagawa)
t8 <- rbind(data.frame(term = unname(pretty), cells, check.names = FALSE, row.names = NULL),
            c("Observations", sapply(ml, nobs)),
            c("R2 (marginal / conditional)",
              sapply(r2, function(x) sprintf("%.2f / %.2f", x$R2_marginal, x$R2_conditional))))
save(t8, "table8_multilevel.csv")

# ---------------------------------------------------------------- Table 9
comp <- c(topic_1_prob = "Topic 1 Social pastoral and human development",
          topic_3_prob = "Topic 3 Episcopal peace calls",
          topic_9_prob = "Topic 9 Chiapas, EZLN and indigenous rights",
          topic_5_prob = "Topic 5 Diocesan pastoral life",
          topic_8_prob = "Topic 8 Theology and homiletics",
          ideology_index = "Composite ideology score")
pole <- c("+ social engagement", "+ social engagement", "+ social engagement",
          "- spiritual/institutional", "- spiritual/institutional", "(all five)")
prot_cell <- function(v, dat) {
  m  <- lmer(as.formula(paste(v, "~", rhs[["Model 4"]], "+ (1|bishop) + (1|state)")),
             data = dat, REML = TRUE, control = ctrl)
  cf <- summary(m)$coefficients["prot_pct", ]
  sprintf("%+.4f%s (%.4f)", cf[1], stars(wald_p(cf[1], cf[2])), cf[2])
}
no_chiapas <- d[d$state_final != "Chiapas", ]
t9 <- data.frame(outcome = unname(comp), pole = pole,
                 protestant_share_coef_se = sapply(names(comp), prot_cell, dat = d),
                 excluding_chiapas = sapply(names(comp), prot_cell, dat = no_chiapas),
                 row.names = NULL)
save(t9, "table9_pole_decomposition.csv")

# ---------------------------------------------------------------- Table S7 + Figure S7
s7 <- do.call(rbind, lapply(names(rhs), function(nm) {
  o  <- lm(as.formula(paste("ideology_index ~", rhs[[nm]])), data = d)
  ct <- coeftest(o, vcov. = vcovCL(o, cluster = d$bishop[as.numeric(rownames(model.frame(o)))],
                                   type = "HC1"))
  cf <- summary(ml[[nm]])$coefficients
  m_ml <- lmer(as.formula(paste("ideology_index ~", rhs[[nm]], "+ (1|bishop) + (1|state)")),
               data = d, REML = FALSE, control = ctrl)   # ML fit for the AIC comparison
  data.frame(model = sub("Model ", "M", nm),
             ols_est = ct["prot_pct", 1], ols_se = ct["prot_pct", 2], ols_p = ct["prot_pct", 4],
             ml_est = cf["prot_pct", 1], ml_se = cf["prot_pct", 2],
             ml_p = wald_p(cf["prot_pct", 1], cf["prot_pct", 2]),
             icc_adjusted = icc(ml[[nm]])$ICC_adjusted,
             aic_gap = AIC(o) - AIC(m_ml))
}))
s7_print <- data.frame(
  Model = s7$model,
  `OLS, clustered SE` = sprintf("%.3f%s (%.3f)", s7$ols_est, stars(s7$ols_p), s7$ols_se),
  `Multilevel, crossed RE` = sprintf("%.3f%s (%.3f)", s7$ml_est, stars(s7$ml_p), s7$ml_se),
  `ICC (bishop+state, adj.)` = sprintf("%.2f", s7$icc_adjusted),
  `AIC gap (favors multilevel)` = sprintf("%.1f", s7$aic_gap), check.names = FALSE)
save(s7_print, "tableS7_ols_vs_multilevel.csv")

png(file.path(FIG, "Figure_S7.png"), width = 7.5, height = 4.8, units = "in", res = 600)
par(mar = c(4, 5.2, 3, 1))
x <- seq_len(nrow(s7))
lo <- c(s7$ols_est - 1.96 * s7$ols_se, s7$ml_est - 1.96 * s7$ml_se)
hi <- c(s7$ols_est + 1.96 * s7$ols_se, s7$ml_est + 1.96 * s7$ml_se)
plot(NA, xlim = c(0.5, nrow(s7) + 0.5), ylim = range(c(lo, hi, 0)), xaxt = "n",
     xlab = "", ylab = "Coefficient on Protestant/Evangelical\npopulation share (pct. points)",
     main = "OLS-clustered vs. multilevel estimates (95% CI)")
axis(1, at = x, labels = s7$model)
abline(h = 0, lty = 2, col = "grey40")
arrows(x - 0.15, s7$ols_est - 1.96 * s7$ols_se, x - 0.15, s7$ols_est + 1.96 * s7$ols_se,
       angle = 90, code = 3, length = 0.04, col = "grey30", lwd = 1.5)
points(x - 0.15, s7$ols_est, pch = 19, col = "grey30", cex = 1.2)
arrows(x + 0.15, s7$ml_est - 1.96 * s7$ml_se, x + 0.15, s7$ml_est + 1.96 * s7$ml_se,
       angle = 90, code = 3, length = 0.04, col = "#2166ac", lwd = 1.5)
points(x + 0.15, s7$ml_est, pch = 18, col = "#2166ac", cex = 1.6)
legend("bottomleft", bty = "n", pch = c(19, 18), col = c("grey30", "#2166ac"),
       legend = c("OLS, bishop-clustered SE", "Multilevel, crossed RE: bishop + state (Table 8)"))
invisible(dev.off())

# ---------------------------------------------------------------- Table S8
d$hdi_raw <- d$hdi
hdi_fit <- function(hvar) {
  m  <- lmer(as.formula(paste("ideology_index ~ prot_pct + indig_pct + year_c +", hvar,
                              "+ (1|bishop) + (1|state)")), data = d, REML = TRUE, control = ctrl)
  cf <- summary(m)$coefficients[hvar, ]
  data.frame(n = nobs(m), coef = round(cf[1], 4), se = round(cf[2], 4),
             p = signif(wald_p(cf[1], cf[2]), 3))
}
s8 <- cbind(specification = c("HDI in levels (chain-linked)", "HDI, within-year z score (reported in Table 8)"),
            rbind(hdi_fit("hdi_raw"), hdi_fit("hdi_z")))
save(s8, "tableS8_hdi_two_scalings.csv")
cat("Raw correlation of HDI with year:", round(cor(d$hdi, d$year_int), 2), "\n")
