# hierNet_variable_selection.R —— 20 个 additives 的变量筛选（回归，双重排名法）
# 运行方式：
#   source("C:/PyCharm/Blueness/hierNet_variable_selection.R", echo = TRUE)
# 或命令行：
#   Rscript C:/PyCharm/Blueness/hierNet_variable_selection.R

# ===== 0) 依赖与全局选项 =====
options(repos = c(CRAN = "https://cloud.r-project.org"))
packages <- c("hierNet", "readxl")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# 是否启用强层级（TRUE 更保守：交互进入前两侧主效应必须在场）
STRONG_HIERARCHY <- FALSE

# 交叉验证 λ 选择规则： "1se"（更保守） 或 "min"（最小误差，激进）
CV_RULE <- "1se"

# 弱层级下交互矩阵可能非对称；为“平均交互效应”直观起见，默认对称化
SYMMETRIZE_THETA <- TRUE

# 交互排名所用的 Top-K（只看正向交互）——按你的要求设为 3
TOPK <- 3

# K 折交叉验证
N_FOLDS <- 10
set.seed(1)

# ===== 1) 数据路径与读取 =====
fn <- "C:/PyCharm/Blueness/data.xlsx"
if (!file.exists(fn)) {
  message("未在固定路径找到 data.xlsx，将弹出对话框，请选择 XLSX 文件。")
  fn <- file.choose()
}
stopifnot(file.exists(fn))

df <- readxl::read_excel(fn)
df <- as.data.frame(df)

names(df) <- tolower(names(df))

# ===== 2) 只保留 20 个 additives，构造 X / y =====
additives <- c(
  "peg20k","pl127","bsa","pva","tw80","glycerol","tw20","imidazole",
  "tx100","edta","mgcl2","sucrose","cacl2","znso4","paa","mnso4","peg200k","feso4","peg6000","peg400"
)

missing_cols <- setdiff(additives, names(df))
if (length(missing_cols) > 0) stop("数据缺少以下列： ", paste(missing_cols, collapse = ", "))
if (!("auc" %in% names(df))) stop("数据缺少应答变量列：auc")

y <- df$auc
X_raw <- as.matrix(df[, additives, drop = FALSE])

# 去缺失（这也会移除上面 as.numeric 转换失败时产生的 NA）
ok <- complete.cases(cbind(y, X_raw))
if (sum(!ok) > 0) message("有 ", sum(!ok), " 行含 NA，已删除。")
y <- y[ok]
X_raw <- X_raw[ok, , drop = FALSE]

# ===== 2.x) Z-score 标准化到均值0方差1（按整份数据；保护常量列）=====
zscore_scale <- function(M) {
  mu <- colMeans(M)
  sdv <- apply(M, 2, sd)
  sdv[sdv == 0 | is.na(sdv)] <- 1
  M_scaled <- sweep(sweep(M, 2, mu, FUN = "-"), 2, sdv, FUN = "/")
  attr(M_scaled, "means") <- mu
  attr(M_scaled, "sds") <- sdv
  M_scaled
}
X <- zscore_scale(X_raw)

# 导出缩放参数（便于未来数据复用同一缩放）
scaler_df <- data.frame(var = colnames(X),
                        mean = attr(X, "means"),
                        sd = attr(X, "sds"))
write.csv(scaler_df, file.path(dirname(fn), "scaler_zscore.csv"), row.names = FALSE)

# ===== 3) 交叉验证选择 λ =====
path_fit <- hierNet.path(X, y, strong = STRONG_HIERARCHY)
cv_fit   <- hierNet.cv(path_fit, X, y, nfolds = N_FOLDS)

cat("lamhat (min error) =", cv_fit$lamhat, "\n")
cat("lamhat.1se (1-SE)  =", cv_fit$lamhat.1se, "\n")
lam <- if (tolower(CV_RULE) == "min") cv_fit$lamhat else cv_fit$lamhat.1se

# ===== 4) 用选定的 λ 拟合最终模型 =====
final_fit <- hierNet(X, y, lam = lam, strong = STRONG_HIERARCHY)
cat("final_fit$lam =", final_fit$lam, "\n")

# ===== 5) 主效应：带符号 β，再取正部分作为“主增强效应分数” =====
main_beta <- with(final_fit, bp - bn)
names(main_beta) <- colnames(X)
main_score <- pmax(main_beta, 0)

# ===== 6) 交互矩阵与“Top-3 正向交互均值” + 个数统计 =====
Theta <- final_fit$th
dimnames(Theta) <- list(colnames(X), colnames(X))

if (SYMMETRIZE_THETA) Theta <- (Theta + t(Theta)) / 2

p <- ncol(X)
Theta_nodiag <- Theta
diag(Theta_nodiag) <- 0

pos_count  <- rowSums(Theta_nodiag >  0)
neg_count  <- rowSums(Theta_nodiag <  0)
zero_count <- (p - 1) - pos_count - neg_count

Theta_pos <- pmax(Theta_nodiag, 0)

topk_mean <- function(v, k = TOPK) {
  v_sorted <- sort(v, decreasing = TRUE)
  k_eff <- min(k, length(v_sorted))
  mean(c(v_sorted[1:k_eff], rep(0, k - k_eff)))
}
pair_score <- apply(Theta_pos, 1, topk_mean, k = TOPK)

# ===== 6.1) 各自排名 =====
r_main <- rank(-main_score, ties.method = "min")
top_main <- names(sort(main_score, decreasing = TRUE))

r_pair <- rank(-pair_score, ties.method = "min")
top_pair <- names(sort(pair_score, decreasing = TRUE))

# ===== 6.2) 双重排名逐步相交选 6 个 =====
borda_all <- r_main + r_pair
selected <- character(0)
k_stop <- NA_integer_

for (k in 1:p) {
  cand <- intersect(top_main[1:k], top_pair[1:k])
  new  <- setdiff(cand, selected)
  if (length(new) > 0) {
    sc <- borda_all[new]
    new <- new[order(sc, decreasing = FALSE)]
    selected <- unique(c(selected, new))
  }
  if (length(selected) >= 6) {
    selected <- selected[1:6]
    k_stop <- k
    break
  }
}

if (length(selected) < 6) {
  rest <- setdiff(names(borda_all), selected)
  rest <- rest[order(borda_all[rest], decreasing = FALSE)]
  selected <- c(selected, head(rest, 6 - length(selected)))
  k_stop <- p
}

# ===== 6.3) 回显对照 =====
cat("\n[双重排名选择] 选出的 6 个 additives：\n")
print(selected)
cat("达到 6 个时的阈值 k =", k_stop, "\n")

cat("\nTop 6 by main effect score ((beta)+):\n")
print(top_main[1:6])
cat(paste0("\nTop 6 by avg of top-", TOPK, " positive interactions:\n"))
print(top_pair[1:6])

# ===== 7) 导出 =====
out_dir <- dirname(fn)
out_csv <- file.path(out_dir, "hiernet_double_ranking.csv")
out_top6_csv <- file.path(out_dir, "hiernet_selected_top6.csv")

ranking_df <- data.frame(
  additive          = names(main_beta),
  main_beta         = as.numeric(main_beta),
  main_score_pos    = as.numeric(main_score),
  main_rank         = as.integer(r_main[names(main_beta)]),
  inter_topk_avg    = as.numeric(pair_score),
  inter_rank        = as.integer(r_pair[names(main_beta)]),
  pos_interactions  = as.integer(pos_count[names(main_beta)]),
  neg_interactions  = as.integer(neg_count[names(main_beta)]),
  zero_interactions = as.integer(zero_count[names(main_beta)]),
  borda_score       = as.integer((r_main + r_pair)[names(main_beta)]),
  selected6         = names(main_beta) %in% selected
)

ord <- order(!ranking_df$selected6, ranking_df$borda_score,
             -ranking_df$main_score_pos, -ranking_df$inter_topk_avg)
ranking_df <- ranking_df[ord, ]

write.csv(ranking_df, out_csv, row.names = FALSE)
sel_tbl <- subset(ranking_df, selected6)
write.csv(sel_tbl, out_top6_csv, row.names = FALSE)

cat("\n结果已导出到：\n")
cat(" - 双重排名明细：", out_csv, "\n")
cat(" - 双重排名 Top-6：", out_top6_csv, "\n")
cat(" - Z-score 缩放参数：", file.path(dirname(fn), "scaler_zscore.csv"), "\n")
cat(" - 交互 Top-K 取值：K =", TOPK, "\n")