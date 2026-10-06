# R Project 3 - Human Activity Recognition
# Classifying Physical Activity from Smartphone Sensor Data
# Last updated 5th of October 2026
# Laura Maria Fetz

library(tidyverse)
library(caret)
library(nnet)
library(e1071)

set.seed(2023)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
data_dir <- "data"
train_dir <- file.path(data_dir, "RawData", "Train")
test_dir <- file.path(data_dir, "RawData", "Test")
results_dir <- "results"
dir.create(results_dir, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Activity and sample labels
# -----------------------------------------------------------------------------
activity_labels <- read_delim(
  file.path(data_dir, "activity_labels.txt"),
  delim = " ", col_names = FALSE, trim_ws = TRUE, show_col_types = FALSE
) %>%
  select(X1, X2) %>%
  rename(activity_id = X1, activity = X2)

segment_labels <- read_delim(
  file.path(train_dir, "labels_train.txt"),
  delim = " ", col_names = FALSE, show_col_types = FALSE
)
colnames(segment_labels) <- c("trial", "userid", "activity_id", "start", "end")
segment_labels <- segment_labels %>%
  left_join(activity_labels, by = "activity_id")

# Expand segment labels to one activity label per raw sample.
sample_labels <- segment_labels %>%
  rowwise() %>%
  mutate(sampleid = list(start:end)) %>%
  ungroup() %>%
  mutate(segment = row_number()) %>%
  unnest(sampleid) %>%
  select(userid, trial, activity, sampleid, segment)

# -----------------------------------------------------------------------------
# Feature helpers
# -----------------------------------------------------------------------------
mode_value <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_character_)
  names(which.max(table(x)))
}

safe_cor <- function(x, y) {
  if (sum(complete.cases(x, y)) < 3) return(NA_real_)
  suppressWarnings(cor(x, y, use = "pairwise.complete.obs"))
}

lagged_cor <- function(x, y = x, lag_n = 1) {
  safe_cor(x, dplyr::lag(y, n = lag_n))
}

rms <- function(x) sqrt(mean(x^2, na.rm = TRUE))
energy <- function(x) mean(x^2, na.rm = TRUE)

spectral_features <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 4 || sd(x) == 0) {
    return(c(spec_mean = 0, spec_var = 0, spec_peak = 0))
  }
  spec <- spectrum(x, plot = FALSE, detrend = TRUE)
  w <- spec$spec / sum(spec$spec)
  mu <- sum(spec$freq * w)
  c(
    spec_mean = mu,
    spec_var = sum((spec$freq - mu)^2 * w),
    spec_peak = spec$freq[which.max(spec$spec)]
  )
}

# -----------------------------------------------------------------------------
# Extract features from one accelerometer or gyroscope file
# -----------------------------------------------------------------------------
extract_features <- function(filename, sensor, labels = NULL) {
  user_id <- as.integer(gsub(".+user(\\d+).+", "\\1", filename))
  exp_id <- as.integer(gsub(".+exp(\\d+).+", "\\1", filename))

  signal <- read_delim(
    filename, delim = " ", col_names = c("X1", "X2", "X3"),
    col_types = "ddd", show_col_types = FALSE, progress = FALSE
  ) %>%
    mutate(userid = user_id, trial = exp_id, sampleid = 0:(n() - 1))

  if (!is.null(labels)) {
    signal <- signal %>%
      left_join(labels, by = c("userid", "trial", "sampleid"))
  } else {
    signal <- signal %>% mutate(activity = NA_character_)
  }

  out <- signal %>%
    mutate(epoch = sampleid %/% 128) %>%
    group_by(epoch) %>%
    summarise(
      user_id = user_id,
      exp_id = exp_id,
      sampleid = first(sampleid),
      activity = mode_value(activity),
      m1 = mean(X1), m2 = mean(X2), m3 = mean(X3),
      mdif1_2 = m1 - m2, mdif1_3 = m1 - m3, mdif2_3 = m2 - m3,
      min1 = min(X1), min2 = min(X2), min3 = min(X3),
      max1 = max(X1), max2 = max(X2), max3 = max(X3),
      sd1 = sd(X1), sd2 = sd(X2), sd3 = sd(X3),
      median1 = median(X1), median2 = median(X2), median3 = median(X3),
      range1 = max1 - min1, range2 = max2 - min2, range3 = max3 - min3,
      q1_25 = quantile(X1, .25), q2_25 = quantile(X2, .25), q3_25 = quantile(X3, .25),
      q1_50 = quantile(X1, .50), q2_50 = quantile(X2, .50), q3_50 = quantile(X3, .50),
      q1_75 = quantile(X1, .75), q2_75 = quantile(X2, .75), q3_75 = quantile(X3, .75),
      AR1_1 = lagged_cor(X1, X1, 1), AR1_2 = lagged_cor(X1, X1, 2),
      AR2_1 = lagged_cor(X2, X2, 1), AR2_2 = lagged_cor(X2, X2, 2),
      AR3_1 = lagged_cor(X3, X3, 1), AR3_2 = lagged_cor(X3, X3, 2),
      AR12_1 = lagged_cor(X1, X2, 1), AR12_2 = lagged_cor(X1, X2, 2),
      AR13_1 = lagged_cor(X1, X3, 1), AR13_2 = lagged_cor(X1, X3, 2),
      AR23_1 = lagged_cor(X2, X3, 1), AR23_2 = lagged_cor(X2, X3, 2),
      skew1 = e1071::skewness(X1), skew2 = e1071::skewness(X2), skew3 = e1071::skewness(X3),
      kurt1 = e1071::kurtosis(X1), kurt2 = e1071::kurtosis(X2), kurt3 = e1071::kurtosis(X3),
      energy1 = energy(X1), energy2 = energy(X2), energy3 = energy(X3),
      rms1 = rms(X1), rms2 = rms(X2), rms3 = rms(X3),
      spec_mean1 = spectral_features(X1)["spec_mean"],
      spec_mean2 = spectral_features(X2)["spec_mean"],
      spec_mean3 = spectral_features(X3)["spec_mean"],
      spec_var1 = spectral_features(X1)["spec_var"],
      spec_var2 = spectral_features(X2)["spec_var"],
      spec_var3 = spectral_features(X3)["spec_var"],
      spec_peak1 = spectral_features(X1)["spec_peak"],
      spec_peak2 = spectral_features(X2)["spec_peak"],
      spec_peak3 = spectral_features(X3)["spec_peak"],
      n_samples = n(),
      .groups = "drop"
    )

  id_cols <- c("epoch", "user_id", "exp_id", "sampleid", "activity", "n_samples")
  feature_cols <- setdiff(names(out), id_cols)
  out %>% rename_with(~ paste0(.x, "_", sensor), all_of(feature_cols))
}

# -----------------------------------------------------------------------------
# Build training features
# -----------------------------------------------------------------------------
train_acc <- map_dfr(
  list.files(train_dir, "^acc_.*\\.txt$", full.names = TRUE),
  extract_features, sensor = "acc", labels = sample_labels
)
train_gyro <- map_dfr(
  list.files(train_dir, "^gyro_.*\\.txt$", full.names = TRUE),
  extract_features, sensor = "gyro", labels = sample_labels
)

train_data <- train_acc %>%
  select(-activity, -n_samples) %>%
  left_join(train_gyro, by = c("epoch", "user_id", "exp_id", "sampleid")) %>%
  filter(n_samples == 128, !is.na(activity))

metadata_cols <- c("epoch", "user_id", "exp_id", "sampleid", "n_samples", "activity")
predictor_cols <- setdiff(names(train_data), metadata_cols)

# Remove non-finite values and incomplete rows.
train_data[predictor_cols] <- lapply(train_data[predictor_cols], function(x) {
  x[!is.finite(x)] <- NA
  x
})
train_data <- train_data %>% drop_na(all_of(c("activity", predictor_cols)))

# Remove near-zero-variance and highly correlated predictors.
nzv <- nearZeroVar(train_data[predictor_cols])
if (length(nzv) > 0) predictor_cols <- predictor_cols[-nzv]

R <- cor(train_data[predictor_cols], use = "pairwise.complete.obs")
high_cor <- findCorrelation(R, cutoff = 0.80)
if (length(high_cor) > 0) predictor_cols <- predictor_cols[-high_cor]

model_data <- train_data %>%
  select(activity, all_of(predictor_cols)) %>%
  mutate(activity = factor(activity))

# -----------------------------------------------------------------------------
# Compare classifiers using 10-fold cross-validation
# -----------------------------------------------------------------------------
ctrl <- trainControl(method = "cv", number = 10)

fit_multinom <- train(
  activity ~ ., data = model_data, method = "multinom",
  trControl = ctrl, MaxNWts = 5000, trace = FALSE
)
fit_lda <- train(activity ~ ., data = model_data, method = "lda", trControl = ctrl)
fit_knn <- train(activity ~ ., data = model_data, method = "knn", trControl = ctrl)
fit_knn_scaled <- train(
  activity ~ ., data = model_data, method = "knn", trControl = ctrl,
  preProcess = c("center", "scale"), metric = "Accuracy"
)

models <- list(
  multinomial = fit_multinom,
  lda = fit_lda,
  knn = fit_knn,
  knn_scaled = fit_knn_scaled
)

comparison <- tibble(
  model = names(models),
  best_cv_accuracy = vapply(models, function(m) max(m$results$Accuracy), numeric(1))
) %>% arrange(desc(best_cv_accuracy))
write_csv(comparison, file.path(results_dir, "rerun_model_comparison.csv"))
print(comparison)

best_model <- models[[comparison$model[1]]]

# -----------------------------------------------------------------------------
# Build test features and create predictions
# -----------------------------------------------------------------------------
test_acc <- map_dfr(
  list.files(test_dir, "^acc_.*\\.txt$", full.names = TRUE),
  extract_features, sensor = "acc", labels = NULL
)
test_gyro <- map_dfr(
  list.files(test_dir, "^gyro_.*\\.txt$", full.names = TRUE),
  extract_features, sensor = "gyro", labels = NULL
)

test_data <- test_acc %>%
  select(-activity, -n_samples) %>%
  left_join(test_gyro %>% select(-activity), by = c("epoch", "user_id", "exp_id", "sampleid")) %>%
  filter(n_samples == 128)

test_data[predictor_cols] <- lapply(test_data[predictor_cols], function(x) {
  x[!is.finite(x)] <- NA
  x
})
if (anyNA(test_data[predictor_cols])) {
  stop("Missing values remain in test predictors.")
}

predictions <- predict(best_model, newdata = test_data[predictor_cols])
submission <- test_data %>%
  transmute(
    Id = sprintf("user%02d_exp%02d_%d", user_id, exp_id, sampleid),
    Predicted = predictions
  )
write_csv(submission, file.path(results_dir, "submission_final.csv"))
