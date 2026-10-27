## Reproducible model build for the weight-lifting exercise project.
## Run from this project directory with: Rscript analysis.R
.libPaths(c("Rlib", .libPaths()))
library(randomForest)
set.seed(20261027)

train <- read.csv("data/pml-training.csv", na.strings = c("NA", ""),
                  stringsAsFactors = FALSE)
quiz <- read.csv("data/pml-testing.csv", na.strings = c("NA", ""),
                 stringsAsFactors = FALSE)
train$classe <- factor(train$classe, levels = c("A", "B", "C", "D", "E"))

## Drop row identifiers, participant/timing/window metadata, and weakly observed
## features. Keep numeric sensor measurements with <=10% missing values.
excluded <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2",
              "cvtd_timestamp", "new_window", "num_window", "classe")
candidates <- setdiff(names(train), excluded)
numeric_features <- candidates[vapply(train[candidates], is.numeric, logical(1))]
missing_rate <- vapply(train[numeric_features], function(x) mean(is.na(x)), numeric(1))
features <- names(missing_rate)[missing_rate <= 0.10]
features <- features[vapply(train[features], function(x) length(unique(x[!is.na(x)])) > 1, logical(1))]
if (length(features) == 0) stop("No usable sensor features were selected")

## Stratified 5-fold cross-validation: each fold holds out about one fifth of
## each classe. The score is the pooled out-of-fold misclassification rate.
k <- 5
fold_id <- integer(nrow(train))
for (cl in levels(train$classe)) {
  rows <- which(train$classe == cl)
  fold_id[rows] <- sample(rep(seq_len(k), length.out = length(rows)))
}
fold_scores <- data.frame(fold = seq_len(k), n = NA_integer_, correct = NA_integer_,
                          accuracy = NA_real_, error = NA_real_)
oof <- factor(rep(NA_character_, nrow(train)), levels = levels(train$classe))
for (f in seq_len(k)) {
  fit_rows <- which(fold_id != f)
  valid_rows <- which(fold_id == f)
  fit <- randomForest(x = train[fit_rows, features, drop = FALSE],
                      y = train$classe[fit_rows], ntree = 250,
                      mtry = max(1, floor(sqrt(length(features)))),
                      importance = TRUE)
  oof[valid_rows] <- predict(fit, train[valid_rows, features, drop = FALSE])
  fold_scores$n[f] <- length(valid_rows)
  fold_scores$correct[f] <- sum(oof[valid_rows] == train$classe[valid_rows])
  fold_scores$accuracy[f] <- fold_scores$correct[f] / fold_scores$n[f]
  fold_scores$error[f] <- 1 - fold_scores$accuracy[f]
  cat(sprintf("Fold %d/%d complete: accuracy %.4f\n", f, k, fold_scores$accuracy[f]))
}
cv_accuracy <- mean(oof == train$classe)
cv_error <- 1 - cv_accuracy
cv_confusion <- table(actual = train$classe, predicted = oof)

## Fit the deployed classifier on all labeled rows and predict the 20 quiz rows.
set.seed(20261027)
final_model <- randomForest(x = train[, features, drop = FALSE], y = train$classe,
                            ntree = 500, mtry = max(1, floor(sqrt(length(features)))),
                            importance = TRUE)
predictions <- predict(final_model, quiz[, features, drop = FALSE])
quiz_output <- data.frame(problem_id = seq_along(predictions),
                          prediction = as.character(predictions))
write.csv(quiz_output, "quiz_predictions.csv", row.names = FALSE)
writeLines(as.character(predictions), "quiz_predictions.txt", useBytes = TRUE)
saveRDS(list(model = final_model, features = features, fold_scores = fold_scores,
             cv_accuracy = cv_accuracy, cv_error = cv_error,
             cv_confusion = cv_confusion, oof_predictions = oof,
             class_counts = table(train$classe), importance = importance(final_model),
             ntree = 500, mtry = max(1, floor(sqrt(length(features)))), seed = 20261027),
        "model_results.rds")
cat(sprintf("\nFeatures: %d | CV accuracy: %.4f | expected error: %.4f\n",
            length(features), cv_accuracy, cv_error))
cat("Confusion matrix (out-of-fold):\n"); print(cv_confusion)
cat("Quiz predictions:\n"); print(quiz_output)

