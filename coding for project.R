# Load necessary libraries
library(ggplot2)
library(reshape2)

# Set seed for reproducibility
set.seed(123)

# Generate data
X <- matrix(rnorm(600), nrow = 200) # matrix of predictors
X <- cbind(rep(1, 200), X) # add a column of ones for the intercept
beta <- c(-1, 4, -5, 2) # true coefficient values
linear_pred <- X %*% beta
prob <- 1 / (1 + exp(-linear_pred)) # convert to probabilities
y <- rbinom(200, size = 1, prob = prob) # generate binary outcomes

# Scale the predictors (excluding the intercept)
X_scaled <- scale(X[, -1])
X_scaled <- cbind(X[, 1], X_scaled) # add the intercept back

# Logistic Regression Function with Bootstrap Confidence Intervals
logistic_regression_with_bootstrap <- function(X, y, alpha = 0.05, n_bootstrap = 20) {
  logistic_loss <- function(beta, X, y) {
    p <- 1 / (1 + exp(-X %*% beta))
    -sum(y * log(p) + (1 - y) * log(1 - p))
  }
  
  init_beta <- rep(0, ncol(X))
  optim_result <- optim(par = init_beta, fn = logistic_loss, X = X, y = y, method = "Nelder-Mead", control = list(maxit = 1000))
  estimated_beta <- optim_result$par
  
  bootstrap_betas <- matrix(NA, nrow = n_bootstrap, ncol = length(estimated_beta))
  set.seed(123) # For reproducibility
  for (i in 1:n_bootstrap) {
    indices <- sample(1:nrow(X), replace = TRUE)
    X_boot <- X[indices, ]
    y_boot <- y[indices]
    optim_result_boot <- optim(par = init_beta, fn = logistic_loss, X = X_boot, y = y_boot, method = "Nelder-Mead", control = list(maxit = 1000))
    bootstrap_betas[i, ] <- optim_result_boot$par
  }
  
  ci_lower <- apply(bootstrap_betas, 2, function(beta) quantile(beta, alpha/2))
  ci_upper <- apply(bootstrap_betas, 2, function(beta) quantile(beta, 1 - alpha/2))
  
  list(
    estimated_beta = estimated_beta,
    bootstrap_betas = bootstrap_betas,
    confidence_intervals = list(lower = ci_lower, upper = ci_upper)
  )
}

# Run the Logistic Regression with Bootstrap
result <- logistic_regression_with_bootstrap(X_scaled, y, alpha = 0.05, n_bootstrap = 20)

# Predicted Probabilities
predict_logistic <- function(X, beta) {
  1 / (1 + exp(-X %*% beta))
}
predicted_probs <- predict_logistic(X_scaled, result$estimated_beta)

# Plotting the Fitted Logistic Curve
plot_data <- data.frame(
  Response = y,
  FittedProbability = predicted_probs,
  LinearPredictor = X_scaled %*% result$estimated_beta
)

ggplot(plot_data, aes(x = LinearPredictor, y = FittedProbability)) +
  geom_point(aes(color = as.factor(Response)), alpha = 0.6) +
  geom_line(aes(group = 1), color = 'blue') +
  scale_color_manual(values = c('red', 'green'), labels = c('0', '1')) +
  labs(title = 'Fitted Logistic Curve', x = expression(X %*% hat(beta)), y = 'Predicted Probability') +
  theme_minimal()

# Metrics Across Cutoffs
calculate_metrics <- function(conf_matrix) {
  TP <- conf_matrix[2, 2]
  FP <- conf_matrix[2, 1]
  TN <- conf_matrix[1, 1]
  FN <- conf_matrix[1, 2]
  
  Accuracy <- (TP + TN) / sum(conf_matrix)
  Sensitivity <- TP / (TP + FN)
  Specificity <- TN / (TN + FP)
  Prevalence <- (TP + FN) / sum(conf_matrix)
  FDR <- FP / (FP + TP)
  DOR <- ifelse(FP == 0 | FN == 0, NA, (TP / FP) / (FN / TN))
  
  return(c(Accuracy, Sensitivity, Specificity, Prevalence, FDR, DOR))
}

evaluate_metrics_across_cutoffs <- function(X, y, predicted_probs) {
  cutoffs <- seq(0.1, 0.9, by = 0.1)
  metrics_list <- list()
  
  for (cutoff in cutoffs) {
    predicted_classes <- ifelse(predicted_probs > cutoff, 1, 0)
    conf_matrix <- table(Predicted = predicted_classes, Actual = y)
    metrics_list[[as.character(cutoff)]] <- calculate_metrics(conf_matrix)
  }
  
  return(metrics_list)
}

metrics_across_cutoffs <- evaluate_metrics_across_cutoffs(X_scaled, y, predicted_probs)

# Convert list to data frame for plotting
# Convert the list of metrics into a data frame
metrics_df <- do.call(rbind, metrics_across_cutoffs)
metrics_df <- as.data.frame(metrics_df)
colnames(metrics_df) <- c("Accuracy", "Sensitivity", "Specificity", "Prevalence", "FDR", "DOR")
metrics_df$Cutoff <- as.numeric(rownames(metrics_df))

# Reshaping the data frame for ggplot
library(tidyr)
metrics_long <- pivot_longer(metrics_df, cols = -Cutoff, names_to = "Metric", values_to = "Value")

# Plotting the Metrics Across Cutoffs
ggplot(metrics_long, aes(x = Cutoff, y = Value, color = Metric)) +
  geom_line() +
  labs(title = "Metrics Evaluated Over Different Cutoffs",
       x = "Cutoff Value",
       y = "Metric Value") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1") +
  guides(color = guide_legend(title = "Metrics"))

# Ref: https://chat.openai.com/share/6282c9b6-5ee4-4cbc-8f6e-08a5c9d01034
  