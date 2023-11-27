#Load libraries
library(ggplot2)
library(reshape2)

# Define the logistic regression function
logistic_regression <- function(X, y) {
  # Logistic function
  logistic_function <- function(beta, X) {
    1 / (1 + exp(-X %*% beta))
  }
  
  # Loss function for logistic regression
  loss_function <- function(beta, X, y) {
    p <- logistic_function(beta, X)
    -sum(y * log(p) + (1 - y) * log(1 - p))
  }
  
  # Initial estimate using least squares
  initial_beta <- solve(t(X) %*% X) %*% t(X) %*% y
  
  # Optimization
  optim_result <- optim(initial_beta, fn = loss_function, X = X, y = y, method = "BFGS")
  return(optim_result$par)
}

# Define the bootstrap confidence interval function
bootstrap_CI <- function(X, y, alpha = 0.05, n_bootstrap = 20) {
  # Store bootstrap samples of coefficients
  boot_coefs <- matrix(NA, nrow = n_bootstrap, ncol = ncol(X))
  
  for (i in 1:n_bootstrap) {
    # Sample with replacement
    sample_indices <- sample(1:nrow(X), replace = TRUE)
    X_boot <- X[sample_indices, ]
    y_boot <- y[sample_indices]
    
    # Estimate coefficients for this bootstrap sample
    boot_coefs[i, ] <- logistic_regression(X_boot, y_boot)
  }
  
  # Calculate confidence intervals
  CI_lower <- apply(boot_coefs, 2, function(coef) quantile(coef, alpha / 2))
  CI_upper <- apply(boot_coefs, 2, function(coef) quantile(coef, 1 - alpha / 2))
  
  # Return a matrix with lower and upper bounds
  CI <- rbind(CI_lower, CI_upper)
  rownames(CI) <- c("Lower", "Upper")
  return(CI)
}

# Function to plot the fitted logistic curve
plot_logistic_curve <- function(X, y, beta_hat) {
  df <- data.frame(X = X[, 2], y = y) # Assuming second column of X is the predictor
  ggplot(df, aes(x = X, y = y)) +
    geom_point() +
    stat_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE, color = "blue") +
    labs(title = "Logistic Regression Curve", x = "Predictor", y = "Response")
}

# Function to calculate confusion matrix and related metrics
calculate_metrics <- function(actual, predicted_probabilities, cutoff = 0.5) {
  predicted <- ifelse(predicted_probabilities > cutoff, 1, 0)
  TP <- sum((predicted == 1) & (actual == 1))
  TN <- sum((predicted == 0) & (actual == 0))
  FP <- sum((predicted == 1) & (actual == 0))
  FN <- sum((predicted == 0) & (actual == 1))
  
  prevalence <- mean(actual)
  accuracy <- (TP + TN) / (TP + FP + FN + TN)
  sensitivity <- TP / (TP + FN) # Also known as recall or true positive rate
  specificity <- TN / (TN + FP) # True negative rate
  false_discovery_rate <- FP / (FP + TP)
  diagnostic_odds_ratio <- (TP / FP) / (FN / TN)
  
  metrics <- list(
    confusion_matrix = matrix(c(TP, FN, FP, TN), nrow = 2, byrow = TRUE, dimnames = list("Actual Positive" = c("Predicted Positive", "Predicted Negative"), "Actual Negative" = c("Predicted Positive", "Predicted Negative"))),
    prevalence = prevalence,
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    false_discovery_rate = false_discovery_rate,
    diagnostic_odds_ratio = diagnostic_odds_ratio
  )
  
  return(metrics)
}

# Function to plot metrics over various cutoff values
plot_metrics_over_cutoff <- function(y, predicted_probabilities) {
  cutoffs <- seq(0.1, 0.9, by = 0.1)
  metrics_data <- sapply(cutoffs, function(cutoff) {
    metrics <- calculate_metrics(y, predicted_probabilities, cutoff)
    c(metrics$accuracy, metrics$sensitivity, metrics$specificity)
  })
  
  # Convert the metrics data to a data frame for plotting
  metrics_df <- as.data.frame(t(metrics_data))
  names(metrics_df) <- c("Accuracy", "Sensitivity", "Specificity")
  metrics_df$Cutoff <- cutoffs
  
  # Plot using ggplot2
  long_metrics_df <- reshape2::melt(metrics_df, id.vars = "Cutoff")
  ggplot(long_metrics_df, aes(x = Cutoff, y = value, colour = variable)) +
    geom_line() +
    labs(title = "Performance Metrics over Various Cutoffs", x = "Cutoff Value", y = "Metric Value") +
    theme_minimal()
}



#Example

set.seed(123) # For reproducibility

# Generate predictors
n <- 2000 # Number of observations
p <- 3   # Number of predictors
X <- matrix(rnorm(n * p), nrow = n)

# Add an intercept
X <- cbind(rep(1, n), X)

# Define true coefficients (including intercept)
beta_true <- c(-1, 2, -3, 1)

# Compute logistic probabilities
logit_probs <- 1 / (1 + exp(-X %*% beta_true))

# Generate binary response variable
y <- rbinom(n, size = 1, prob = logit_probs)

# Check the first few rows of X and y
head(X)
head(y)

# Logistic regression to estimate coefficients
estimated_beta <- logistic_regression(X, y)
print(estimated_beta)
# Calculate bootstrap confidence intervals
CI <- bootstrap_CI(X, y, alpha = 0.05, n_bootstrap = 20)
print(CI)
# Plotting the curve
plot_logistic_curve(X, y, estimated_beta)

# Logistic function using the estimated coefficients
logistic_function <- function(beta, X) {
  1 / (1 + exp(-X %*% beta))
}

# Calculate the predicted probabilities using the estimated coefficients
predicted_prob <- logistic_function(estimated_beta, X)

# Calculate metrics using a cutoff of 0.5
metrics <- calculate_metrics(y, predicted_prob, cutoff = 0.5)
print(metrics$confusion_matrix)
print(paste("Prevalence:", metrics$prevalence))
print(paste("Accuracy:", metrics$accuracy))
print(paste("Sensitivity:", metrics$sensitivity))
print(paste("Specificity:", metrics$specificity))
print(paste("False Discovery Rate:", metrics$false_discovery_rate))
print(paste("Diagnostic Odds Ratio:", metrics$diagnostic_odds_ratio))

# Plotting the metrics over various cutoff values
plot_metrics_over_cutoff(y, predicted_prob)


# Calculate metrics using a cutoff of 0.5
metrics <- calculate_metrics(y, predicted_prob, cutoff = 0.5)
print(metrics)

#Ref: https://chat.openai.com/g/g-zZ3HN933F-r-code-helper/c/bbb01533-affe-48d5-8e44-6cb8138a4a50

