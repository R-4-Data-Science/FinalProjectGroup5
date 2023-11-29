# Logistic probability function
logistic_probability <- function(beta, X) {
  1 / (1 + exp(-X %*% beta))
}

# Loss function for logistic regression
loss_function <- function(beta, X, y) {
  p <- logistic_probability(beta, X)
  -sum(y * log(p) + (1 - y) * log(1 - p))
}

#' @title Optimization Function Generating Initial Beta Values
#'
#' @description This function computes the initial values for optimization
#' obtained from the least squares formula \code{(X^(T)X)^(-1)X^(T)y}.
#' @param X An \code{nxr matrix} of data/observations. There are \code{n} observations
#' and \code{r} variables.
#' @param y A \code{vector} containing \code{n} binary responses.
#' @return A \code{vector} returning estimated values for each \code{beta}.
#' @author Ethan Sax, Nozara Sundus, Mohammad Maydanchi
#' @export
#' @examples
#'#Generate predictors
#'n <- 2000 # Number of observations
#'p <- 3   # Number of predictors
#'X <- matrix(rnorm(n * p), nrow = n)
#'#Add an intercept
#'X <- cbind(rep(1, n), X)
#'#Define true coefficients (including intercept)
#'beta_true <- c(-1, 2, -3, 1)
#'#Compute logistic probabilities
#'logit_probs <- logistic_probability(beta_true, X)
#'#Generate binary response variable
#'y <- rbinom(n, size = 1, prob = logit_probs)
#'#Logistic regression to estimate coefficients
#'estimated_beta <- logistic_regression(X, y)
#'print(estimated_beta)
logistic_regression <- function(X, y) {

  # Initial estimate using least squares
  initial_beta <- solve(t(X) %*% X) %*% t(X) %*% y

  # Optimization
  optim_result <- optim(initial_beta, fn = loss_function, X = X, y = y, method = "BFGS")
  return(optim_result$par)
}

#' @title Bootstrap Confidence Interval Generator
#'
#' @description This function generates a \code{matrix} of confidence intervals
#' for each predicted \code{beta} value.
#' @param X An \code{nxr matrix} of data/observations. There are \code{n} observations
#' and \code{r} variables.
#' @param y A \code{vector} containing \code{n} binary responses.
#' @param alpha A \code{numeric} determining the significance level \code{alpha}
#' to obtain for the \code{1-alpha} confidence intervals. The default value is
#' 0.05, which produces a 95 percent confidence interval.
#' @param n_bootstrap A \code{numeric} determining the number of bootstraps
#' for the function to carry out. The default value is 20 bootstrap iterations.
#' @return A \code{matrix} with upper and lower confidence levels for each \code{beta}.
#' @author Ethan Sax, Nozara Sundus, Mohammad Maydanchi
#' @importFrom stats coef
#' @importFrom stats quantile
#' @export
#' @examples
#'#Generate predictors
#'n <- 2000 # Number of observations
#'p <- 3   # Number of predictors
#'X <- matrix(rnorm(n * p), nrow = n)
#'#Add an intercept
#'X <- cbind(rep(1, n), X)
#'#Define true coefficients (including intercept)
#'beta_true <- c(-1, 2, -3, 1)
#'#Compute logistic probabilities
#'logit_probs <- logistic_probability(beta_true, X)
#'#Generate binary response variable
#'y <- rbinom(n, size = 1, prob = logit_probs)
#'#Calculate bootstrap confidence intervals
#'CI <- bootstrap_CI(X, y, alpha = 0.05, n_bootstrap = 20)
#'print(CI)
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

#' @title Plotting the Fitted Logistic Regression Curve
#'
#' @description This function plots the fitted logistic regression curve
#' to the binary responses \code{y}. The \code{x-axis} represents a sequence of values.
#' @param X An \code{nxr matrix} of data/observations. There are \code{n} observations
#' and \code{r} variables.
#' @param y A \code{vector} containing\code{n} binary responses.
#' @param beta_hat A \code{vector} containing the estimated \code{beta} values
#' generated from the \code{logistic_regression} optimization function.
#' @return A \code{plot} of the fitted logistic regression curve.
#' @author Ethan Sax, Nozara Sundus, Mohammad Maydanchi
#' @importFrom ggplot2 ggplot
#' @importFrom ggplot2 aes
#' @importFrom ggplot2 geom_smooth
#' @importFrom ggplot2 geom_point
#' @importFrom ggplot2 labs
#' @export
#' @examples
#'#Generate predictors
#'n <- 2000 # Number of observations
#'p <- 3   # Number of predictors
#'X <- matrix(rnorm(n * p), nrow = n)
#'#Add an intercept
#'X <- cbind(rep(1, n), X)
#'#Define true coefficients (including intercept)
#'beta_true <- c(-1, 2, -3, 1)
#'#Compute logistic probabilities
#'logit_probs <- logistic_probability(beta_true, X)
#'#Generate binary response variable
#'y <- rbinom(n, size = 1, prob = logit_probs)
#'#Logistic regression to estimate coefficients
#'estimated_beta <- logistic_regression(X, y)
#'#Plotting the curve
#'plot_logistic_curve(X, y, estimated_beta)
plot_logistic_curve <- function(X, y, beta_hat) {
  df <- data.frame(X = X[, 2], y = y) # Assuming second column of X is the predictor
  ggplot(df, aes(x = X, y = y)) +
    geom_point() +
    stat_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE, color = "blue") +
    labs(title = "Logistic Regression Curve", x = "Predictor", y = "Response")
}

#' @title Function to Create the Confusion Matrix for the Data
#'
#' @description This function generates the resulting "Confusion Matrix" using
#' a cut-off value for prediction at 0.5. Along with the matrix, it also computes
#' the \code{prevalence}, \code{accuracy}, \code{sensitivity}, \code{specificity},
#' \code{false discovery rate}, and \code{diagnostic odds ratio} metrics.
#' @param actual A \code{vector} containing \code{n} binary responses.
#' @param predicted_probabilities A \code{vector} containing predicted
#' responses using the estimated \code{beta} coefficients.
#' @param cutoff A \code{numeric} determining the cut-off value for prediction.
#' The default value is 0.5, where predictions above 0.5 are assigned a value of
#' 1 and predictions below or equal to 0.5 are assigned a 0. The cut-off value
#' must lie within the range of (0,1).
#' @return A \code{list} containing the confusion matrix along with each
#' specified metric in the description.
#' @author Ethan Sax, Nozara Sundus, Mohammad Maydanchi
#' @export
#' @examples
#'#Generate predictors
#'n <- 2000 # Number of observations
#'p <- 3   # Number of predictors
#'X <- matrix(rnorm(n * p), nrow = n)
#'#Add an intercept
#'X <- cbind(rep(1, n), X)
#'#Define true coefficients (including intercept)
#'beta_true <- c(-1, 2, -3, 1)
#'#Compute logistic probabilities
#'logit_probs <- logistic_probability(beta_true, X)
#'#Generate binary response variable
#'y <- rbinom(n, size = 1, prob = logit_probs)
#'#Logistic regression to estimate coefficients
#'estimated_beta <- logistic_regression(X, y)
#'#Calculate the predicted probabilities using the estimated coefficients
#'predicted_prob <- logistic_probability(estimated_beta, X)
#'#Calculate metrics using a cutoff of 0.5
#'metrics <- calculate_metrics(y, predicted_prob, cutoff = 0.5)
#'print(metrics)
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

#' @title Confusion Matrix Metrics Plot
#'
#' @description This function gives the user the ability to plot any of
#' metrics computed from the confusion matrix over a grid of cut-off values
#' for prediction going from 0.1 to 0.9 with steps of 0.1.
#' @param y A \code{vector} containing \code{n} binary responses.
#' @param predicted_probabilities A \code{vector} containing predicted
#' responses using the estimated \code{beta} coefficients.
#' @param plot_prevalence A \code{boolean} determining whether the prevalence
#' metric is plotted or not. It has a default value of FALSE.
#' @param plot_accuracy A \code{numeric} determining whether the accuracy metric
#' is plotted or not. It has a default value of TRUE.
#' @param plot_sensitivity A \code{numeric} determining whether the sensitivity
#' metric is plotted or not. It has a default value of TRUE.
#' @param plot_specificity A \code{numeric} determining whether the specificity
#' metric is plotted or not. It has a default value of TRUE.
#' @param plot_fdr A \code{boolean} determining whether the false discovery rate
#' metric is plotted or not. It has a default value of FALSE.
#' @return A \code{plot} of the selected metrics computed from
#' the confusion matrix
#' @author Ethan Sax, Nozara Sundus, Mohammad Maydanchi
#' @importFrom reshape2 melt
#' @importFrom ggplot2 ggplot
#' @importFrom ggplot2 geom_path
#' @importFrom ggplot2 aes
#' @importFrom ggplot2 labs
#' @export
#' @examples
#'#Generate predictors
#'n <- 2000 # Number of observations
#'p <- 3   # Number of predictors
#'X <- matrix(rnorm(n * p), nrow = n)
#'#Add an intercept
#'X <- cbind(rep(1, n), X)
#'#Define true coefficients (including intercept)
#'beta_true <- c(-1, 2, -3, 1)
#'#Compute logistic probabilities
#'logit_probs <- logistic_probability(beta_true, X)
#'#Generate binary response variable
#'y <- rbinom(n, size = 1, prob = logit_probs)
#'#Logistic regression to estimate coefficients
#'estimated_beta <- logistic_regression(X, y)
#'#Calculate the predicted probabilities using the estimated coefficients
#'predicted_prob <- logistic_probability(estimated_beta, X)
#'#Plotting the metrics over various cutoff values with user's choice
#'plot_metrics_over_cutoff(y, predicted_prob, plot_prevalence = FALSE,
#' plot_accuracy = TRUE, plot_sensitivity = TRUE,
#' plot_specificity = TRUE, plot_fdr = FALSE)
plot_metrics_over_cutoff <- function(y, predicted_probabilities,
                                     plot_prevalence = FALSE, plot_accuracy = TRUE,
                                     plot_sensitivity = TRUE, plot_specificity = TRUE,
                                     plot_fdr = FALSE) {
  cutoffs <- seq(0.1, 0.9, by = 0.1)
  metrics_list <- lapply(cutoffs, function(cutoff) {
    calculate_metrics(y, predicted_probabilities, cutoff)
  })

  metrics_df <- do.call(rbind, lapply(1:length(metrics_list), function(i) {
    data.frame(
      Cutoff = cutoffs[i],
      Prevalence = metrics_list[[i]]$prevalence,
      Accuracy = metrics_list[[i]]$accuracy,
      Sensitivity = metrics_list[[i]]$sensitivity,
      Specificity = metrics_list[[i]]$specificity,
      FDR = metrics_list[[i]]$false_discovery_rate
    )
  }))

  # Melting the data for plotting with ggplot2
  metrics_melted <- melt(metrics_df, id.vars = 'Cutoff')

  # Filter based on user preference
  metrics_to_plot <- c('Cutoff')
  if (plot_prevalence) metrics_to_plot <- c(metrics_to_plot, 'Prevalence')
  if (plot_accuracy) metrics_to_plot <- c(metrics_to_plot, 'Accuracy')
  if (plot_sensitivity) metrics_to_plot <- c(metrics_to_plot, 'Sensitivity')
  if (plot_specificity) metrics_to_plot <- c(metrics_to_plot, 'Specificity')
  if (plot_fdr) metrics_to_plot <- c(metrics_to_plot, 'FDR')

  metrics_filtered <- metrics_melted[metrics_melted$variable %in% metrics_to_plot[-1], ]

  # Plotting
  ggplot(metrics_filtered, aes(x = Cutoff, y = value, colour = variable)) +
    geom_line() +
    labs(title = "Performance Metrics over Various Cutoffs", x = "Cutoff Value", y = "Metric Value") +
    theme_minimal()
}
