
# Logistic Regression Package

This package provides functions of Logistic Regression, which are :

1.  `bootstrap_CI()`
2.  `calculate_metrics()`
3.  `logistic_regression()`
4.  `plot_logistic_curve()`
5.  `plot_metrics_over_cutoff()`

Below, we introduce Logistic Regression, covering its purpose,
functionality, mathematics, applications, assumptions, and
interpretation.

# Logistic Regression Overview

Logistic Regression is a statistical method used for binary
classification problems. It predicts the probability of an outcome that
has two possible values, such as yes/no, win/lose, or alive/dead.

## Functionality and Purpose

- **Purpose**: Logistic regression is used for classification problems,
  specifically to predict binary outcomes.

- **Functionality**: It uses the logistic (or sigmoid) function to
  convert the output of a linear equation into a probability between 0
  and 1. This probability is the likelihood of the dependent variable
  being a ‘1’ or ‘success’.

## Mathematical Formula

The logistic function in logistic regression is defined as:

$$ P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X)}} $$

Where: $P(Y=1)$ is the probability of the dependent variable equaling
‘1’. $e$ is the base of natural logarithm. $\beta_0$ and $\beta_1$ are
the coefficients of the model.

## Applications

Logistic regression is widely used in fields like medicine, social
sciences, and machine learning for solving binary classification
problems.

## Assumptions

The key assumptions of logistic regression include: - A linear
relationship between the log odds of the dependent variable and the
independent variables. - No multicollinearity among independent
variables. - Independent variables are measured accurately and without
error.

## Interpretation of Results

In logistic regression, the coefficients represent the change in the log
odds of the dependent variable for a one-unit change in the independent
variable, while other variables are held constant.

This information was created with the assistance of this
[link](https://chat.openai.com/share/a531ba79-9803-4e75-ae64-a4c2ca1eaa7f).
