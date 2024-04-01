library(dplyr)
library(tidyr)
library(tidyverse)
library(ggplot2)
library(rstan)
library(tidybayes)
library(modelr)
library(brms)
library(rstan)
library(bayesplot)
library(mcmcplots)
library(bridgesampling)
library(loo)

# Load dataset
#("~/Documents")
booking_data1 <- read.csv("D:/Germany/Study Files-TUD/TU Dortmund/--------Semester-8-Winter Term--------2023-2024/Applied Bayesian Data Analysis/R/booking.csv")

# Sample 5000 observations randomly
sampled_data <- booking_data1 %>%
  sample_n(5000)

# Convert Cancellation Status, start by converting the categorical "booking.status" variable into numeric format:
sampled_data$cancellation_status <- ifelse(sampled_data$booking.status == "Canceled", 1, 0)

# unique(sampled_data$cancellation_status)
print(sampled_data$cancellation_status)


head(sampled_data)
summary(sampled_data)
str(sampled_data)

# Model 1
## Setting up seed & the priors and formulate the Bayesian Logistic Regression formula
set.seed(456)

priors <- c(
  prior(normal(3.5, 1), class = "Intercept"),  # Adjusted mean and smaller scale
  prior(normal(0, 0.5), class = "b"),         # Smaller scale
  prior(normal(0, 0.5), class = "sd", coef = "sd")  # Smaller scale
)

lr_model <- brm(
  formula = booking.status ~
    number.of.adults + number.of.children + number.of.weekend.nights +
    number.of.week.nights + car.parking.space + lead.time + 
    P.C + P.not.C + average.price + special.requests + room.type, 
  data = sampled_data, 
  family = bernoulli("logit"),
  control = list(adapt_delta = 0.99, max_treedepth = 15),
  iter = 2000,
  save_pars = save_pars(all = TRUE)
)


summary(lr_model)
mcmc_trace(lr_model)
# Perform posterior predictive check
pp_check(lr_model)


# Model 2
# Formulate the Binomial Beta formula
# Fit the Exponential Gamma regression model
binomial_beta_model <- brm(
  formula = cancellation_status | trials(5000) ~ 
    number.of.adults + number.of.children + number.of.weekend.nights + 
    number.of.week.nights + car.parking.space + 
    room.type + lead.time  + P.C + P.not.C + average.price + special.requests,
  data = sampled_data,
  family = binomial(link = "logit"),
  prior = c(
    prior(normal(0, 5), class = "Intercept"),
    prior(normal(0, 2), class = "b")
  ),
  control = list(adapt_delta = 0.99, max_treedepth = 15),
  iter = 2000,
  save_pars = save_pars(all = TRUE)
)

# Print the summary of the model
summary(binomial_beta_model)
summary(binomial_beta_model)
mcmc_trace(binomial_beta_model)

# Perform posterior predictive check
pp_check(binomial_beta_model)



## Model Comparison
# Extract LOO-CV information for each model
loo_lr <- loo(lr_model)
loo_binomial_beta <- loo(binomial_beta_model)

# Compare models
loo_compare_result <- loo_compare(loo_lr, loo_binomial_beta)

# Print the comparison result
print(loo_compare_result)

## Posterior Predictive Check
# Perform posterior predictive check
pp_check(lr_model)
pp_check(binomial_beta_model)

### New out-of-sample data prediction
# Step 1: Create a new dataframe with hypothetical values for prediction
new_booking_data <- data.frame(
  number.of.adults = c(2, 1, 1),  # Example values
  number.of.children = c(0, 1, 0),
  number.of.weekend.nights = c(1, 2, 0),
  number.of.week.nights = c(2, 3, 1),
  car.parking.space = c(1, 0, 0),
  lead.time = c(10, 20, 15),
  P.C = c(0, 1, 0),
  P.not.C = c(1, 0, 1),
  average.price = c(100, 150, 100),
  special.requests = c(2, 1, 3),
  room.type = factor(c("Room_Type1", "Room_Type2", "Room_Type5"), 
                     levels = levels(sampled_data$room.type))
)

# Step 2: Make predictions using the trained model
predictions <- predict(lr_model, newdata = new_booking_data, type = "response")

# Add predictions to the new data for review
new_booking_data$cancellation_status <- predictions

# Print the new data with predictions
print(predictions)