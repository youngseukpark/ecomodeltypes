##########################################################
# R script used in Chapter 11 MLP for ecological modeling
# Elsevier 2025. Ecological Model Types
# Authors: Young-Seuk Park
# Date: 2025.07.17
##########################################################

# 1. variable selection
# 2. Variable preprocessing
# 3. Model building and training
# 4. Result Visualization
# 5. Interpretability: Partial dependence plot (PDP)

# MLP
# Install and load the required package
install.packages("neuralnet") # Run only once to install packages
install.packages("car")  
install.packages("pdp")
install.packages("gridExtra")

library(neuralnet)
library(car)
library(pdp)
library(ggplot2)
library(gridExtra)

# Load the dataset
setwd("D:/Ecological Model Types/MLP/")
FSR = read.csv(file="FFishSREnv.csv")

# View the first few rows of the dataset
head(FSR)

###############################################################
# 1. Variable selection
###############################################################

# To select variables using VIF to reduce variable colinearity
# Fit an initial linear model
SRmodel <- lm(SR ~ ., data = FSR) # SR: dependent variable 

# Check VIF in car package
vif(SRmodel)

# Function to Iteratively Remove High VIF Variables
# This function does backward elimination based on VIF
vif_selection <- function(data, target, threshold = 5) {
  formula <- as.formula(paste(target, "~ ."))
  model <- lm(formula, data = data)
  vifs <- vif(model)
  
  while (any(vifs > threshold)) {
    max_vif_var <- names(which.max(vifs))
    message(sprintf("Removing '%s' with VIF = %.2f", max_vif_var, max(vifs)))
    
    data <- data[, !(names(data) %in% max_vif_var)]
    formula <- as.formula(paste(target, "~ ."))
    model <- lm(formula, data = data)
    vifs <- vif(model)
  }
  return(names(data))
}

# Conducting variable selection
# SR: your target (dependent) variable
# Remove predictors with VIF > 5. It can be changed with with different value of threshold
selected_vars <- vif_selection(data = FSR, target = "SR", threshold = 5) 
print(selected_vars)

###############################################################
# 2. Variable preprocessing
###############################################################

# scaling
# standardization
SFSR = as.data.frame(scale(FSR))

# MAX-MIN NORMALIZATION
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
MinMaxSR <- as.data.frame(lapply(FSR, normalize))


# Split the data into training and testing sets
set.seed(123)  # For reproducibility
index <- sample(1:nrow(MinMaxSR), 0.7 * nrow(MinMaxSR))
train_data <- MinMaxSR[index, ]
test_data <- MinMaxSR[-index, ]


###############################################################
# 3. Model building and training
###############################################################

# Define the formula for the model
formula <- SR ~ Elevation + DS + Urban + Agri 
#formula <- dependent_variable ~ independent variable 1+ variable 1....

# Train the MLP model
model <- neuralnet(formula, 
                   data = train_data, 
                   hidden = c(10,5), 
                   linear.output = FALSE,
                   lifesign = "full",     # Show full training log
                   lifesign.step = 1     # Print every iteration (optional)
                   )

# Make predictions on the test set
prediction1 <- predict(model, train_data)
prediction2 <- predict(model, test_data)

###############################################################
# 4. Result Visualization
###############################################################

########################################
# Plot prediction results: scatter plot
########################################

par(mfrow = c(2,2))
CORtr<-cor.test(train_data$SR, prediction1)
CORts<-cor.test(test_data$SR, prediction2)

r_tr <- round(CORtr$estimate, 3)
r_ts <- round(CORts$estimate, 3)

# training results
plot(train_data$SR, prediction1, 
     xlim = c(0, 1), 
     ylim = c(0, 1),
     main = "Train Results",
     xlab = "Actual SR", ylab = "Predicted SR"
)
abline(0, 1, col = "red", lty = 2)
# r °ª Ãß°¡
text(0.05, 0.95, paste("r =", r_tr), pos = 4)

# test results
plot(test_data$SR, prediction2, 
     xlim = c(0, 1), 
     ylim = c(0, 1),
     main = "Test Results",
     xlab = "Actual SR", ylab = "Predicted SR"
)
abline(0, 1, col = "red", lty = 2)
# r °ª Ãß°¡
text(0.05, 0.95, paste("r =", r_ts), pos = 4)


TrainedSR<-cbind(train_data$SR, prediction1)
TestedSR<-cbind(test_data$SR, prediction2)

###########################################
# plot model structure in a new open window
###########################################
plot(model)
model$result.matrix

###############################################################
# 5. Interpretability: Partial dependence plot (PDP)
###############################################################

# Create prediction wrapper for pdp
nn_predict <- function(object, newdata) {
  compute(object, newdata)$net.result
}

# Extract predictor variable names from formula
predictor_vars <- all.vars(formula)[-1]  # drop response var

# Generate PDPs for all predictors
pdp_list <- lapply(predictor_vars, function(varname) {
  pd <- partial(
    object = model,
    pred.var = varname,
    pred.fun = nn_predict,
    train =train_data,
    grid.resolution = 50
  )
  autoplot(pd) +
    ggtitle(paste("PDP for", varname)) +
    theme_minimal()
})

# Display all PDPs in a grid
do.call(grid.arrange, pdp_list)


# PDP with 95% CI calculation
#################
# Predictors
predictors <- c("Elevation", "DS", "Urban", "Agri")

# Calculate residual SD from training data for CI
train_preds <- compute(model, train_data[, predictors])$net.result
resid_sd <- sd(train_data$SR - train_preds)

pdp_with_ci <- function(var, model, data, n = 50) {
  # Ensure variable is numeric
  x <- data[[var]]
  if (!is.numeric(x)) x <- as.numeric(as.character(x))
  
  vals <- seq(min(x), max(x), length.out = n)
  others <- colMeans(data[, predictors, drop = FALSE])
  
  preds <- sapply(vals, function(v) {
    newdata <- as.data.frame(as.list(others))
    newdata[[var]] <- v
    compute(model, newdata)$net.result
  })
  
  data.frame(
    x = vals,
    y = as.vector(preds),
    upper = preds + 1.96 * resid_sd,
    lower = preds - 1.96 * resid_sd,
    var = var
  )
}

#################
# PDP with 95% CI for plot
#################
# Get PDPs for all variables
pdp_all <- do.call(rbind, lapply(predictors, pdp_with_ci, model = model, data = MinMaxSR))

# Plot with base R
par(mfrow = c(length(predictors), 1), mar = c(4, 4, 2, 1))
par(mfrow = c(2,2))

# Plot option 2: display with gray color for CI 
for (v in predictors) {
  sub <- subset(pdp_all, var == v)
  
  # Plot mean prediction line
  plot(sub$x, sub$y, type = "l", lwd = 2, col = "blue",
#       ylim = range(c(sub$lower, sub$upper)),
       ylim = range(c(0, 1)),
       xlab = paste(v, "(normalized)"), ylab = "Predicted SR",
       main = paste("PDP for", v, "with 95% CI"))
  
  # Draw gray confidence interval band
  polygon(c(sub$x, rev(sub$x)), c(sub$upper, rev(sub$lower)),
          col = rgb(0.8, 0.8, 0.8, 0.5), border = NA)
  
  # Re-draw prediction line on top
  lines(sub$x, sub$y, col = "blue", lwd = 2)
  lines(sub$x, sub$upper, col = "blue", lty = 3)
  lines(sub$x, sub$lower, col = "blue", lty = 3)
}


