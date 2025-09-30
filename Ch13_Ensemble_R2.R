###########################################
# R script used in Chapter 13 Ensemble models for ecological modeling
# Elsevier 2025. Ecological Model Types
# Authors: Young-Seuk Park & Sovan Lek
# Date: 2025.07.17
###########################################

library(rpart)

# Fit a classification tree
fit <- rpart(Species ~ ., data = iris, method = "class", cp = 0.01)

# Print the cost-complexity table
printcp(fit)

# Plot the cross-validated error vs cp
plotcp(fit)

install.packages("BiodiversityR")
library(BiodiversityR)
install.packages("vegan")
library(vegan)
data(CucurbitaClim)

library(BiodiversityR)
data(CucurbitaClim) 
df<- CucurbitaClim	# 2404 x 24
# sepate data (climate="baseline") to learn the models
learn.set<-subset(df, climate == "baseline", select = c("species","bioc1","bioc2","bioc3","bioc4","bioc5","bioc6","bioc7","bioc8","bioc9","bioc10","bioc11","bioc12","bioc13","bioc14","bioc15","bioc16","bioc17","bioc18","bioc19"))
X.learn<-subset(df, climate == "baseline", select = c("bioc1","bioc2","bioc3","bioc4","bioc5","bioc6","bioc7","bioc8","bioc9","bioc10","bioc11","bioc12","bioc13","bioc14","bioc15","bioc16","bioc17","bioc18","bioc19"))
Y.learn<-subset(df,climate=="baseline",select="species")
# future projection data to test the model (climate="future")
X.test<-subset(df, climate == "future", select = c("bioc1","bioc2","bioc3","bioc4","bioc5","bioc6","bioc7","bioc8","bioc9","bioc10","bioc11","bioc12","bioc13","bioc14","bioc15","bioc16","bioc17","bioc18","bioc19"))
Y.test<-subset(df, climate == "future", select ="species")
baseline<-learn.set
splitSample <- sample(1:2, size=nrow(baseline), prob=c(0.6,0.4), replace = TRUE)
learn.set <- baseline[splitSample==1,]		# 720 20
valid.set <- baseline[splitSample==2,]                     # 482 20

library(caret)
set.seed(62433)		# for reproducibility
modelFitRF <- train(species ~ ., data = learn.set, method = "rf")
modelFitGBM <- train(species ~ ., data = learn.set, method = "gbm",verbose=F)
modelFitLDA <- train(species ~ ., data = learn.set, method = "lda")
modelFitRpart <- train(species ~ ., data = learn.set, method = "rpart")
modelFitSVM <- train(species ~ ., data = learn.set, method = "svmRadial")
modelFitnnet <- train(species ~ ., data = learn.set, method = "nnet")
modelFitknn <- train(species ~ ., data = learn.set, method = "knn")

predRF <- predict(modelFitRF, newdata = valid.set)
predGBM <- predict(modelFitGBM, newdata = valid.set)
predLDA <- predict(modelFitLDA, newdata = valid.set)
predRpart <- predict(modelFitRpart, newdata = valid.set)
predSVM <- predict(modelFitSVM, newdata = valid.set)
prednnet <- predict(modelFitnnet, newdata = valid.set)
predknn <- predict(modelFitknn, newdata = valid.set)

confusionMatrix(predRF, valid.set$species)$overall[1]
confusionMatrix(predGBM, valid.set$species)$overall[1]
confusionMatrix(predLDA, valid.set$species)$overall[1]
confusionMatrix(predRpart, valid.set$species)$overall[1]
confusionMatrix(predSVM, valid.set$species)$overall[1]
confusionMatrix(prednnet, valid.set$species)$overall[1]
confusionMatrix(predknn, valid.set$species)$overall[1]

# ensemble Metalearner
predDF <- data.frame(predRF, predGBM, predLDA, predRpart, predSVM, prednnet, predknn, species = valid.set$species, stringsAsFactors = F)
# Train the ensemble
modelStack <- train(species ~ ., data = predDF, method = "rf")
confusionMatrix(predict(modelStack), valid.set$species)

# Generate predictions on the test set
testPredRF <- predict(modelFitRF, newdata = X.test)
testPredGBM <- predict(modelFitGBM, newdata = X.test)
testPredLDA <- predict(modelFitLDA, newdata = X.test)
testPredRpart <- predict(modelFitRpart, newdata = X.test)
testPredSVM <- predict(modelFitSVM, newdata = X.test)
testPrednnet <- predict(modelFitnnet, newdata = X.test)
testPredknn <- predict(modelFitknn, newdata = X.test)

confusionMatrix(testPredRF, Y.test$species)$overall[1]
confusionMatrix(testPredGBM, Y.test$species)$overall[1]
confusionMatrix(testPredLDA, Y.test$species)$overall[1]
confusionMatrix(testPredRpart, Y.test$species)$overall[1]
confusionMatrix(testPredSVM, Y.test$species)$overall[1]
confusionMatrix(testPredRF, Y.test$species)$overall[1]
confusionMatrix(testPredknn, Y.test$species)$overall[1]
confusionMatrix(testPrednnet, Y.test$species)$overall[1]
length(testPredRF)
# Using the base learner test set predictions, 
# create the level-one dataset to feed to the ensemble
testPredLevelOne <- data.frame(testPredRF, testPredGBM, testPredLDA, testPredRpart, testPredSVM, testPredknn, testPrednnet, species = Y.test$species, stringsAsFactors = F)
testStack <- train(species ~ ., data = testPredLevelOne, method = "rf")
# Test performance
confusionMatrix(predict(testStack), Y.test$species)


install.packages("agricolae")
library(agricolae)
data(yacon) # 432 x 13Q + 1Q
scale_values <- function(x){(x-min(x))/(max(x)-min(x))}
yacon$IH <- scale_values(yacon$IH)
# split dat to select 75% for model learning and 25% for model testing
split<-createDataPartition(yacon$IH, p = .75, list = FALSE)
trainData = yacon[split,c(6:19)]
testData = yacon[-split,c(6:19)]

set.seed(7)
fit.glm <- train(IH ~ ., data = trainData, method  = "glm")
fit.cart <- train(IH~., data=trainData, method="rpart")
fit.svm <- train(IH~., data=trainData, method="svmRadial")
fit.knn <- train(IH~., data=trainData, method="knn")
fit.rf <- train(IH~., data=trainData, method="rf")
fit.gbm <- train(IH~., data=trainData, method="gbm")
fit.nnet <- train(IH~., data=trainData, method="nnet")

# collect resamples
results <- resamples(list(CART=fit.cart,GLM=fit.glm,SVM=fit.svm,KNN=fit.knn, RF=fit.rf, NNET=fit.nnet, GBM=fit.gbm))
summary(results)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

# Prediction	
pred.cart <- predict(fit.cart, newdata = testData); cor(testData$IH, pred.cart)^2 
pred.glm <- predict(fit.glm, newdata = testData); cor(testData$IH, pred.glm)^2 
pred.svm <- predict(fit.svm, newdata = testData); cor(testData$IH, pred.svm)^2 
pred.knn <- predict(fit.knn, newdata = testData); cor(testData$IH, pred.knn)^2 
pred.rf <- predict(fit.rf, newdata = testData); cor(testData$IH, pred.rf)^2 
pred.nnet <- predict(fit.nnet, newdata = testData); cor(testData$IH, pred.nnet)^2 
pred.gbm <- predict(fit.gbm, newdata = testData); cor(testData$IH, pred.gbm)^2 

# Ensemble Meta-learner
predDF <- data.frame(pred.rf, pred.gbm, pred.glm, pred.cart, pred.svm, pred.nnet, pred.knn, IH = testData$IH)
# Train the ensemble
modelStack <- train(IH ~ ., data = predDF, method = "rf")
cor(predDF$IH,predict(modelStack))^2


### caretEnsemble
install.packages("caretEnsemble")
library(caretEnsemble)
library(caret)

install.packages("gtable")

##
## For regression type
###############################
#### Train base learner via caretList()
###############################
# data : agricolae package 

trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions="final", 
                             summaryFunction = defaultSummary
)


algorithmList <- c('rf', 'gbm', 'rpart', 'glm', 'svmRadial', 'knn', 'nnet')

set.seed(100)
models <- caretList(IH ~ ., data=trainData, 
                    trControl=trainControl, 
                    methodList=algorithmList,
                    metric = "RMSE"
                    ) 

# for single model
#m_lm <- train(IH ~ ., data = trainData, 
#              method = "glm", 
#              trControl = trainControl, 
#              metric="RMSE"
#)

scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

###############################
#Stack Models with caretStack()
#Train stacking meta-model
###############################
stackControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             summaryFunction = defaultSummary
                            ) 

set.seed(100)
# using RF
stack.rf <- caretStack(models, 
                       method="rf", 
                       metric="RMSE", 
                       trControl=stackControl
                       )
print(stack.rf)

###############################
# Make Predictions and Evaluate
###############################

stack_pred <- predict(stack.glm, newdata=testData)
cor(stack_pred, testData$IH)^2

rmse <- sqrt(mean((stack_pred$pred - testData$IH)^2))
cat("Stacked model RMSE on test data:", round(rmse, 3), "\n")



###############################
## For classification type
###############################
#### Train base learner via caretList()
###############################
# data : BiodiversityR 

trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions="final", 
                             classProbs = TRUE
)

# 'glm',  removed because of error
CLalgorithmList <- c('rf', 'gbm', 'rpart', 'svmRadial', 'knn', 'nnet')

set.seed(100)
models <- caretList(species ~ ., data=learn.set, 
                    trControl=trainControl, 
                    methodList=CLalgorithmList,
                    metric = "Accuracy"
) 

# for single model
m_lm <- train(species ~ ., data = learn.set, 
              method = "rpart", 
              trControl = trainControl, 
              metric="Accuracy"
)

scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

###############################
#Stack Models with caretStack()
#Train stacking meta-model
###############################
stackControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             classProbs = TRUE
) 

set.seed(100)
# Using RF
stack.rf <- caretStack(models, 
                        method="rf", 
                        metric="Accuracy", 
                        trControl=stackControl
)
print(stack.rf)

###############################
# Make Predictions and Evaluatton
###############################

stack_pred <- predict(stack.rf, newdata=valid.set)


# stack_pred is a data.frame with one probability column per class

# 1. Find the class with the highest predicted probability for each row
predicted_classes <- colnames(stack_pred)[max.col(stack_pred, ties.method = "first")]
# 2. Convert to factor with same levels as your true labels
stack_pred_factor <- factor(predicted_classes, levels = levels(valid.set$species))
# 3. Now confusionMatrix will work
confusionMatrix(stack_pred_factor, valid.set$species)

# a contingency table for final model output 
table(Predicted = predicted_classes, Actual = valid.set$species)

