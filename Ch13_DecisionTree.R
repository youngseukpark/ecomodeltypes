###########################################
# R script used in Chapter 13 Decision tree models in machine learning
# Elsevier 2025. Ecological Model Types
# Authors: Sovan Lek & Young-Seuk Park
# Date: 2025.07.17
###########################################

# rpart package: to build decision tree models
# BiodiversityR package: to use data for classification

# install.packages("rpart")
# install.packages("BiodiversityR")

# load packages
library(rpart)   


library(BiodiversityR)

data(CucurbitaClim)  # load built-in dataset CucurbitaClim in BiodiversityR
ccdata <- CucurbitaClim	# 2404 x 24, assign data to ccdata 

# target variable ("species") includes 3 species
# rename species with short names
ccdata$species <- as.character(ccdata$species)
ccdata$species[ccdata$species=="Cucurbita_cordata"] <- "Cco" 
ccdata$species[ccdata$species=="Cucurbita_digitata"] <- "Cdi" 
ccdata$species[ccdata$species=="Cucurbita_palmata"] <- "Cpa" 
ccdata$species <- as.factor(ccdata$species)

# Split data (climate="baseline") to learn and test data
train.set <- subset(ccdata, climate == "baseline", select = c("species","bioc1","bioc2","bioc3","bioc4","bioc5","bioc6","bioc7","bioc8","bioc9","bioc10","bioc11","bioc12","bioc13","bioc14","bioc15","bioc16","bioc17","bioc18","bioc19"))
future.set <- subset(ccdata, climate == "future", select = c("bioc1","bioc2","bioc3","bioc4","bioc5","bioc6","bioc7","bioc8","bioc9","bioc10","bioc11","bioc12","bioc13","bioc14","bioc15","bioc16","bioc17","bioc18","bioc19"))
future.sp <- subset(ccdata, climate == "future", select ="species")

# initialize the model with random number
set.seed(222) 

# Build the model with rpart()
rpart.mod.cl<-rpart(species~.,train.set, cp=0.01) # cp default: 0.01

# plot decision tree, branch=1 for strait
plot(rpart.mod.cl, uniform=TRUE, branch=0.8, margin=0.05)
text(rpart.mod.cl,all=TRUE, use.n=TRUE, cex=0.8)

plotcp(rpart.mod) # plot cp vs x-val relative error

# predict with train.set
rpart.mod.cl.pred<-predict(rpart.mod.cl,type="class")
# contingency table of output
table(train.set$species,rpart.mod.cl.pred)

install.packages("vcd")
library(vcd)  # to use agreementplot()

# Plot agreement between predicted and observed values 
agreementplot(table(train.set$species,rpart.mod.cl.pred), xlab="Observed values", ylab="Predicted values")

# Plot variable importance
barplot(rpart.mod.cl$variable.importance,las=2, xlab="Input Variables", ylab="Importance")

# Prediction with new data
rpart.future.cl.pred<-predict(rpart.mod.cl,type="class", newdata=future.set)
table(future.sp$species,rpart.future.cl.pred)


##################################################
# to apply regression decision tree
##################################################

# agricolae package: to use data for regression
# install.packages("agricolae")
library(agricolae)

# load data
data(yacon) # 432 x 13Q + 1Q

# split data to train and test data with 75%:25% ratio
split = sample.split(yacon$IH, SplitRatio = .75)
Ytrain.set = subset(yacon[,6:19], split == TRUE)
Ytest.set = subset(yacon[,6:19], split == FALSE)

# Build (or fit) a regression decision tree with rpart() 
rpart.mod.reg<-rpart(IH~., Ytrain.set)

# Plot the regression decision tree
plot(rpart.mod.reg, uniform=TRUE, branch=0.8, margin=0.05) # branch=1 for strait 
text(rpart.mod.reg,all=TRUE, use.n=F, cex=0.6)
     
# Calculate coefficient of determination (r2) and root mean square error (rmse)
r2<-round(cor(predict(rpart.mod.reg),Ytrain.set$IH)^2, 4)
rmse=round(sqrt(mean((predict(rpart.mod.reg)-Ytrain.set$IH)^2)), 4)
table(r2,rmse)

pred.tr.reg <- predict(rpart.mod.reg)

# Print the cost-complexity table
printcp(rpart.mod.reg)

# Plot the cross-validated error vs cp (complexity parameter)
plotcp(rpart.mod.reg)

# prune with new cp 
rpart.mod.reg.prune<- prune(rpart.mod.reg, cp = 0.017720)

# plot decision tree with new cp, branch=1 for strait
plot(rpart.mod.reg.prune, uniform=TRUE, branch=0.8, margin=0.05)
text(rpart.mod.reg.prune, all=TRUE, use.n=F, cex=0.6)

# #######
# Pruning 
# Plot the relations between observed and predicted values
pred.tr.prune <- predict(rpart.mod.reg.prune, Ytrain.set)
plot(Ytrain.set$IH, pred.tr.prune, xlim=c(0.1,0.4), ylim=c(0.1,0.4), xlab="Observed values", ylab="Predicted values")
abline(0,1)

# Calculate coefficient of determination (r2) and root mean square error (rmse)
r2<-round(cor(pred.tr.prune,Ytrain.set$IH)^2, 4)
rmse=round(sqrt(mean(pred.tr.prune-Ytrain.set$IH)^2), 4)
table(r2,rmse)

# #######
# Test with new data
# predict with new data (test data, 25%)
pred.ts.reg<-predict(rpart.mod.reg.prune, newdata=Ytest.set[,-14])

# Plot the relations between observed and predicted values
plot(Ytest.set$IH, pred.ts.reg, xlim=c(0.1,0.4), ylim=c(0.1,0.4), xlab="Observed values", ylab="Predicted values")
abline(0,1)

# Calculate model performance with new data: R^2 and RMSE
r2<-round(cor(pred.ts.reg, Ytest.set$IH)^2, 4)
rmse<-round(sqrt(mean((pred.ts.reg - Ytest.set$IH)^2)),4)
table(r2,rmse)

# Plot variable importance
barplot(rpart.mod.reg.prune$variable.importance, las=2, xlab="Input Variables", ylab="Importance")

