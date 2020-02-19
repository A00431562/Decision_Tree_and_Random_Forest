#***********************************1. R- code for Decision Tree :-
#Importing the libraries
getwd()
setwd("C:/Users/sidha/Documents/SMU/Semester 2/Data Mining/Assignments/Assignments Combined/Assignment 2/Assignment 2")
library(rpart)
library(rpart.plot)

#Import Dataset
car <- read.csv("car.csv", header = T)
car

# Splitting the datasets
set.seed(100)
train <- sample(nrow(car), 0.7*nrow(car), replace = FALSE)
TrainSet <- car[train,]
ValidSet <- car[-train,]
summary(TrainSet)
summary(ValidSet)

#Creating the tree
treecar = rpart(shouldBuy~., data = TrainSet, method = "class", control = rpart.control(minsplit = 30))
treecar
rpart.plot(treecar, type = 3,branch = .4, clip.right.labs = FALSE)

#Predict to create confusion matrix
predcar = predict(treecar, newdata = TrainSet, type = "class")
predcar
treeCar_matrix=table(TrainSet[,7],predcar)
treeCar_matrix
sum(diag(treeCar_matrix))/sum(treeCar_matrix)
predcarprob = predict(treecar, newdata = TrainSet, type = "prob")
predcarprob

library(pROC)
install.packages("pROC")
roc(TrainSet[,7],predcarprob[,2])
roc(TrainSet[,7],predcarprob[,1])
roc(TrainSet[,7],predcarprob[,3])
roc(TrainSet[,7],predcarprob[,4])

#ROC curve between Sensitivity and specificity
plot(roc(TrainSet[,7],predcarprob[,2]))
plot(roc(TrainSet[,7],predcarprob[,1]))
plot(roc(TrainSet[,7],predcarprob[,3]))
plot(roc(TrainSet[,7],predcarprob[,4]))

#--------------------------------------------
#Creating the tree while tuning minsplit =20
treecar1 = rpart(shouldBuy~., data = TrainSet, method = "class", control = rpart.control(minsplit = 20))
treecar1
predcar1 = predict(treecar1, newdata = TrainSet, type = "class")
predcar1
treeCar1_matrix =table(TrainSet[,7],predcar1)
treeCar1_matrix
sum(diag(treeCar1_matrix))/sum(treeCar1_matrix)

#--------------------------------------------
#Creating the tree while tuning minsplit =10
treecar2 = rpart(shouldBuy~., data = TrainSet, method = "class", control = rpart.control(minsplit = 10))
treecar2
predcar2 = predict(treecar2 , newdata = TrainSet, type = "class")
predcar2
treeCar2_matrix =table(TrainSet[,7],predcar2)
treeCar2_matrix
sum(diag(treeCar2_matrix))/sum(treeCar2_matrix)

#---------------------------Test Dataset---------------------------------
#Creating the tree
treecarvalid = rpart(shouldBuy~., data = ValidSet, method = "class", control = rpart.control(minsplit = 30))
treecarvalid
rpart.plot(treecarvalid,type = 3,branch = .4, clip.right.labs = FALSE)

#Predict to create confusion matrix
predcarvalid = predict(treecarvalid, newdata = ValidSet, type = "class")
predcarvalid
treeCarvalid_matrix =table(ValidSet[,7],predcarvalid)
treeCarvalid_matrix
sum(diag(treeCarvalid_matrix))/sum(treeCarvalid_matrix)
predcarprobvalid = predict(treecarvalid, newdata = ValidSet, type = "prob")
predcarprobvalid

library(pROC)
install.packages("pROC")
roc(ValidSet[,7],predcarprobvalid[,1])
roc(ValidSet[,7],predcarprobvalid[,2])
roc(ValidSet[,7],predcarprobvalid[,3])
roc(ValidSet[,7],predcarprobvalid[,4])

#ROC curve between Sensitivity and specificity
plot(roc(TrainSet[,7],predcarprob[,2]))
plot(roc(TrainSet[,7],predcarprob[,1]))
plot(roc(TrainSet[,7],predcarprob[,3]))
plot(roc(TrainSet[,7],predcarprob[,4]))




#*******************************2. R- code for Random Forest :-
###  R-Script :- 
library(randomForest)
library(pROC)
carData=read.csv("car.csv",header=T)
summary(carData)
set.seed(100)

#Constructing training and test dataset
train <- sample(nrow(carData), 0.7*nrow(carData), replace = FALSE)
TrainSet <- carData[train,]
TestSet <- carData[-train,]
summary(TrainSet)
summary(TestSet)

#Constructing model with defalut parameters
xc=TrainSet[,1:6]
yc=TrainSet[,7]
rf=randomForest(xc,yc)
rfp=predict(rf,xc)
rfCM=table(rfp,yc)
rfCM
sum(diag(rfCM))/sum(rfCM)

rfProb=predict(rf,xc,type="prob")
roc(TrainSet[,7],rfProb[,2])
plot(roc(TrainSet[,7],rfProb[,2]))

#Constructing tuned model with training dataset
xc=TrainSet[,1:6]
yc=TrainSet[,7]
rf=randomForest(xc,yc,ntree = 500,mtry = 4,nodesize = 10)
rfp=predict(rf,xc)
rfCM=table(rfp,yc)
rfCM
sum(diag(rfCM))/sum(rfCM)
rfProb=predict(rf,xc,type="prob")
roc(TrainSet[,7],rfProb[,1])
plot(roc(TrainSet[,7],rfProb[,1]))

#validating model on test dataset
xc=TestSet[,1:6]
yc=TestSet[,7]
rf=randomForest(xc,yc,ntree = 500,mtry = 4,nodesize = 10)
rfp=predict(rf,xc)
rfCM=table(rfp,yc)
rfCM
sum(diag(rfCM))/sum(rfCM)
rfProb=predict(rf,xc,type="prob")
roc(TestSet[,7],rfProb[,1])
plot(roc(TestSet[,7],rfProb[,1]))




#******************************************3. R- code for Random Forest using K-Fold Cross Validation: -
  # Random Forest with Default values but not fine-tuned.

library(caret)
library(randomForest)
x=carData[,1:6]
y=carData[,7]
head(x)
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
rf_default <- train(x,y, method="rf", metric=metric, trControl=control)
print(rf_default)
varImp(rf_default)
ggplot(varImp(rf_default))

# Random Forest using 'Random Search' and 'TuneLength' Parameters
library(caret)
control <- trainControl(method="repeatedcv", number=10, repeats=3, verboseIter = TRUE, search = "random")
seed <- 7
metric <- "Accuracy"
set.seed(seed)
rf_search <- train(x,y, method="rf", metric=metric,tuneLength = 10, trControl=control)
print(rf_search)
plot(rf_search)

# Random Forest using TuneGrid parameter for all mtry values.
library(caret)
control <- trainControl(method="repeatedcv", number=10, repeats=3, verboseIter = TRUE)
seed <- 7
metric <- "Accuracy"
set.seed(seed)

cmtry <- c(1,2,3,4,5,6)
tunegrid <- expand.grid(.mtry=cmtry)
rf_tuneGrid <- train(x,y, method="rf", metric=metric,tuneGrid = tunegrid, trControl=control)
print(rf_tuneGrid)
plot(rf_tuneGrid)

# tuneRF for OOB vs mtry
trf<-tuneRF(x,y,stepFactor = 1.5, ntreeTry = 400, doBest = TRUE)

# Final Model fitted with TuneGrid parameter for mtry = 2 && 4
library(caret)
control <- trainControl(method="repeatedcv", number=10, repeats=3, verboseIter = TRUE)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
cmtry <- c(2,4)
tunegrid <- expand.grid(.mtry=cmtry)
rf_2_vs_4 <- train(x,y, method="rf", metric=metric,tuneGrid = tunegrid, trControl=control)
print(rf_2_vs_4)
plot(rf_2_vs_4)
varImp(rf_2_vs_4)
ggplot(varImp(rf_2_vs_4))

