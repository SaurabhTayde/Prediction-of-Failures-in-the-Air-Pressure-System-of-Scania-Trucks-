#setwd('D:/Saurabh T/R/Project/APS')
setwd('G:/DSP/Project/Final Theoery & Codes')

############### Import the dataset (Skip unnecessaory rows at start):

training_data_imp = read.csv('APSTrain.csv')
test_data_imp = read.csv('APSTest.csv')

class(training_data_imp$class)
class(test_data_imp$class)

training_data_imp$class = as.factor(training_data_imp$class)
test_data_imp$class = as.factor(test_data_imp$class)


############### Import libraries required:

library(dplyr)
install.packages('caret')
library(caret)

install.packages('caretEnsemble')
library(caretEnsemble)

install.packages('mice')
library(mice)

install.packages('doParallel')
library(doParallel)
library(car)


############### View Data. First look:

glimpse(training_data_imp)
glimpse(test_data_imp)


############### Chcking class imbalance:

summary(training_data_imp$class)
summary(test_data_imp$class)

prop.table(table(training_data_imp$class))  #Class is totally imbalanced.
prop.table(table(test_data_imp$class))      #Class is totally imbalanced.


#check dimensions
dim(training_data_imp)
dim(test_data_imp)



#Now as there is high class imbalance we will downsample the data

'%ni%' <- Negate('%in%')  # define 'not in' func

set.seed(123)

down_training_data_imp <- downSample(x = training_data_imp[, colnames(training_data_imp) %ni% "class"],
                         y = training_data_imp$class)

table(down_training_data_imp$Class)
dim(down_training_data_imp)


#################### Model Building



logitmod = glm(Class~., family = 'binomial', data = down_training_data_imp)

summary(logitmod)

#install.packages('MASS')

#library(MASS)

#stepAIC(logitmod)

#summary(logitmod2)

#predictTrain = predict(logitmod2, type = 'response')

predictTrain = predict(logitmod, type = 'response')

head(predictTrain)

tapply(predictTrain, down_training_data_imp$Class, mean)

table(down_training_data_imp$Class, predictTrain > 0.9 )

library(caret)
library(e1071)

class(predictTrain)
class(down_training_data_imp$Class)

#We need to convert both of these into factors as it is a pre-requisite for confusionMatrix

predictTrain1 = ifelse(predictTrain > 0.5, 1, 0)

predictTrain1 = as.factor(as.character(predictTrain1))

class(predictTrain1)


confusionMatrix(data= predictTrain1, reference=down_training_data_imp$Class)
# Accuracy : 0.9465  

#             Reference
#Prediction   0   1
#   0       969  76
#   1       31 924


#ROCR:

install.packages('ROCR')
library(ROCR)

ROCPred = prediction(predictTrain,down_training_data_imp$Class)

ROCPerf = performance(ROCPred, 'tpr', 'fpr')

plot(ROCPerf, colorize = T, print.cutoffs.at = seq(0,1,0.1), text.adj = c(-0.2, 1.7))




#Lets check test dataset now:

predicttest = predict(logitmod, type = 'response', newdata = test_data_imp)

predicttest1 = ifelse(predicttest>0.5, 1, 0)

table(test_data_imp$class, predicttest1> 0.5)


#Confusion marix for test data:

class(predicttest1)
class(test_data_imp$class)

predicttest1  = as.factor(as.character(predicttest1))

confusionMatrix(data = predicttest1, reference = test_data_imp$class) 
# Accuracy = 95.76

#               Reference
#Prediction     0     1
#       0     14991  44
#       1     634   331
