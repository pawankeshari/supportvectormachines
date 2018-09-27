##################################################### SVM Digits Recogniser Assignment #######################################################
# Cleaning up the environment
remove (list=ls())

# Install & Load the required packages
#install.packages("caret")
#install.packages("kernlab")
#install.packages("dplyr")
#install.packages("readr")
#install.packages("ggplot2")
#install.packages("gridExtra")

library(caret)
library(kernlab)
library(dplyr)
library(readr)
library(ggplot2)
library(gridExtra)

##############################################################################################################################################

### 1. Business Understanding

# Our goal is to develop a model which can correctly identify the digits (between 0-9) written in an image form.
# We would be preseneted with a training & a test data set for this task.


##############################################################################################################################################

### 2. Data Understanding

# Importing the dataset
mnistTrainDataSet <- read.csv('./Predictive_Analytics_II/Assignment - Support Vector Machine/mnist_train.csv', stringsAsFactors = F, na.strings=c("NA","#DIV/0!", "","NaN"))
mnistTestDataSet <- read.csv('./Predictive_Analytics_II/Assignment - Support Vector Machine/mnist_test.csv', stringsAsFactors = F, na.strings=c("NA","#DIV/0!", "","NaN"))

# From below command we can see that there are 785 rows. Out of this 1st column is the target column i.e. target label.
# Remaining 784 columns indicate pixel values of 28*28 image size.
ncol(mnistTrainDataSet)
ncol(mnistTestDataSet)

# Also we can see that there are around ~60K observations in the train data set. and around ~ 10K observations in test data set.
nrow(mnistTrainDataSet)
nrow(mnistTestDataSet)

# Understanding the structure of dataset
str(mnistTrainDataSet)
# As we can see all the variables are in "int" format & this is the expected one for model training.


##############################################################################################################################################

### 3. Data Preparation

# Checking for NA values
sum(is.na(mnistTrainDataSet))
sum(is.na(mnistTestDataSet))
# As we can see there no NA values in this data set. Above command returns "0".

# Checking for duplicate values
length(unique(mnistTrainDataSet))
length(unique(mnistTestDataSet))
# As we can see there no duplicate values in this data set. Above command returns "785" columns.
# NOTE: Since there are only values ranging from "0-9", "0-255" in this dataset, running unique command over columns doens't make sense.
# Therefore, running "unique" command over columns will tell us only about that there are values only these ranges.

# Renaming all the columns in MNIST train and test data to standardize the column names.
names(mnistTrainDataSet) <- paste0("column", seq_len(ncol(mnistTrainDataSet)))
names(mnistTestDataSet) <- paste0("column", seq_len(ncol(mnistTestDataSet)))

# Renaming "X5" as "target" variable
colnames(mnistTrainDataSet)[1] <- "target"
colnames(mnistTestDataSet)[1] <- "target"

# Coverting our target variable in factor format i.e. "target"
mnistTrainDataSet$target <- factor(mnistTrainDataSet$target)
mnistTestDataSet$target <- factor(mnistTestDataSet$target)
str(mnistTrainDataSet)
str(mnistTestDataSet)

# Scaling up of data set
# Since the values of all pixels are already in a range of 0-255, we don't need any explicit scaling here.
# So we will not go ahead scale the data set.

# Removing outliers
# Again since all the entries in the dataset is in range of 0-255, there are no outliers present in this dataset.
# So outlier removing process will be done.

# Deriving new fields for dataset
# Again since all the input data represents only pixel, so we don't have to create any new column or feature here for training the model.
# The already provided dataset should suffice for modelling here.


##############################################################################################################################################

### 4. Model Building

# We will be using following methods here to compute our model & then validate it on test set.
# Linear kernel
# RBF Kernel
# SVMRadial with 5-fold Cross Validation  


# Creating train & test data set
# NOTE: As mentioned in assignments and by our student mentor, that MNIST train dataset is very large and to create models on a smaller data set.
# And therefore we will take only 15% of the data set for now.
# Sampling the train data set for only 15% of the data.
set.seed(1)
train.indices = sample(1:nrow(mnistTrainDataSet), 0.15*nrow(mnistTrainDataSet))
train = mnistTrainDataSet[train.indices, ]

# For test data we can take whole MNIST test data set to consider it as a test data.
# Since more test data is going to useful only to make our predictions better. We will not take 15% of the test data as we did for train.
test = mnistTestDataSet

#Constructing the Models

#Using Linear Kernel
modelLinear <- ksvm(target~ ., data = train, scaled = FALSE, kernel = "vanilladot")
evaluationLinear<- predict(modelLinear, test)
#confusion matrix - Linear Kernel
confusionMatrix(evaluationLinear,test$target)
# This gives us the accuracy of around 91%. The sensivity and specificity for most of the digits are more than 85% which is fine.

# Using polydot
ModelPolydot <- ksvm(target~ ., data = train, scaled = FALSE, kernel = "polydot")
evaluationPolydot<- predict(ModelPolydot, test)
#confusion matrix - RBF Kernel
confusionMatrix(evaluationPolydot,test$target)
# This gives us the accuracy of around 91%. The sensivity and specificity for most of the digits are more than 85% which is same as "vanilladot".

# Since we tried "vanilladot" & "polydot", both are giving same accuracy and other parameters. This means that now we require some non-liner kernels to separate the data into hyperplanes.

# So let's try RBF Kernel
ModelRBF <- ksvm(target~ ., data = train, scaled = FALSE, kernel = "rbfdot")
evaluationRBF<- predict(ModelRBF, test)
#confusion matrix - RBF Kernel
confusionMatrix(evaluationRBF,test$target)
# This gives us the accuracy of around 95%. The sensivity and specificity for most of the digits are more than 90% which is better than "vanilladot".
# Also we can get the value of sigma from "ModelRBF"
ModelRBF
# The value is "Hyperparameter : sigma =  1.63157694405042e-07"

# So from above we can see that the "rbfdot" kernel is giving us more accuracy than "vanilladot" & "polydot".
# Now we will go ahead and try tuning the "Sigma" & "C" (Cost of misclassification) on "svmRadial" kernel using cross-validation.


##############################################################################################################################################

### 5. Hyperparameter tuning & cross validation

# Now we will use cross validation technique to find the best value of "Sigma" and "C".
# We will use "svmRadial" for training our model here. Now we have decided to choose "svmRadial" because of the fact that it is non-linear model and also gives a flixibility of tuning 2 parameters i.e. "C" & "Sigma".
# In this we can make our trained model more robust.

# We will use for this "traincontrol" to set the values of parameters and create a train function. We will do the 5-fold cross validation here.
trainControl <- trainControl(method="cv", number=5)

# We will set, Metric <- "Accuracy" to our evaluation metric as "Accuracy".
metric <- "Accuracy"

# We will use the "Expand.grid" functions to set our hyperparameters. We will pass this to our model.
set.seed(7)
#grid <- expand.grid(.sigma=c(0.025, 0.05), .C=c(0.1,0.5,1,2))

### Now we will go ahead and train out model on different hyperparameters.

# Model_Creation_1:
# We will use the value of "Sigma" which we got from "ModelRBF". Here we will first fix "Sigma" & vary the values of "C" from "0-300"
grid1 <- expand.grid(.sigma=c(1.63157694405042e-07), .C=c(0, 10, 100, 300) )
# Train function takes Target ~ Prediction, Data, Method = Algorithm, Metric = Type of metric, tuneGrid = Grid of Parameters, trcontrol = Our traincontrol method
fit.svm1 <- train(target~., data=train, method="svmRadial", metric=metric, tuneGrid=grid1, trControl=trainControl)
# Check the fit values
print(fit.svm1)

# From this model we can see that the, Accuracy is best for "C=10" around "96.22%". As we increase the "C", the accuracy is decreasing.
# So we can assume that, increasing the "C" is overfitting the trained model. Now we will go ahead and try some values of "C" between "1-10" to see if we are able to increase the accuracy or not.
# For values greater than "C>300", we can assume that it is going to overfit more amd more. Hence we will not look for values of "C" greater than "300".

# Let's plot and see the variations of the "C"
plot(fit.svm1)
# Again the accuracy is degrading as we increase the value of "C".

# Validating the model results on test data
evaluateSVMRadial1<- predict(fit.svm1, test)
confusionMatrix(evaluateSVMRadial1, test$target)
# "Model_Creation_1" gives us the accuracy of around 96.53%. The sensivity and specificity for most of the digits are more than 95% which is better than all the previously trained models.
# We can see that the training and test data accuracy is almost similar at around 96%.


# Model_Creation_2:
# We will again use the same value of "Sigma" which we got from "ModelRBF". We will vary the value of "C" from "0.1-9"
grid2 <- expand.grid(.sigma=c(1.63157694405042e-07), .C=c(0.1, 3, 6, 9) )
# Train function takes Target ~ Prediction, Data, Method = Algorithm, Metric = Type of metric, tuneGrid = Grid of Parameters, trcontrol = Our traincontrol method
fit.svm2 <- train(target~., data=train, method="svmRadial", metric=metric, tuneGrid=grid2, trControl=trainControl)
# Check the fit values
print(fit.svm2)

# From this model we can see that the, Accuracy is best for "C=3" around "96.05%". For "C" smaller and greater than this the accuracy is lesser.
# So we can assume that, for "C=3" is the best for this trained model.

# Let's plot and see the variations of the "C"
plot(fit.svm2)
# Again the accuracy as we described above for this model. Best at "C=3"

# Validating the model results on test data
evaluateSVMRadial2<- predict(fit.svm2, test)
confusionMatrix(evaluateSVMRadial2, test$target)
# "Model_Creation_2" gives us the accuracy of around 96.54%. The sensivity and specificity for most of the digits are more than 95% which is better than all the previously trained models and almost similar to "Model_Creation_1".
# We can see that the training and test data accuracy is almost similar at around 96%.


## Now summarizing for "Model_Creation_1" & "Model_Creation_2" we can see that:
  # a. "Model_Creation_1" has training accuracy of 96.22% with "C = 10" & "Sigma = 1.63157694405042e-07", Test data accuracy is 96.53%.
  # b. "Model_Creation_2" has training accuracy of 96.05% with "C = 3" & "Sigma = 1.63157694405042e-07", Test data accuracy is 96.54%.
  # c. So we can pick any of "C = 3" or "C = 10" for this assignment as both has almost similar train & test data accuracy.
  # d. Now since "C" is cost of misclassification & to keep model simple we can pick "C = 3" as our final "C" value. Lower "C" values generalize well.

# Now we will tune the "Sigma parameter" by keeping "C = 3"


# Model_Creation_3:
# We will keep "C = 3" & vary the values of "Sigma". Since we already have a value of "Sigma = 1.63157694405042e-07" (which is similar to 1e-07), we will use this to create our first set of values. We will vary the value of "Sigma" from "1e-05 to 1e-09"
grid3 <- expand.grid(.sigma=c(1e-05, 1e-06, 1e-07, 1e-08, 1e-09 ), .C=c(3) )
# Train function takes Target ~ Prediction, Data, Method = Algorithm, Metric = Type of metric, tuneGrid = Grid of Parameters, trcontrol = Our traincontrol method
fit.svm3 <- train(target~., data=train, method="svmRadial", metric=metric, tuneGrid=grid3, trControl=trainControl)
# Check the fit values
print(fit.svm3)

# From this model we can see that the, Accuracy is best for "Sigma = 1e-06" around "95.33%". For "sigma" smaller and greater than this the accuracy is lesser.
# Forlarger values of "sigma" i.e. "1e-05" training accuracy is very poor at around 19%. And For very small values i.e. 1e-09 the accuracy is much lesser than what it is at "Sigma = 1e-06".
# So we can assume that, for "Sigma = 1e-06" is the best "Sigma" for this trained model.

# Let's plot and see the variations of the "Sigma"
plot(fit.svm3)
# Again the accuracy as we described above for this model. Best at "Sigma = 1e-06". The graph decrease very sharply as we increase "Sigma" value.
# This shows that for larger values of "Sigma", model is behaving like a liner kernel rather than gaussian kernel. 

# Validating the model results on test data
evaluateSVMRadial3<- predict(fit.svm3, test)
confusionMatrix(evaluateSVMRadial3, test$target)
# "Model_Creation_3" gives us the accuracy of around 95.76%. The sensivity and specificity for most of the digits are more than 95% which is better than all the previously trained models and almost similar to "Model_Creation_1" & "Model_Creation_2".
# We can see that the training and test data accuracy is almost similar at around 95%.

## Summarizing the outcomes:
  # a. For "Model_Creation_3", we have values of our hyperparameters i.e."C = 3" & "Sigma = 1e-06".
    # i. Training Accuracy: 95.33%
    # ii. Sensivity & Specificity: ~ > 95% for all the digits.
    # iii. Test data accuracy: 95.76%
  # b.For "Model_Creation_2", we have values of our hyperparameters i.e."C = 3" & "Sigma = 1.63157694405042e-07".
      # i. Training Accuracy: 96.05%
      # ii. Sensivity & Specificity: ~ > 95% for all the digits.
      # iii. Test data accuracy: 96.54%
  # Keeping in minds these results, we can see that the overall best accuracies, sensitivity & specificity we are getting for "Kernel = svmRadial", "C = 3" & "Sigma = 1.63157694405042e-07".

### So our final model has parameters "Kernel = svmRadial", "C = 3" & "Sigma = 1.63157694405042e-07" in terms of best accuracies, sensitivity & specificity for all the digits involved in training process.

############################################################ END OF CASE STUDY ###############################################################