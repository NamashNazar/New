
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggpubr)) install.packages("ggpubr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(purrr)) install.packages("purrr", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(tinytex)) install.packages("tinytex", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(data.table)
library(ggpubr)
library(gridExtra)
library(purrr)
library(randomForest)
library(knitr)
library(tinytex)

# We load the data sets
url_train <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn"

# We see what the data looks like. 
class(url_train)
read_lines(url_train , n_max = 3)

# We find that the data is separated by space. We therefore use read.table function to turn it into a dataframe. 
train_set <- read.table(url_train )

# We then have a look at the data and structure to see if it worked.
str(train_set)

# We do the same steps for test data
url_test <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst"
read_lines(url_test , n_max = 3)
test_set <- read.table(url_test)
str(test_set)

# We now need to give the columns descriptive names to make analysis easier. 

train_set <- train_set %>% setNames(c("Top_left_spec1", "Top_left_spec2", "Top_left_spec3", "Top_left_spec4",
                                      "Top_center_spec1", "Top_center_spec2", "Top_center_spec3", "Top_center_spec4",
                                      "Top_right_spec1", "Top_right_spec2", "Top_right_spec3", "Top_right_spec4",
                                      "Middle_left_spec1", "Middle_left_spec2", "Middle_left_spec3", "Middle_left_spec4",
                                      "Middle_center_spec1", "Middle_center_spec2", "Middle_center_spec3", "Middle_center_spec4",
                                      "Middle_right_spec1", "Middle_right_spec2", "Middle_right_spec3", "Middle_right_spec4",
                                      "Bottom_left_spec1", "Bottom_left_spec2", "Bottom_left_spec3", "Bottom_left_spec4",
                                      "Bottom_center_spec1", "Bottom_center_spec2", "Bottom_center_spec3", "Bottom_center_spec4",
                                      "Bottom_right_spec1", "Bottom_right_spec2", "Bottom_right_spec3", "Bottom_right_spec4", 
                                      "Classification_code"))

# We give the output classification descriptive names as well.
train_set <- train_set %>% mutate(Classification= ifelse(Classification_code==1,"red soil", 
                                                         ifelse(Classification_code==2,"cotton crop",
                                                                ifelse(Classification_code==3,"grey soil",
                                                                       ifelse(Classification_code==4, "damp grey soil",
                                                                              ifelse(Classification_code==5,"soil with vegetation stubble",
                                                                                     ifelse(Classification_code==6, "mixture ","very damp grey soil")))))))

str(train_set)

# We do the same for test set
test_set <- test_set %>% setNames(c("Top_left_spec1", "Top_left_spec2", "Top_left_spec3", "Top_left_spec4",
                                      "Top_center_spec1", "Top_center_spec2", "Top_center_spec3", "Top_center_spec4",
                                      "Top_right_spec1", "Top_right_spec2", "Top_right_spec3", "Top_right_spec4",
                                      "Middle_left_spec1", "Middle_left_spec2", "Middle_left_spec3", "Middle_left_spec4",
                                      "Middle_center_spec1", "Middle_center_spec2", "Middle_center_spec3", "Middle_center_spec4",
                                      "Middle_right_spec1", "Middle_right_spec2", "Middle_right_spec3", "Middle_right_spec4",
                                      "Bottom_left_spec1", "Bottom_left_spec2", "Bottom_left_spec3", "Bottom_left_spec4",
                                      "Bottom_center_spec1", "Bottom_center_spec2", "Bottom_center_spec3", "Bottom_center_spec4",
                                      "Bottom_right_spec1", "Bottom_right_spec2", "Bottom_right_spec3", "Bottom_right_spec4", 
                                      "Classification_code"))
test_set <- test_set %>% mutate(Classification= ifelse(Classification_code==1,"red soil", 
                                                         ifelse(Classification_code==2,"cotton crop",
                                                                ifelse(Classification_code==3,"grey soil",
                                                                       ifelse(Classification_code==4, "damp grey soil",
                                                                              ifelse(Classification_code==5,"soil with vegetation stubble",
                                                                                     ifelse(Classification_code==6, "mixture ","very damp grey soil")))))))

str(test_set)

# We know that the dataset provides 4 spectral values for 9 pixels in a 3x3 pixel set. For ease of analysis we will use the middle pixels.

train_set <- train_set %>% select(c("Middle_center_spec1", "Middle_center_spec2", "Middle_center_spec3", "Middle_center_spec4","Classification"))
test_set <- test_set %>% select(c("Middle_center_spec1", "Middle_center_spec2", "Middle_center_spec3", "Middle_center_spec4","Classification"))

str(train_set)
str(test_set)

summary(train_set)

# How many times do different classifications appear

count_of_classifications<- train_set %>% ggplot(aes(Classification))+ geom_bar() + coord_flip()
count_of_classifications

# We make density plots for the four spectral bands for each land classification

dens_spec1<- train_set %>% 
  ggplot(aes(Middle_center_spec1, fill = Classification)) +
  geom_density(alpha=0.2)
dens_spec2<- train_set %>% 
  ggplot(aes(Middle_center_spec2, fill = Classification)) +
  geom_density(alpha=0.2)
dens_spec3<- train_set %>% 
  ggplot(aes(Middle_center_spec3, fill = Classification)) +
  geom_density(alpha=0.2)
dens_spec4<- train_set %>% 
  ggplot(aes(Middle_center_spec4, fill = Classification)) +
  geom_density(alpha=0.2)

dens_grid <- ggarrange(dens_spec1,dens_spec2,dens_spec3,dens_spec4,ncol=2,nrow=2, common.legend= TRUE, legend="bottom")
dens_grid

# Mean and Standard deviation of predictors distribution by classification
train_set %>% group_by(Classification) %>% summarize(mean_spec1=mean(Middle_center_spec1), sd_spec1=sd(Middle_center_spec1),
                                                     mean_spec2=mean(Middle_center_spec2), sd_spec2=sd(Middle_center_spec2),
                                                     mean_spec3=mean(Middle_center_spec3), sd_spec3=sd(Middle_center_spec3),
                                                     mean_spec4=mean(Middle_center_spec4), sd_spec4=sd(Middle_center_spec4))


# # Develop QQ plots
p <- seq(0.05, 0.95, 0.05)
sample_quantiles_spec1 <- quantile(train_set$Middle_center_spec1, p)
sample_quantiles_spec2 <- quantile(train_set$Middle_center_spec2, p)
sample_quantiles_spec3 <- quantile(train_set$Middle_center_spec3, p)
sample_quantiles_spec4 <- quantile(train_set$Middle_center_spec4, p)

theoretical_quantiles_spec1 <- qnorm(p, mean = mean(train_set$Middle_center_spec1), sd = sd(train_set$Middle_center_spec1))
theoretical_quantiles_spec2 <- qnorm(p, mean = mean(train_set$Middle_center_spec2), sd = sd(train_set$Middle_center_spec2))
theoretical_quantiles_spec3 <-  qnorm(p, mean = mean(train_set$Middle_center_spec3), sd = sd(train_set$Middle_center_spec3))
theoretical_quantiles_spec4 <- qnorm(p, mean = mean(train_set$Middle_center_spec4), sd = sd(train_set$Middle_center_spec4))

spec1_qq <-qplot(theoretical_quantiles_spec1, sample_quantiles_spec1) + geom_abline()
spec2_qq <-qplot(theoretical_quantiles_spec2, sample_quantiles_spec2) + geom_abline()
spec3_qq <-qplot(theoretical_quantiles_spec3, sample_quantiles_spec3) + geom_abline()
spec4_qq <-qplot(theoretical_quantiles_spec4, sample_quantiles_spec4) + geom_abline()

qq_grid <- ggarrange(spec1_qq,spec2_qq,spec3_qq,spec4_qq,ncol=2,nrow=2, common.legend= TRUE, legend="bottom")
qq_grid


# Convert our classification variable from "character" to "factor" to use caret package functions 
class(train_set$Classification)
train_set$Classification<- as.factor(train_set$Classification)
class(train_set$Classification)

class(test_set$Classification)
test_set$Classification<- as.factor(test_set$Classification)
class(test_set$Classification)

# Lets start with simple random classification to set a baseline. Note: we have not used "mixture" classification because we know for a fact that is not a part of the dataset
set.seed(1)
y_hat <- sample(c("red soil", "cotton crop","grey soil","damp grey soil","soil with vegetation stubble", "very damp grey soil"), length(nrow(test_set)), replace = TRUE) %>%
  factor(levels = levels(test_set$Classification))

baseline_acc <- mean(y_hat == test_set$Classification)
baseline_acc

# Set trainControl to "none" so the models apply to entire dataset instead of cross-validating using smaller datasets

control <- trainControl(method = "none")

# We first use LDA but we anticipate this would not perform well since we have more than 2 classifications being predicted
train_lda <- train(Classification ~ ., method = "lda", 
                   data = train_set,
                   trControl = control)

cm_lda<-confusionMatrix(predict(train_lda, test_set), test_set$Classification)
lda_acc_test <-confusionMatrix(predict(train_lda, test_set), test_set$Classification)$overall["Accuracy"]
lda_acc_test
lda_acc_train <-confusionMatrix(predict(train_lda, train_set), train_set$Classification)$overall["Accuracy"]
lda_acc_train

#We then use QDA to better fit the multiple classification and our accuracy improves
train_qda <- train(Classification ~ ., method = "qda", 
                      data = train_set,
                      trControl = control)
cm_qda<- confusionMatrix(predict(train_qda, test_set), test_set$Classification)
qda_acc_test<- confusionMatrix(predict(train_qda, test_set), test_set$Classification)$overall["Accuracy"]
qda_acc_test
qda_acc_train<- confusionMatrix(predict(train_qda, train_set), train_set$Classification)$overall["Accuracy"]
qda_acc_train

# KNN is further added to test performance but it performs almost same as QDA
train_knn <- train(Classification ~ ., method = "knn", 
                   data = train_set,
                   trControl = control)

cm_knn<-confusionMatrix(predict(train_knn, test_set), test_set$Classification)
knn_acc_test<- confusionMatrix(predict(train_knn, test_set), test_set$Classification)$overall["Accuracy"]
knn_acc_test
knn_acc_train<- confusionMatrix(predict(train_knn, train_set), train_set$Classification)$overall["Accuracy"]
knn_acc_train

# Finally we try random forests. There is only a marginal improvement over the QDA model, probably because of our control limiting cross-validation. 
train_rf <- randomForest(Classification ~ ., trControl = control, data=train_set)
cm_rf<-confusionMatrix(predict(train_rf, test_set),test_set$Classification)
rf_acc_test<- confusionMatrix(predict(train_rf, test_set),test_set$Classification)$overall["Accuracy"]
rf_acc_test
rf_acc_train<- confusionMatrix(predict(train_rf, train_set),train_set$Classification)$overall["Accuracy"]
rf_acc_train

## svmLinear:
train_svmLinear <- train(Classification ~ ., method = "svmLinear", 
                         data = train_set,
                         trControl = control)

cm_svmLinear<-confusionMatrix(predict(train_svmLinear, test_set), test_set$Classification)
svmLinear_acc_test<- confusionMatrix(predict(train_svmLinear, test_set), test_set$Classification)$overall["Accuracy"]
svmLinear_acc_test
svmLinear_acc_train<- confusionMatrix(predict(train_svmLinear, train_set), train_set$Classification)$overall["Accuracy"]
svmLinear_acc_train

## svmRadial:
train_svmRadial <- train(Classification ~ ., method = "svmRadial", 
                         data = train_set,
                         trControl = control)

cm_svmRadial<-confusionMatrix(predict(train_svmRadial, test_set), test_set$Classification)
svmRadial_acc_test<- confusionMatrix(predict(train_svmRadial, test_set), test_set$Classification)$overall["Accuracy"]
svmRadial_acc_test
svmRadial_acc_train<- confusionMatrix(predict(train_svmRadial, train_set), train_set$Classification)$overall["Accuracy"]
svmRadial_acc_train

## multinom
train_multinom <- train(Classification ~ ., method = "multinom", 
                        data = train_set,
                        trControl = control)

cm_multinom<-confusionMatrix(predict(train_multinom, test_set), test_set$Classification)
multinom_acc_test<- confusionMatrix(predict(train_multinom, test_set), test_set$Classification)$overall["Accuracy"]
multinom_acc_test
multinom_acc_train<- confusionMatrix(predict(train_multinom, train_set), train_set$Classification)$overall["Accuracy"]
multinom_acc_train

# Summary
Accuracy_results <- tibble(method = c("LDA","QDA","KNN","Random Forest","svmLinear","svmRadial","multinom"), Test_Accuracy = c(lda_acc_test,qda_acc_test,knn_acc_test,rf_acc_test,svmLinear_acc_test,svmRadial_acc_test, multinom_acc_test), Train_Accuracy=c(lda_acc_train,qda_acc_train,knn_acc_train,rf_acc_train,svmLinear_acc_train,svmRadial_acc_train,multinom_acc_train))
Accuracy_results %>% mutate(difference=Train_Accuracy-Test_Accuracy)

# We then try the approach of ensembles to see if it can improve our predictions. 
models <- c("qda", "knn","rf","multinom", "svmRadial")
fits <- lapply(models, function(model){ 
  print(model)
  train(Classification ~ ., method = model,trControl = control, data = train_set)
}) 

# We assign each fit with the name of the model used to train
names(fits) <- models

# We then get predictions for each model in the form of a data frame
pred <- sapply(fits, function(object) 
  predict(object, newdata = test_set))
dim(pred)
head(pred)

# We confirm the dimensions of our results. We should have one column for each model and one row for each row in test set.
length(models)
length(test_set$Classification)

# Now we compute accuracy for each model
acc <- colMeans(pred == test_set$Classification)
acc

# Now we commute average accuracy across models
mean(acc)

# We use voting approach (more than 50% votes) to pick ensemble prediction.
# First we find the proportion of models that have predicted each classification in each row. 
votes_cotton_crop  <- rowMeans(pred == "cotton crop")
votes_damp_grey_soil  <- rowMeans(pred == "damp grey soil")
votes_grey_soil  <- rowMeans(pred == "grey soil")
votes_red_soil  <- rowMeans(pred == "red soil")
votes_soil_with_vegetation_stubble  <- rowMeans(pred == "soil with vegetation stubble")
votes_very_damp_grey_soil<- rowMeans(pred == "very damp grey soil")

# Based on these proportions we made the ensembling prediction using the classification given by more than 50% of the models
y_hat <- ifelse(votes_cotton_crop > 0.5, "cotton crop", 
                ifelse(votes_damp_grey_soil>0.5, "damp grey soil",
                       ifelse(votes_grey_soil>0.5, "grey soil",
                              ifelse(votes_red_soil>0.5, "red soil",
                                     ifelse(votes_soil_with_vegetation_stubble>0.5,"soil with vegetation stubble",
                                            ifelse(votes_very_damp_grey_soil>0.5,"very damp grey soil",
                                            "poor prediction"))))))

# How many predictions have less than 50% of models predicting one classificatiom
length(which(y_hat=="poor prediction"))

# We then test its accuracy which is higher than individual accuracies of all models
en_acc_test<- mean(y_hat == test_set$Classification)
en_acc_test