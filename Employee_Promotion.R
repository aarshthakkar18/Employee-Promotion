# Importing the library
library(knitr)
library(tidyverse)
library(rstatix)
library(caTools)
library(pROC)
library(ggplot2)
library(lsr)
library(dplyr)
library(corpcor)
library(corrplot)
library(caret)
library(randomForest)
library(e1071)

# Importing data
promotion_data <- read.csv('train.csv', na.strings = c(""))
# employee_id has no importance in the modeling or research so we would remove it from the dataset
promotion_data <- promotion_data[-1]
colnames(promotion_data)[10] <- "KPI>80"
colnames(promotion_data)[11] <- "awards_won"
# Data Description
str(promotion_data)
summary(promotion_data)
# Using the str and summary function to get a glance of data attributes and associated data types

# checking for missing data in the dataset
sapply(promotion_data,function(x) sum(is.na(x)))
# we found out that there are 2409 missing values in education column, whereas 4124 missing values in previous_year_rating column.

# We will remove missing data from our dataset
# using na.omit to remove missing values from the dataset
promotion_data <- na.omit(promotion_data)

# we will also check for the number of unique value in the column
sapply(promotion_data, function(x) length(unique(x)))
# from this we can get an idea of categorical variables existing in the dataset

# As noticed earlier that there are few data attributes which are categorical, so we will convert them into factor variables.
# using factor function to create factor variables for categorical data
promotion_data$education <- factor(promotion_data$education, ordered = TRUE, levels = c("Below Secondary","Bachelor's","Master's & above"))
promotion_data$no_of_trainings <- factor(promotion_data$no_of_trainings, ordered = TRUE)
promotion_data$previous_year_rating <- factor(promotion_data$previous_year_rating, ordered = TRUE)
promotion_data$region <- factor(promotion_data$region)
promotion_data$department <- factor(promotion_data$department)
promotion_data$gender <- factor(promotion_data$gender)
promotion_data$recruitment_channel <- factor(promotion_data$recruitment_channel)
promotion_data$is_promoted <- factor(promotion_data$is_promoted)
promotion_data$`KPI>80` <- factor(promotion_data$`KPI>80`)
promotion_data$awards_won <- factor(promotion_data$awards_won)
str(promotion_data)

# dimensionality reduction
pro <- promotion_data$is_promoted
a <- table(promotion_data$department,pro)
b <- table(promotion_data$region,pro)
c <- table(promotion_data$education,pro)
d <- table(promotion_data$gender,pro)
e <- table(promotion_data$recruitment_channel,pro)
f <- table(promotion_data$no_of_trainings,pro)
g <- table(promotion_data$previous_year_rating,pro)
h <- table(promotion_data$`KPI>80`,pro)
i <- table(promotion_data$awards_won,pro)

chisq.test(a,simulate.p.value = TRUE)
chisq.test(b,simulate.p.value = TRUE)
chisq.test(c,simulate.p.value = TRUE)
chisq.test(d,simulate.p.value = TRUE)
chisq.test(e,simulate.p.value = TRUE)
chisq.test(f,simulate.p.value = TRUE)
chisq.test(g,simulate.p.value = TRUE)
chisq.test(h,simulate.p.value = TRUE)
chisq.test(i,simulate.p.value = TRUE)


# checking the correlation of numeric variables amongst each other
promotion_data.numeric <- select_if(promotion_data, is.numeric)
corr <- cor(promotion_data.numeric)
corrplot(corr, tl.srt = 25, type = "lower", method = "number", tl.col = "black",
         sig.level = 0.05, insig = "blank", tl.cex = .8, main = "Numeric & Ordinal Correlations",
         mar = c(0,0,2,0))

set.seed(11)
Flagged <- findCorrelation(corr, 0.7, verbose = FALSE, names = TRUE, exact = TRUE)
print(Flagged)
# there is no flag variable in these

# now we will explore the association of continuous variable with target variable
aa <- table(promotion_data$age,pro)
bb <- table(promotion_data$length_of_service,pro)
cc <- table(promotion_data$avg_training_score,pro)

chisq.test(aa,simulate.p.value = TRUE)
chisq.test(bb,simulate.p.value = TRUE)
chisq.test(cc,simulate.p.value = TRUE)


# Exploratory data analysis
# promotion 
cc <- scale_fill_brewer(palette = "Set1")
# univariate analysis of is_promoted variable
is_pro <- ggplot(promotion_data,aes(is_promoted,fill = is_promoted))+geom_bar()+cc+ggtitle("Target variable")

# bivariate analysis of investigating variables discussed in background with respect to is_promoted
dep <- ggplot(promotion_data,aes(department,fill = is_promoted))+geom_bar()+cc
edu <- ggplot(promotion_data,aes(education,fill = is_promoted))+geom_bar()+cc
gen <- ggplot(promotion_data,aes(gender,fill = is_promoted))+geom_bar()+cc
len <- ggplot(promotion_data,aes(length_of_service,fill = is_promoted))+geom_bar()+cc


# Method

# splitting the dataset into training set and test set
set.seed(11)
split = sample.split(promotion_data$is_promoted, SplitRatio = 0.25)
training_set = subset(promotion_data, split == FALSE)
test_set = subset(promotion_data, split == TRUE)

# training the logistic regression model on the training_set
set.seed(11)
logistic_model = train(is_promoted ~ .,
                       training_set, method = "glm")
summary(logistic_model)
saveRDS(logistic_model,"Models/logistic_model.rds")

# used the saved model to save time in training the models 
logistic_Model <- readRDS("Models/logistic_model.rds")
summary(logistic_Model)

# training the random forest model on the training_set
set.seed(11)
random_forest_model = train(is_promoted ~ .,
                       training_set,method = "rf", 
                       ntree = 100, tuneGrid = data.frame(mtry = 6))
summary(random_forest_model)
saveRDS(random_forest_model,"Models/random_forest_model.rds")

# used the saved model to save time in training the models 
random_Forest_Model <- readRDS("Models/random_forest_model.rds") 

# training the decision tree model on the training_set
set.seed(11)
dt_model = train(is_promoted ~ .,training_set,
                 method = "rpart")
summary(dt_model)
saveRDS(dt_model,"Models/decisiontreemodel.rds")

# used the saved model to save time in training the models s
decision_Tree_Model <- readRDS("Models/decisiontreemodel.rds")

# predictions
set.seed(11)
pred_glm <- predict(logistic_Model, test_set[-13])
set.seed(11)
pred_rf <- predict(random_Forest_Model, test_set[-13])
set.seed(11)
pred_dt <- predict(decision_Tree_Model, test_set[-13])

pred_dt <- as.data.frame(pred_dt)
pred_dt$Output <- ifelse(pred_dt$`0`>pred_dt$`1`,0,1)

promotion_flag <- test_set$is_promoted
cm_glm <- table(promotion_flag, pred_glm)
cm_glm
cm_rf <- table(promotion_flag, pred_rf)
cm_rf
cm_dt <- table(promotion_flag, pred_dt$Output)
cm_dt

# calculating the accuracy of the logistic regression model
n_glm = sum(cm_glm) # total number of instances in the confusion matrix
diag_glm = diag(cm_glm) # number of correctly classified instances in the confusion matrix
accuracy_glm = sum(diag_glm) / n_glm 
accuracy_glm*100 # multiplying by 100 to get percentage value

# calculating the accuracy of the random forest model
n_rf = sum(cm_rf) # total number of instances in the confusion matrix
diag_rf = diag(cm_rf) # number of correctly classified instances in the confusion matrix
accuracy_rf = sum(diag_rf) / n_rf
accuracy_rf*100 # multiplying by 100 to get percentage value

# calculating the accuracy of the decision tree model
n_dt = sum(cm_dt) # total number of instances in the confusion matrix
diag_dt = diag(cm_dt) # number of correctly classified instances in the confusion matrix
accuracy_dt = sum(diag_dt) / n_dt
accuracy_dt*100 # multiplying by 100 to get percentage value