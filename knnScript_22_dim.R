#wokring for train.default
require(dplyr)
require(unbalanced)
require(randomForest)
require(h2o)
require(caret) #main lib
require(magrittr)
require(tidyquant)
require(class)
require(caret)




train<-read.csv("~/Kaggle_Training_Dataset_v2.csv")
test<-read.csv("~/Kaggle_Test_Dataset_v2.csv")



#checking the dataset
dim(train)

dim(test)



#heading some rows of it and checking metadata using names fx
head(train)

head(test)

names(train)

names(test)   


#using summary for basic analysis
summary(train)

summary(test)


#basic analysis using str
str(train)

str(test)


#phase for data processing


#selected
preprocess_raw_data <- function(data) {
  # data = data frame of backorder data
  data[data == -99] <- NA
  data %>%
    select(-sku) #%>%
  #mutate_if(is.character, .funs = function(x) ifelse(x == "Yes", 1, 0)) 
}

train <- preprocess_raw_data(train) 
test  <- preprocess_raw_data(test)



change_factor_level <- function(data) {
  data %>%
    mutate(went_on_backorder = as.factor(as.numeric(went_on_backorder)-2)) %>%
    mutate(rev_stop = as.factor(as.numeric(rev_stop)-2)) %>%
    mutate(stop_auto_buy = as.factor(as.numeric(stop_auto_buy)-2)) %>%
    mutate(ppap_risk = as.factor(as.numeric(ppap_risk)-2)) %>%
    mutate(oe_constraint = as.factor(as.numeric(oe_constraint)-2)) %>%
    mutate(deck_risk = as.factor(as.numeric(deck_risk)-2)) %>%
    mutate(potential_issue = as.factor(as.numeric(potential_issue)-2))
  
}

train <- change_factor_level(train)
test <- change_factor_level(test)


# How many NA values are there checking
sum(is.na(train))

sum(is.na(test))



#intension to remove NA
#Inner apply() reduces each value in column to TRUE or FALSE further
#The outer apply(), sums up column wise and gives output

apply(apply(train,2,is.na),2,sum) ; nrow(train)  

apply(apply(test,2,is.na),2,sum) ; nrow(test)


#problem with the last row all are NA so
#Removing last row from train and test data as that seems to be number of rows and an invalid row.
tail(train,1)
tail(test,1)


train_count <- nrow(train) 

test_count <- nrow(test)

train_count

test_count

train<-train[-train_count,]

test<-test[-test_count,]

apply(apply(train,2,is.na),2,sum) ; nrow(train)  

apply(apply(test,2,is.na),2,sum) ; nrow(test)


train %<>% na.roughfix()
test %<>% na.roughfix()



#Check for any further NA if there exist
sum(is.na(train))

sum(is.na(test))

str(train)
str(test)


#balancing the dataset
names(train)

X<-train[, -c(22)]    

# Exclude Id and target

#    Get y as a factor NOT 1,2 as now but 0 and 1 as ubSMOTE needs binary Y as input.


y<-as.factor(as.numeric(train[, 22])-2) 

#Balancing

b_data <- ubSMOTE(X = X, Y = y,   # Also y be a vector not a dataframe
                  perc.over=200,   #  200/100 = 2 instances generated for every rare instance
                  perc.under=200,  #  500/100 = 5 instances selected for every smoted observation
                  k=5,
                  verbose=TRUE) 


# ubSMOTE returns balanced data frame in b_data$X
#      and corresponding class values, as vector in b_data$Y
#       Return value 'b_data' itself is a list-structure
#     So complete and balanced train data is:

train <- cbind(b_data$X, went_on_backorder = b_data$Y)

table(train$went_on_backorder)/nrow(train)


#make every colume numeric


train$deck_risk <- as.numeric(train$deck_risk) -2
train$deck_risk <- as.numeric(train$deck_risk) -2
train$potential_issue <- as.numeric(train$potential_issue) -2
train$oe_constraint <- as.numeric(train$oe_constraint) -2
train$ppap_risk <- as.numeric(train$ppap_risk) -2
train$stop_auto_buy <- as.numeric(train$stop_auto_buy) -2
train$rev_stop <- as.numeric(train$rev_stop) -2
train$went_on_backorder <- as.numeric(train$went_on_backorder)+1

nrow(train)

names(train)

str(train)

dim(train)


#normalize the data between 0 and 1
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }


train_n <-as.data.frame(lapply(train, normalize))

str(train_n)


#dividing data between 70 to 30 ratio
set.seed(3456)

trainIndex <- createDataPartition(train_n$went_on_backorder, p = .7,  list = FALSE, times = 1)

train_data <- train_n[ trainIndex,]
test_data  <- train_n[-trainIndex,]


#excluding target variable
train_data1<-train_data[, -c(22)]
test_data1<-test_data[, -c(22)]



prc_test_pred <- knn(train_data1, test_data1, train_data[, 22], k=10)

confusionMatrix(prc_test_pred, test_data[, 22])


#KNN.cv model
knn_model <- knn.cv(train_n[,-22],train_n[, 22], k=10)

#prc_test_pred1 <- predict(knn_model,test$went_on_backorder)

confusionMatrix(knn_model,train_n$went_on_backorder)







