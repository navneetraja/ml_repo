#wokring for train.default
require(dplyr)
require(unbalanced)
require(randomForest)
require(h2o)
require(caret) #main lib
require(magrittr)
require(tidyquant)
require(class)
require(xgboost)
require(readr)
require(stringr)
require(car)
require(Matrix)



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


#intension to remove NA
#Inner apply() reduces each value in column to TRUE or FALSE further
#The outer apply(), sums up column wise and gives output

apply(apply(train,2,is.na),2,sum) ; nrow(train)  

apply(apply(test,2,is.na),2,sum) ; nrow(test)



train %<>% na.roughfix()
test %<>% na.roughfix()

#Check for any further NA if there exist
sum(is.na(train))

sum(is.na(test))

str(train$went_on_backorder)
str(test)

# How many NA values are there checking
sum(is.na(train))

sum(is.na(test))




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



#######################reducting some dimensions##########################


#selected
selection <- function(data) {
  # data = data frame of backorder data
  data %>%
    select(-potential_issue) %>%
    select(-pieces_past_due) %>%
    select(-perf_6_month_avg) %>%
    select(-perf_12_month_avg) %>%
    select(-deck_risk) %>%
    select(-oe_constraint) %>%
    select(-ppap_risk) %>%
    select(-stop_auto_buy) %>%
    select(-rev_stop) 
}

train <- selection(train) 
test  <- selection(test)

#improtant

names(test)
names(test)

X<-train[, -c(13)]    

# Exclude Id and target

#    Get y as a factor NOT 1,2 as now but 0 and 1 as ubSMOTE needs binary Y as input.


y<-as.factor(as.numeric(train[, 13]))

#Balancing

b_data <- ubSMOTE(X = X, Y = train[,13],   # Also y be a vector not a dataframe
                  perc.over=200,   #  200/100 = 2 instances generated for every rare instance
                  perc.under=200,  #  500/100 = 5 instances selected for every smoted observation
                  k=5,
                  verbose=TRUE) 


# ubSMOTE returns balanced data frame in b_data$X
#      and corresponding class values, as vector in b_data$Y
#       Return value 'b_data' itself is a list-structure
#     So complete and balanced train data is:


train <- cbind(b_data$X, went_on_backorder = b_data$Y)



table(train1$went_on_backorder)/nrow(train1)

nrow(train)

names(train)

str(train$went_on_backorder)


#######################reducting some dimensions##########################
#for training
train1<-sparse.model.matrix(went_on_backorder ~., data = train)

train_label <- train[,13]
train_matrix <- xgb.DMatrix(data = as.matrix(train1),label = train_label) 


#for test
test1<-sparse.model.matrix(went_on_backorder ~., data = test)
test_label <- test[,13]
test_matrix <- xgb.DMatrix(data = as.matrix(test1), label = test_label)

# Parameters
nc <- length(unique(train_label))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
watchlist <- list(train = train_matrix, test = test_matrix)

# eXtreme Gradient Boosting Model
bst_model <- xgb.train(data = train_matrix,
                       nrounds = 300,
                       watchlist = watchlist,
                       eta = 0.001,
                       nthread = 40,
                       max.depth = 15,
                       seed = 333)

# Training & test error plot
e <- data.frame(bst_model$evaluation_log)
plot(e$iter, e$train_mlogloss, col = 'blue')
lines(e$iter, e$test_mlogloss, col = 'red')

min(e$test_mlogloss)
e[e$test_mlogloss == 0.625217,]

# Feature importance
imp <- xgb.importance(colnames(train_matrix), model = bst_model)
print(imp)
xgb.plot.importance(imp)

# Prediction & confusion matrix - test data
p <- predict(bst_model, newdata = test_matrix)

summary(p)
p <- ifelse (p > 0.6547,1,0)

confusionMatrix(as.factor(p),test$went_on_backorder)

pred <- matrix(p) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label, max_prob = max.col(., "last")-1)
table(Prediction = pred$max_prob, Actual = pred$label)



 
preds<-predict(model,test$went_on_backorder)



#cv
xgbcv <- xgb.cv(data = train_matrix, nrounds = 500, nfold = 5,
                 maximize = F,nthread=200)
xgbcv$
