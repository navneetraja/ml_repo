require(xgboost)
require(readr)
require(stringr)
require(caret)
require(car)
require(dplyr)
require(unbalanced)


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


#balancing train  dataset
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
train$went_on_backorder <- as.numeric(train$went_on_backorder)-2

nrow(train)

names(train)

str(train)

dim(train)




#balancing test  dataset
names(train)

X<-test[, -c(22)]    

# Exclude Id and target

#    Get y as a factor NOT 1,2 as now but 0 and 1 as ubSMOTE needs binary Y as input.


y<-as.factor(as.numeric(test[, 22])-2) 

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

test <- cbind(b_data$X, went_on_backorder = b_data$Y)

table(test$went_on_backorder)/nrow(train)


#make every colume numeric


test$deck_risk <- as.numeric(test$deck_risk) -2
test$deck_risk <- as.numeric(test$deck_risk) -2
test$potential_issue <- as.numeric(test$potential_issue) -2
test$oe_constraint <- as.numeric(test$oe_constraint) -2
test$ppap_risk <- as.numeric(test$ppap_risk) -2
test$stop_auto_buy <- as.numeric(test$stop_auto_buy) -2
test$rev_stop <- as.numeric(test$rev_stop) -2
test$went_on_backorder <- as.numeric(test$went_on_backorder)-1

nrow(test)

names(test)

str(test)

dim(test)





#for training
train_label <- train[,22]
train_matrix <- xgb.DMatrix(data = as.matrix(train[,-22]), label = train_label) 


#for test
test_label <- test[,22]
test_matrix <- xgb.DMatrix(data = as.matrix(test[,-22]), label = test_label)



# Parameters
nc <- length(unique(train_label))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
watchlist <- list(train = train_matrix, test = test_matrix)

# eXtreme Gradient Boosting Model
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = 1000,
                       watchlist = watchlist,
                       eta = 0.001,
                       max.depth = 3,
                       gamma = 0,
                       subsample = 1,
                       colsample_bytree = 1,
                       missing = NA,
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
pred <- matrix(p, nrow = nc, ncol = length(p)/nc) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label, max_prob = max.col(., "last")-1)
table(Prediction = pred$max_prob, Actual = pred$label)

confusionMatrix(Prediction = pred$max_prob, Actual = pred$label)





























xgb <- xgboost(data = data.matrix(train[,-22]), 
               label = train[,22], 
               eta = 0.1,
               max_depth = 15, 
               nround=25, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "merror",
               objective = "multi:softprob",
               num_class = 22,
               nthread = 3
)



# predict values in test set
y_pred <- predict(xgb, data.matrix(test[,22]))


confusionMatrix(y_pred,data.matrix(train[,22]))


















############






