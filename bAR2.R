library(caret)
library(glmnet)
library(psych)
library(mlbench)
library(DescTools)
library(rpart)
library(pROC)
library(rpart.plot)
library(partykit)
# Data
read.csv('R:/BAR/Project/Bar1.xls', header = TRUE)
read.csv('R:/BAR/Project/BAR/Predict1.xlsx')
bar = Bar1[,-1]
LES =Predict1[,2:24]

# Encoding the target feature as factor
bar$Y<- as.factor(bar$Y)
bar$X2<- as.factor(bar$X2)
bar$X3<- as.factor(bar$X3)
bar$X4<- as.factor(bar$X4)
bar$X6<- as.factor(bar$X6)
bar$X8<- as.factor(bar$X8)
bar$X9<- as.factor(bar$X9)
bar$X10<- as.factor(bar$X10)
bar$X11<- as.factor(bar$X11)
# Encoding the target feature as factor
LES$Y<- as.factor(LES$Y)
LES$X2<- as.factor(LES$X2)
LES$X3<- as.factor(LES$X3)
LES$X4<- as.factor(LES$X4)
LES$X6<- as.factor(LES$X6)
LES$X8<- as.factor(LES$X8)
LES$X9<- as.factor(LES$X9)
LES$X10<- as.factor(LES$X10)
LES$X11<- as.factor(LES$X11)

# train test and validation
index_no<-sample(1:nrow(bar),9000,replace = F)
train<-bar[-index_no,]
test<-bar[index_no,]


# Custom Control Parameters
custom <- trainControl(method = "repeatedcv",
                       number = 5,
                       repeats = 5,
                       verboseIter = TRUE)
#logistic regression

set.seed(1234)
glm <- train(Y~.,
            bar,
            method='glmnet',
            family = 'binomial',
            tuneGrid = expand.grid(alpha = 0,
                                   lambda = 0),
            trControl = custom)

glm

predict <- predict(glm,test,type = 'prob')
pred<-ifelse(predict > 0.5, 1,0)
pred1<-pred[,2]
table(test$Y, pred1)
roc(test$Y,pred1)
plot(roc(test$Y,pred1,
         direction="<"),
     col="yellow", lwd=3, 
     main="logistic regression performance")




# Ridge Regression
set.seed(1234)
ridge <- train(Y~.,
               bar,
               method = 'glmnet',
               family= 'binomial',
               tuneGrid = expand.grid(alpha = 0,
                                      lambda = seq(0.005,1, length = 5)),
               trControl = custom)

# Plot Results
plot(ridge)
ridge
plot(ridge$finalModel, xvar = "lambda", label = T)
plot(ridge$finalModel, xvar = 'dev', label=T)
plot(varImp(ridge, scale=T))

#EVALUATION OF THE MODEL

predict <- predict(ridge,test,type = 'prob')
predr<-ifelse(predict > 0.5, 1,0)
predr1<-predr[,2]
table(test$Y, predr1)
roc(test$Y,predr1)


# Lasso Regression
set.seed(1234)
lasso <- train(Y~.,
               bar,
               method = 'glmnet',
               tuneGrid =expand.grid(alpha = 1,
                                     lambda=seq(0.01,1, length = 5)),
               trControl = custom)

# Plot Results
plot(lasso)
lasso
plot(lasso$finalModel, xvar = 'lambda', label=T)
plot(lasso$finalModel, xvar = 'dev', label=T)

#EVALUATION OF THE MODEL

predict <- predict(lasso,test,type = 'prob')
predl<-ifelse(predict > 0.5, 1,0)
predl1<-predl[,2]
table(test$Y, predl1)
roc(test$Y,predl1)

# Elastic Net Regression
set.seed(1234)
en <- train(Y~.,
            train,
            method = 'glmnet',
            tuneGrid = expand.grid(alpha = seq(0.01,1, length = 10),
                                   lambda = seq(0.01,.2, length = 10)),
            trControl = custom)
en

# Plot Results
plot(en)
plot(en$finalModel, xvar = 'lambda', label=T)
plot(en$finalModel, xvar = 'dev', label=T)
plot(varImp(en))

# Compare Models
model_list <- list(LinearModel = glm, Ridge = ridge, Lasso = lasso, ElasticNet = en)
res <- resamples(model_list)
summary(res)
bwplot(res)

# Best 
en$bestTune
best <- en$finalModel
coef(best, s = en$bestTune$lambda)


predict <- predict(en,test, type = 'prob')
pred<-ifelse(predict > 0.5, 1,0)
pred2<-pred[,2]
#confusion matrix
table(test$Y, pred2)


##CART model using Caret function.All three Decision trees models is included

bar2<-bar[,-22]
cart_models <- c("rpart", "rpart2", "rpart1SE")
lapply(cart_models, function(model) {train(Y ~., 
                                             data = bar2,
                                             method = model,
                                             trControl = custom,
                                             tuneLength = 10)}) -> cart_list_models
summary(cart_list_models)
cart_list_models

# Results based on test data by 4 CART models: 

lapply(cart_list_models, 
       function(model) {confusionMatrix(predict(model, test), 
                                        test$Y)})

##Random Forest
bar2<-bar[,-22]
RandomF <- train(Y~.,
            bar2,
            method='rf',
            ntree = 1000,
            tuneLength = 5,
            tuneGrid = 
            metric = "Accuracy",
            trControl = custom)
print(RandomF)

#Best model is chosen as per Confusion matrix F-measure value which is decision tree using rpart.
