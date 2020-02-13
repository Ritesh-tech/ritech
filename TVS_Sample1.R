library(standardize)

library(corrplot)
library(Hmisc)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)


#Importing the data
read.csv('R:/TVS Credit E.P.I.C. Analytics Case Study/Case 1 Personal Loan Propensity on Two Wheeler Customer Base/TVS sample1.csv')
dt1 = TVS_sample1

summary(dt1)

#Preprocessing
dt1$Age <- scale(dt1$Age)
dt1$V4 <- scale(dt1$V4)
dt1$V5 <- scale(dt1$V5)

dt1$V9 <- scale(dt1$V9)
dt1$V10 <- scale(dt1$V10)
dt1$V13 <- scale(dt1$V13)
dt1$V14 <- scale(dt1$V14)
dt1$V26 <- scale(dt1$V26)
summary(dt1)
#write.csv(dt1, "dt_scaled.csv")
#plot(dt1$Age)
bar = dt1[,]
#Correlation Matrix
bar$V32=as.numeric(bar$V32)
bar$V16=as.numeric(bar$V16)
bar$V31=as.numeric(bar$V31)
bar$V30=as.numeric(bar$V30)
bar$V29=as.numeric(bar$V29)
bar$V27=as.numeric(bar$V27)
bar$V28=as.numeric(bar$V28)
bar$V26=as.numeric(bar$V26)
bar$V25=as.numeric(bar$V25)
bar$V24=as.numeric(bar$V24)
bar$V23=as.numeric(bar$V23)
bar$V22=as.numeric(bar$V22)
bar$V21=as.numeric(bar$V21)
bar$V20=as.numeric(bar$V20)
bar$V19=as.numeric(bar$V19)
bar$V18=as.numeric(bar$V18)
bar$V17=as.numeric(bar$V17)
bar$V16=as.numeric(bar$V16)
bar$V15=as.numeric(bar$V15)
bar$V14=as.numeric(bar$V14)
bar$V13=as.numeric(bar$V13)
bar$V12=as.numeric(bar$V12)
bar$V11=as.numeric(bar$V11)
bar$V10=as.numeric(bar$V10)
bar$V9=as.numeric(bar$V9)
bar$V8=as.numeric(bar$V8)
bar$V7=as.numeric(bar$V7)
bar$V6=as.numeric(bar$V6)
bar$V5=as.numeric(bar$V5)
bar$V2=as.numeric(bar$V2)
bar$V4=as.numeric(bar$V4)
bar$Age=as.numeric(bar$Age)
dt2=as.matrix(bar)

res2 <- cor(dt2)
res2<-rcorr(dt2)
res2$P




#XG boost Preprocessing
x = bar[,-30]
y = bar[,30]

label = as.integer(bar$V32)

# train test and validation
v32 = bar$V32
VV32 = 0.25*v32
label = as.integer(bar$V32)
bar$V32 = NULL


n = nrow(bar)
train.index = sample(n,floor(0.75*n))
train.data = as.matrix(bar[train.index,])
train.label = label[train.index]
test.data = as.matrix(bar[-train.index,])
test.label = label[-train.index]

# Transform the two data sets into xgb.Matrix
xgb.train = xgb.DMatrix(data=train.data,label=train.label)
xgb.test = xgb.DMatrix(data=test.data,label=test.label)

# Define the parameters for classification
num_class = length(levels(v32))
params = list(
  booster="gbtree",
  eta=0.001,
  max_depth=5,
  gamma=3,
  subsample=0.75,
  colsample_bytree=1,
  objective="binary:logistic",
  eval_metric="auc"
)

# Train the XGBoost classifer
xgb.fit=xgb.train(
  params=params,
  data=xgb.train,
  nrounds=10000,
  nthreads=1,
  watchlist=list(val1=xgb.train,val2=xgb.test),
  verbose=TRUE
)

# Review the final model and results
xgb.fit
#XGBoost 
predict = predict(xgb.fit, xgb.test)
pred = as.data.frame(predict)
predd = ifelse(pred> 0.5,1,0)
x= as.data.frame(test.label)
table(x$test.label, predd)
x= as.data.frame(test.label)





