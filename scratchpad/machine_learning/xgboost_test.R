library(xgboost)

xgb <- xgboost(data = as.matrix(iris[,1:4]), label = iris[,5], nrounds = 2)
