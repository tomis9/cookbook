library(e1071)
library(forecast)

svc <- svm(Species ~ ., iris)

pred <- as.character(predict(svc, iris[, 1:4]))
mean(pred == iris["Species"])
