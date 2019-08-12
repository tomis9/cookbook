dtc <- rpart::rpart(Species ~ ., iris)
rpart.plot::rpart.plot(dtc)
