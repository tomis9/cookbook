library(titanic)
library(futile.logger)
library(gsubfn)

prepare_data <- function(titanic_train, test_perc) {
  drops <- c("PassengerId", "Name", "Ticket", "Cabin", "Embarked")
  train <- titanic_train[, !names(titanic_train) %in% drops]
  train$Survived <- factor(train$Survived, levels = c(0, 1))

  train <- train[sample(1:nrow(train)),]
  threshold <- floor(test_perc * nrow(train))
  test <- train[1:threshold,]
  train <- train[(threshold + 1):nrow(train),]

  mean_age <- mean(train[, "Age"], na.rm = TRUE)
  train[is.na(train$Age),"Age"] <- mean_age
  test[is.na(test$Age),"Age"] <- mean_age

  return(list(train = train, test = test))
}

bootstrap_models <- function(n, m, train, test, func, predict_class = NULL, ...) {
  accs <- c()
  models <- list()
  for (i in 1:n) {
    flog.info("calculating model no %d", i)
    sub_ind <- sample(1:nrow(train), floor(nrow(train) * 0.8), replace = TRUE)
    sub_train <- train[sub_ind, c(1, sample(2:ncol(train), m))]

    models[[i]] <- func(Survived ~ ., data = sub_train, ...)

    preds <- predict(models[[i]], test, class = predict_class)
    accs <- c(accs, sum(preds == test$Survived) / nrow(test))
  }
  return(list(accs = accs, models = models))
}

aggregate_scores <- function(models, test, pred_threshold) {
  preds <- predict(models[[1]], test) > pred_threshold

  for (i in 2:length(models)) {
    preds <- preds + (predict(models[[i]], test) > pred_threshold)
  }

  if (is.vector(preds)) {
    sum((preds < ceiling(nrow(test) / 2)) == as.integer(test$Survived)) / nrow(test)
  } else {
    print(sum(colnames(preds)[apply(preds, 1, which.max)] == 
              test$Survived) / nrow(test))
  }
  preds
}

list[train, test] <- prepare_data(titanic::titanic_train, 0.3)


list[accs, models] <- bootstrap_models(200, 4, train, test, func = rpart::rpart, predict_class = "class", control = list(maxdepth = 2))
sc <- aggregate_scores(models, test, pred_threshold = 0.5)

single_model <- rpart::rpart(Survived ~ ., train)
sum(predict(single_model, test, type = "class") == test$Survived) / nrow(test)


list[accs, models] <- bootstrap_models(200, 4, train, test, func = glm, family = binomial)
sc <- aggregate_scores(models, test, pred_threshold = 0)

# dlaczego dla niektórych jest NA?
single_model <- glm(Survived ~ ., train, family = binomial)
sum((as.integer(predict(single_model, test) > 0)) == test$Survived, na.rm = TRUE) / nrow(test)

