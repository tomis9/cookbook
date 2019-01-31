# https://www.datacamp.com/community/tutorials/keras-r-deep-learning

devtools::install_github('rstudio/keras')

library(keras)

install_keras()

iris <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), header = FALSE)

names(iris) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width",
                 "Species")

# Determine sample size
ind <- sample(2, nrow(iris), replace = TRUE, prob = c(0.67, 0.33))

# Split the `iris` data
iris.training <- iris[ind == 1, 1:4]
iris.test <- iris[ind == 2, 1:4]

# Split the class attribute
iris.trainingtarget <- iris[ind == 1, 5]
iris.testtarget <- iris[ind == 2, 5]


iris.trainLabels <- to_categorical(unclass(iris.trainingtarget))
iris.testLabels <- to_categorical(unclass(iris.testtarget))

model <- keras_model_sequential()
model %>%
    layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>%
    layer_dense(units = 3, activation = 'softmax')

# this is what happens when you transalte OOP in Python to R ;)
get_config(model)
get_layer(model, "dense_1")
model$layers
model$layers[[1]]$name

model$inputs
model$outputs

model %>% compile(
     loss = "categorical_crossentropy",
     optimizer = "adam",
     metrics = "accuracy"
)

model %>% fit(
     as.matrix(iris.training),
     as.matrix(iris.trainLabels)[,2:4],
     epochs = 200,
     batch_size = 5,
     validation_split = 0.2
)

history <- model %>% fit(
     as.matrix(iris.training),
     as.matrix(iris.trainLabels)[,2:4],
     epochs = 200,
     batch_size = 5,
     validation_split = 0.2
)

plot(history)


plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
lines(history$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))
