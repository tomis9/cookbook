library(car)

plot(population ~ year, data = USPop, main = "(a)")
abline(lm(population ~ year, data = USPop))

model <- nls(population ~ theta1  / (1 + exp( - (theta2 + theta3 * year))),
             data = USPop)
# trzeba wymyślić jakieś wartości startowe

# starting values

# choose t1, which is larger than any value in the data
t1 <- 400
model_start <- lm(logit(population / t1) ~ year, USPop)

starts <- c(t1, model_start$coefficients)
names(starts) <- c("theta1", "theta2", "theta3")

model <- nls(population ~ theta1  / (1 + exp(- (theta2 + theta3 * year))),
             data = USPop,
             start = starts,
             trace = TRUE)


f <- function(n) {
  x <- seq(-1, 1, length.out = n)
  t1 <- 5
  t2 <- 1
  t3 <- 1
  y <- abs(t1  / (1 + exp(- (t2 + t3 * x))) + rnorm(n, 0, 0.8)) + 1
  my_data <- data.frame(Y = y, X = x)

  t1 <- max(y) * 2
  model_start <- lm(logit(Y / t1) ~ X, my_data)

  starts <- c(t1, model_start$coefficients)
  names(starts) <- c("theta1", "theta2", "theta3")

  make_model <- function(theta1, theta2, theta3, my_data) {
     nls(
        Y ~ theta1 / (1 + exp(- (theta2 + theta3 * X))),
        data = my_data,
        start = starts,
        # trace = TRUE,
        control = nls.control(warnOnly = TRUE)
      )
  }
  make_simple_model <- function(my_data) lm(Y ~ X, my_data)

  model <- tryCatch(
    make_model(theta1, theta2, theta3, my_data),
    warning = function(w) {
      if (paste(strsplit(w$message, " ")[[1]][1:2], collapse = " ") %in%
          c("singular gradient", "number of", "step factor")) {
        model <- suppressWarnings(make_model(theta1, theta2, theta3, my_data))
        if (any(abs(coefficients(model)[1:3]) > 50))
          model <- make_simple_model(my_data)
        model
      } else {
        print(w$message)
        make_simple_model(my_data)
      }
    }
  )

  x_p <- seq(-1, 1.5, length.out = 100)
  new.data <- data.frame(X = x_p)
  yhat <- predict(model, new.data)
  print(model)
  plot(x, y, xlim = c(-1, 1.5), ylim = c(0, 8))
  lty <- if (class(model) == "nls") 1 else 2
  lines(x_p, yhat, col="blue", lty = lty)
}
f(3)

# a może warto by dodać addytywną stałą?
