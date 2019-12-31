# http://thestatsgeek.com/2015/06/08/my-first-foray-with-stan/

model_code <- "
data {
  int N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real sigma;
}
model {
  y ~ normal(alpha + beta * x, sigma);
}
"

set.seed(123)
n <- 100
x <- rnorm(n)
y <- x + rnorm(n)
mydata <- list(N = n, y = y, x=x)

fit <- stan(model_code = model_code, data = mydata, 
            iter = 1000, chains = 4)
fit

result <- extract(fit)
str(result)

summary(lm(y~x))

# http://mc-stan.org/shinystan/articles/shinystan-package.html

library(shinystan)
launch_shinystan(fit)


# http://m-clark.github.io/workshops/bayesian/04_R.html#rstanarm
library(rstanarm)
mydata_df <- data.frame(y = y, x = x)
fit2 <- stan_glm(y ~ x, data = mydata_df, iter=1000, chains = 4)
fit2

launch_shinystan(fit2)

