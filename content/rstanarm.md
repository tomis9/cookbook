---
title: "rstanarm"
date: 2018-12-23T20:19:18+01:00
draft: false
categories: ["scratchpad"]
tags: []
---

<center>
# This is not a proper blog post yet.

rstanarm (TODO)
</center>

## 1. What is rstanarm and why would you use it?

* it's an R interface to [stan](https://mc-stan.org/)

* it's better than rstan, because (according to [rstanarm webpage](https://mc-stan.org/users/interfaces/rstanarm))

    * models are specified with formula syntax,

    * data is provided as a data frame, and

    * additional arguments are available to specify priors.

* in a nutshell, rstanarm let's you estimate various bayesian models and examine them with [shinystan](https://mc-stan.org/users/interfaces/shinystan).

## 2. A "Hello World" example

### rstan example
```
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

summary(lm(y~x))

# http://mc-stan.org/shinystan/articles/shinystan-package.html

library(shinystan)
my_sso <- launch_shinystan(fit)
```

What happened here?

* we defined `model_code`, in which we provided all the important information about estimation of the model; we could have save it in a `.stan` file;

* we created a sample dataset and saved it into a list `mydata`;

* we fitted the model with `stan`;

* we may compare it's results with a regular model estimated with `lm()`;

* we launched the `shinystan` GUI to look into the model.

### rstanarm example

Let's use the dataset from the previous section (rstan example).
```
# http://m-clark.github.io/workshops/bayesian/04_R.html#rstanarm

library(rstanarm)

mydata_df <- data.frame(y = y, x = x)
fit2 <- stan_glm(y ~ x, data = mydata_df, iter=1000, chains = 4)
fit2

library(shinystan)
launch_shinystan(fit2)
```

What happened here?

* we loaded `rstanarm` package;

* we created a `data.frame`, which contains our vectors from previous section;

* we fitted a generalised linear model with `glm()`

* and launched `shinystan` to view it in GUI.


## 3. Useful links

* [a very good introductory article](http://m-clark.github.io/workshops/bayesian/04_R.html#rstanarm)

* [example of least squares in rstan](http://thestatsgeek.com/2015/06/08/my-first-foray-with-stan/)

* [tutorial/documentation/vignette](https://cran.r-project.org/web/packages/rstanarm/vignettes/rstanarm.html)

* [other vignettes](http://mc-stan.org/rstanarm/articles/)

* [rstanarm + mixed effects](http://kemacdonald.com/materials/langcog_rstanarm_tutorial_sleep.nb.html)

* [datacamp course of rstanarm](https://www.datacamp.com/courses/bayesian-regression-modeling-with-rstanarm)

* [User-friendly Bayesian regression modeling: A tutorial with rstanarm and shinystan](http://www.tqmp.org/RegularArticles/vol14-2/p099/p099.pdf)

* [RStan](https://mc-stan.org/users/interfaces/rstan.html)

* [Doing Bayesian Data Analysis](http://www.users.csbsju.edu/~mgass/robert.pdf)

* https://www.kaggle.com/avehtari/bayesian-logistic-regression-with-rstanarm
