# plyr - tutorial
# 2017-03-01

library(plyr)

d <- data.frame(year=c(rep(2014, 12), rep(2015, 12)),
                month=c(1:12, 1:12),
                revenue=rnorm(24, 1000, 300),
                cost=rnorm(24, 1000, 300))

# podstawowe funkcje to summarise i transform
ddply(d, "year", summarise, rev_mean=mean(revenue), cost_sd=sd(cost))

ddply(d, "year", transform, rev_mean=mean(revenue))

# jak widać argumentem funkcji jest d
ddply(d, "year", function (x) {
          m <- mean(x$revenue)
          s <- sd(x$revenue)
          sm <- (x$revenue - m) / s
          return(data.frame(sm=sm))})

