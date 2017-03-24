# ggplot2 - tutorial

library(ggplot2)

d <- data.frame(a=letters[1:10], b=1:10)

ggplot(d, aes(x=a, y=b)) + 
    geom_point()

# ciekawostka
library(plotly)

p <- ggplot(d, aes(x=a, y=b)) +
    geom_point()

ggplotly(p)
