dt_iris <- data.table::as.data.table(iris)
mltools::one_hot(dt_iris)

# dplyr is not that clever
library(dplyr)
iris %>%
  mutate("Species_setosa" = ifelse(Species == "setosa", 1, 0)) %>%
  mutate("Species_virgninica" = ifelse(Species == "virgninica", 1, 0)) %>%
  mutate("Species_versicolor" = ifelse(Species == "versicolor", 1, 0)) %>%
  head()
