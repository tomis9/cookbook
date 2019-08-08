library(class)

train_test_split <- function(test_proportion = 0.75, dataset) {
    smp_size <- floor(test_proportion * nrow(dataset))
    train_ind <- sample(seq_len(nrow(dataset)), size = smp_size)

    train <- dataset[train_ind, ]
    test <- dataset[-train_ind, ]
    return(list(train = train, test = test))
}
library(gsubfn)
list[train, test] <- train_test_split(0.5, iris)


class::knn(train[,1:4], test[,1:4], cl = train[,5], k = 3) == test[,5]

