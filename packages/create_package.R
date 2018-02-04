#install.packages('devtools')
library(devtools)
#devtools::install_github('klutometis/roxygen')
library(roxygen2)

setwd('~/.R/my_libs')
getwd()
create('decisionTree')

setwd('./decisionTree')
getwd()

# w tym momencie tworzymy funkcje w folderze markovCluster/R

document()  # tworzy dokumentację


# instalacja biblioteki
setwd('..')
install('decisionTree')

