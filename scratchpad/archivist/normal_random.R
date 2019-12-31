#!/usr/bin/env Rscript
x <- rnorm(100)

archivist::saveToRepo(x, repoDir = "arepo", userTags = c("elo", "melo"))

archivist::
