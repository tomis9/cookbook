library(archivist)

getTagsLocal(repoDir = "arepo")

ahistory()

help(package = "archivist")

showLocalRepo(repoDir = "./arepo")
showLocalRepo(repoDir = "./arepo", method = "tags")

shinySearchInLocalRepo("./arepo")

xx <- searchInLocalRepo("elo", "./arepo") %>%
  lapply(loadFromLocalRepo, repoDir = "./arepo", value = TRUE)

searchInLocalRepo("x", "./arepo")

mt <- lapply(1:10000, function(x) mtcars)
mts <- dplyr::bind_rows(mt)
write.csv(mts, 'mtcars.csv')
archivist::saveToRepo(mts, repoDir = "arepo", userTags = c("elo", "melo"))

mts2 <- loadFromLocalRepo("58af88f344d21cfd75491f5698ea3687", "./arepo", TRUE)

identical(mts, mts2)


model <- loadFromLocalRepo("f15a0aa13220546f9a20f98f0448c7", "~/models", TRUE)

searchInLocalRepo("2019-03-14", "~/models", fixed = FALSE)
dd <- showLocalRepo(repoDir = "~/models", method = "tags") %>%
  as_tibble() %>%
  filter(tag == "FR_biostenix") %>%
  arrange(desc(createdDate)) %>%
  top_n(1) %>%
  select(artifact) %>%
  pull()

library(dplyr)

as_tibble(a) %>%
  count(tag) %>%
  tail()
