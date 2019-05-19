km <- kmeans(iris[,1:4], centers = 3)

iris$Species

plot(km$cluster, iris$Species)

# https://www.datacamp.com/community/tutorials/pca-analysis-r
pca <- prcomp(iris[,1:4])

summary(pca)
str(pca)




mtcars.pca <- prcomp(mtcars[,c(1:7,10,11)], center = TRUE, scale. = TRUE)
summary(mtcars.pca)

# library(devtools)
# install_github("vqv/ggbiplot")

library(ggbiplot)

ggbiplot(mtcars.pca)

pca <- prcomp(iris[,1:4], center = TRUE, scale. = TRUE)
ggbiplot(pca, obs.scale = 1, var.scale = 1, groups = iris$Species, 
         ellipse = TRUE, circle = TRUE)

# wymiarowość pca można interpretować na dwa sposoby: tak, jak na ggbiplot - strzałki, ale też jako zmniejszenie liczby zmiennych objasniających do dwóch (osie)
# ggbiplot jest kompatybilne z ggplot2
# http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/118-principal-component-analysis-in-r-prcomp-vs-princomp/

iris_pca <- scale(iris[,1:4]) %*% pca$rotation 
iris_pca <- as.data.frame(iris_pca)
iris_pca <- cbind(iris_pca, Species = iris$Species)

library(ggplot2)
ggplot(iris_pca, aes(x = PC1, y = PC2, color = Species)) +
  geom_point()

km0 <- kmeans(iris_pca[,1], centers = 3)
km1 <- kmeans(iris_pca[,1:2], centers = 3)
km2 <- kmeans(iris_pca[,1:4], centers = 3)

plot(fitted(km))

# https://stats.stackexchange.com/questions/31083/how-to-produce-a-pretty-plot-of-the-results-of-k-means-cluster-analysis

library(cluster)
library(fpc)

plotcluster(iris[,1:4], pca$cluster)

par(mfrow=c(1, 3))
plot(iris_pca[,1:2], col = km0$cluster, pch = as.integer(iris_pca$Species))
points(km2$centers, col = 1:2, pch = 8, cex = 2)
plot(iris_pca[,1:2], col = km1$cluster, pch = as.integer(iris_pca$Species))
points(km1$centers, col = 1:2, pch = 8, cex = 2)
plot(iris_pca[,1:2], col = km2$cluster, pch = as.integer(iris_pca$Species))
points(km2$centers, col = 1:2, pch = 8, cex = 2)

