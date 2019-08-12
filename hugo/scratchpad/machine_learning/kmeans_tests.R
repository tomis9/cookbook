# devtools::install_github("vqv/ggbiplot")

pca <- prcomp(iris[,1:4], center = TRUE, scale. = TRUE)
ggbiplot::ggbiplot(pca, obs.scale = 1, var.scale = 1, groups = iris$Species, 
                   ellipse = TRUE, circle = TRUE)

iris_pca <- scale(iris[,1:4]) %*% pca$rotation 
iris_pca <- as.data.frame(iris_pca)
iris_pca <- cbind(iris_pca, Species = iris$Species)

ggplot2::ggplot(iris_pca, aes(x = PC1, y = PC2, color = Species)) +
  geom_point()

plot_kmeans <- function(km, iris_pca) {
  plot(iris_pca[,1:2], col = km$cluster, pch = as.integer(iris_pca$Species))
  points(km$centers, col = 1:2, pch = 8, cex = 2)
}
par(mfrow=c(1, 3))
sapply(list(kmeans(iris_pca[,1], centers = 3),
            kmeans(iris_pca[,1:2], centers = 3),
            kmeans(iris_pca[,1:4], centers = 3)),
       plot_kmeans, iris_pca = iris_pca)
