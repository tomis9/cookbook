# d <- as.matrix(iris[,1:4])
d <- matrix(1:4, 2)
# for (i in 1:5) d <- rbind(d, d)
n <- nrow(d)

X <- t(d)
phi <- 0.5
dyn.load("fun.so")
storage.mode(X) <- "double"
p <- as.integer(nrow(X))
n <- as.integer(ncol(X))
phi <- as.double(phi)
w <- double(n*(n-1)/2)
system.time(sol <- .C('fun',X=X,p=p,n=n,phi=phi,w=w))
weights=sol$w
weights

gaussianKernel <- function(x, y) exp(-sum((x - y) ^ 2) / 2) # phi to / 2
A <- matrix(0, nrow=n, ncol=n)
system.time(
for (i in 1:n) {
    for (j in 1:n) {
        A[i, j] <- gaussianKernel(d[i, ], d[j, ])    
    }
}
)
A
