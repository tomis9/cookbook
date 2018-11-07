real.roots <- function(a, b, c) {
    if (a == 0.)
        stop("Leading term cannot be zero")

    delta <- b * b - 4 * a * c # discriminant

    if (delta < 0)
       rr <- c()
    else if (delta == 0)
       rr <- c(-b / (2 * a))
    else
        rr <- c((-b - sqrt(delta)) / (2 * a),
                (-b + sqrt(delta)) / (2 * a))

    return(rr)
}
