# reshape2 - tutorial

library(reshape2)

d <- data.frame(konto=paste(rep(7, 9), 1:9, sep=""),
                styczen=rnorm(9, 10, 1),
                luty=rnorm(9, 10, 2),
                marzec=rnorm(9, 10, 3))

# na postać normalną
dn <- melt(d, "konto", variable.name="miesiac", 
           value.name="przychody")

# na pivot table
dp <- dcast(dn, konto~miesiac, value.var="przychody")
