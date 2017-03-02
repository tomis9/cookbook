# data.table - tutorial
# 2017-03-01

# podobno tu jest dobry tutorial
# https://www.datacamp.com/courses/data-analysis-the-data-table-way

library(data.table)

df <- data.frame(x=c("b","b","b","a","a"),
                 v=rnorm(5))

dt <- data.table(x=c("b","b","b","a","a"),
                 v=rnorm(5))

# podgląd wszystkich dostępnych data.tables - muszą się mieścić w RAMie
tables()

dt[x=="b",]  # tak mogę
df[x=="b",]  # a tak nie

# w tabeli warto ustalić klucz
setkey(dt, x)

# żeby można było zrobić na przykład tak
dt["b"]  # ta jakby dt[x=="b",], bo x jest domyślny


# JOIN
# inny przykład - duża tabela - uzasadnienie, że binary search jest 
# szybsze od vector scan
grpsize <- ceiling(1e8/26^2)
system.time(DF <- data.frame(x=rep(LETTERS, each=26*grpsize),
                             y=rep(letters, each=grpsize),
                             v=runif(grpsize*26^2),
                             stringsAsFactors=F))
system.time(ans1 <- DF[DF$x=='R' & DF$y=='h',])  # 'vector scan'

DT <- as.data.table(DF)
system.time(setkey(DT,x,y))
system.time(ans2 <- DT[list('R', 'h')])  # binary search
# list można zapisać jako .
system.time(ans2 <- DT[.('R', 'h')])  # binary search
# chociaż to, co wykonaliśmy wyżej, to tak naprawdę join z listą ('R', 'h'),
# czyli zawierającą jeden wiersz i dwie kolumny - jest to pierwszy argument
# i nazywa się i

identical(ans1$v, ans2$v)


# GROUP
DT[,sum(v)]
DT[,sum(v),by=x]

# zwykły group by po dwóch kolumnach: x i y
system.time(ss <- DT[,sum(v),by="x,y"])

