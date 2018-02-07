# sqldf - tutorial
# 2017-02-28

library(sqldf)

# zauważmy, że odpadły nazwy wierszy
mpgs <- sqldf('select * from mtcars where mpg > 21')
mpgs

cyls <- sqldf('select mpg, cyl, disp from mtcars where cyl = 6')
cyls

typeof(cyls)
class(cyls)

