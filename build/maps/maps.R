# tutorial z tworzenia map w R

# http://quantup.pl/2015/03/13/analiza-danych/wizualizacja-danych-na-mapach-w-r/
# download.file("http://www.gis-support.pl/downloads/wojewodztwa.zip", "wojewodztwa.zip") #ściągamy plik z shapefilem

library(sp)
library(rgdal)
 
poland.map <- readOGR(dsn="wojewodztwa", "wojewodztwa") 
summary(poland.map)
plot(poland.map)
# pomalowanie jednego województwa
plot(poland.map[poland.map$jpt_nazwa_ == "wielkopolskie", ], col = "blue", add = TRUE)
 
# dane punktowe - w tym wypadku straże pożarne
eur.sp <- read.csv("http://quantup.pl/dane/blog/straz-pozarna.csv")
coords <- data.frame(eur.sp$x1, eur.sp$x2)  #tworzymy df ze wspolrzednymi
names(coords) <- c("x", "y") 
data <- data.frame(eur.sp$name)  #ramka z danymi
 
poland.sp <- SpatialPointsDataFrame(coords, data, proj4string = CRS(proj4string(poland.map)), match.ID=TRUE)
 
plot(poland.sp, col='yellow')
plot(poland.map, add=TRUE)


library(rgeos)
 
#zwraca TRUE, jeśli współrzędne punktu są wewnątrz obszaru poland.map
int <- gIntersects(poland.sp, poland.map, byid = T) #może to chwilę potrwać
clipped <- apply(!int, MARGIN = 2, all)
 
plot(poland.map)
points(poland.sp[which(!clipped), ], col = "green")
points(poland.sp[which(clipped), ])  #wycięte punkty
poland.sp2 <- poland.sp[which(!clipped), ]

# agregacja danych
poland.aggr <- aggregate(x = poland.sp2["eur.sp.name"], by = poland.map, FUN = length)
#dodajmy interesujące nas dane do poland.map
poland.map@data$straze <- poland.aggr@data$eur.sp.name
head(poland.map@data, n=4)

q <- cut(poland.map@data$straze, 
         breaks= c(quantile(poland.map@data$straze)), 
         include.lowest=T, 
         dig.lab=nchar(max(poland.map@data$straze)))
clr <- as.character(factor(q, labels = paste0("grey", seq(80, 20, -20))))
plot(poland.map, col = clr)
title("Liczba jednostek SP w województwach")
legend(legend = paste0(levels(q)[1:4]), fill = paste0("grey", seq(80, 20, -20)), "bottomleft")

EPSG <- make_EPSG()
EPSG[grepl("WGS 84$", EPSG$note), ]
poland.map <- spTransform(poland.map, CRS("+init=epsg:4326")) #WGS-84

#argumentami jest obiekt przestrzenny oraz nazwa kolumny, która będzie nowym id
poland.map.gg <- fortify(poland.map, region="jpt_nazwa_")
head(poland.map.gg, n=2)

poland.map.gg <- merge(poland.map.gg, poland.map@data, by.x="id", by.y="jpt_nazwa_", sort=FALSE)
head(poland.map.gg, n=2) #mamy wszystkie dane

library(ggplot2)
map <- ggplot() +
    geom_polygon(data = poland.map.gg, 
                 aes(long, lat, group = group, 
                     fill = straze), 
                 colour = "black", lwd=0.1) +
ggtitle("Mapa Polski") +
labs(x = "E", y = "N", fill = "Liczba\njednostek SP")
map <- map + scale_fill_gradient(low = "white", high = "red")

library(plotly)
ggplotly(map)

# rysowanie na mapie z google maps
poland.bb <- bbox(poland.map)
poland.bb[1, ] <- poland.bb[1, ]*c(0.96, 1.04)
poland.bb[2, ] <- poland.bb[2, ]*c(0.96, 1.04)

library(ggmap) 
poland.img <- ggmap(get_map(location = poland.bb, source = 'google', maptype = 'hybrid')) #google jest domyslny, wiec nie trzeba podawać
poland.img <- poland.img + geom_polygon(data = poland.map.gg, aes(x = long, y = lat, group = group, fill = straze), alpha = 0.5,
colour = "black", lwd=0.1) + labs(x = "E", y = "N", fill = "Liczba\njednostek SP") + ggtitle("Mapa Polski") + scale_fill_gradient(low = "white", high = "red")

poland.img
