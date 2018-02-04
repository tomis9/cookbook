# Programowanie w języku R, s. 463-488
# tworzenie nowej klasy S4
setClass('Wielokat', 
         slots = c(x='numeric', y='numeric'),
         )

# nowy obiekt
w1 <- new('Wielokat', x=1:3, y=c(1, 1, 2))

slotNames(w1)
w1@x  # do wywoływania atrybutów nie służy ".", tylko "@"
w1@y

# w1@x <- 4:10

# dziedziczenie
setClass('Trojkat',
         contains='Wielokat',
         validity=function(object) {
             if (length(object@x) != 3) {
                 return('nalezy podac 3 wierzcholki')
             }
             TRUE
         })

t1 <- new('Trojkat', x=c(1, 2, 3), y=c(1, 1, 3))
class(t1)
new('Trojkat', x=c(1, 2, 3, 4), y=c(1, 1, 3, 4))  # error

# metody
# przeciążenie print, które nazywa się show (żeby było trudniej)
setMethod('show',
          signature=c(object='Wielokat'),
          function(object) {
              cat(sprintf('To jest %d-kat o wierzchołkach: ', length(object@x)))
              cat(sprintf('(%g, %g)', object@x, object@y))
              cat('\n')
          })
print(w1)

# pythonowe __init__
setClass("A", representation(a="numeric"), "VIRTUAL")
setClass("B", representation(b="numeric"), contains="A")

setMethod("initialize", "A", function(.Object, data){
  .Object@a <- data[1]
  .Object
})

setMethod("initialize", "B", function(.Object, data){
  .Object@b <- data[2]
  .Object <- callNextMethod(.Object, data)
  .Object
})

Cl <- setClass("Cl", representation(a="data.frame", b="list"))
myCl <- Cl(a=data.frame(a=1:3, b=letters[1:3]))
