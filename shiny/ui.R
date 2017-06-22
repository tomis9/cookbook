# przykładowy program w shiny
# aby uruchomić aplikację: runApp()

library(shiny)

shinyUI(fluidPage(

  titlePanel("Hello Shiny!"),

  sidebarLayout(
    sidebarPanel(
      sliderInput("bins",
        "Number of bins:",
        min = 1,
        max = 50,
        value = 30),
      selectInput("select", label = h3("Select box"), 
        choices = as.list(1:12), 
        selected = 1)
    ),

    mainPanel(
      plotOutput("distPlot"),
      uiOutput("many")
    )
  )
))
