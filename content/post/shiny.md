---
title: "shiny"
date: 2017-03-24T09:13:23+01:00
draft: true
categories: ["R"]
tags: ["draft"]
---

## A minimal application:

ui.R

```{r, eval = FALSE}
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
```

***
server.R

```{r, eval = FALSE}
# przykładowy program w shiny
# aby uruchomić aplikację: runApp("./shiny")

library(shiny)
library(ggplot2)
library(plotly)

inputBins <- 10
# Define server logic required to draw a histogram
shinyServer(function(input, output) {

    r <- reactiveValues()

    er <- observeEvent(input$bins, {
        r$inputBins <- input$bins
        output$distPlot <- renderPlot(f())
    })

    rysujPlot <- renderPlot({
        x    <- faithful[, 2]
        bins <- seq(min(x), max(x), length.out = inputBins + 1)
        hist(x, breaks = bins, col = 'darkgray', border = 'white')
    })

    f <- function() {
        x    <- faithful[, 2]
        inputBins <- r$inputBins
        bins <- seq(min(x), max(x), length.out = inputBins  + 1)
        hist(x, breaks = bins, col = 'darkgray', border = 'white')
    }

    output$many <- renderUI({
      lapply(1:10, function(i) {
        id <- paste0("p", i)
        numericInput(id, NULL, 0) 
      })
    })

})
```

