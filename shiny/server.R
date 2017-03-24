# przykładowy program w shiny

library(shiny)
library(ggplot2)
library(plotly)

# Define server logic required to draw a histogram
shinyServer(function(input, output) {

  # Expression that generates a histogram. The expression is
  # wrapped in a call to renderPlot to indicate that:
  #
  #  1) It is "reactive" and therefore should re-execute automatically
  #     when inputs change
  #  2) Its output type is a plot

  output$distPlot <- renderPlot({
    x    <- faithful[, 2]  # Old Faithful Geyser data
    bins <- seq(min(x), max(x), length.out = input$bins + 1)

    # draw the histogram with the specified number of bins
    hist(x, breaks = bins, col = 'darkgray', border = 'white')
  })

  output$tab2plot <- renderPlotly({
      plot_ly(mtcars, x = ~mpg, y = ~wt)
#      d <- data.frame(a=letters[1:10], b=1:10)
#      p <- ggplot(d, aes(x=a, y=b)) +
#          geom_point()
#      ggplotly(p)
  })
})

