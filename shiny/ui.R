# przykładowy program w shiny

library(shiny)

# Define UI for application that draws a histogram
shinyUI(fluidPage(

  # Application title
  titlePanel("Hello Shiny!"),

  # Sidebar with a slider input for the number of bins
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

    # Show a plot of the generated distribution
      mainPanel(
                tabsetPanel(
                            tabPanel(
                            "Tab 1",
                            plotOutput("distPlot")),
                            tabPanel(
                                     "Tab 2",
                                     plotOutput("tab2plot"))
                            )
        )
     )
  ))
