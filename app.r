library(shiny)
library(shinydashboard)        
library(nortest)
library(mvnormtest)
library(MASS)
library(shinyLP)
library(class)
library(gmodels)
library(caret)
library(rattle)
library(ranger)
library(klaR)
library(kernlab)
library(e1071)
library(NeuralNetTools)
library(neuralnet)
library(nnet)
library(mclust)
library(ggplot2)
library(LiblineaR)
library(readr)

load("RegularizedLogisticRegression.rda")    # Load saved model
load("RegularizedLogisticRegression.rda")    # Load saved model

source("featureMapping.R")                         #  a function for feature engineering. 
#  You can include data imputation, data manipulation, data cleaning,
#  feature transformation, etc.,  functions



# Define UI for application that draws a histogram
ui <- fluidPage(
  navbarPage(title = "ML",
             tabPanel("Home",
                      jumbotron("Hi Welcome ", paste("This is web application for Machine Learning ! ") )
             ),
             tabPanel("Data",
                          sidebarLayout(
                            sidebarPanel(
                              fileInput("file1", "Choose CSV File", accept=c('text/csv', 'text/comma-separated-values', 'text/plain', '.csv')),
                              radioButtons("indata", "Choice:", choices = c("Full", "Columns")),
                              radioButtons("disp", "Display",
                                           choices = c(Head = "head",
                                                       All = "all"),
                                           selected = "head"),
                              hr(),
                              radioButtons("plotoption", "Choose the Option:", choices = c("Histogram", "BarPlot", "Scatter", "Pie" )),
                              selectInput("cols6", "Choose Varibale 1:", choices = "", selected = " ", multiple = TRUE),
                              textInput("xaxisname", "Write X Axis Name"),
                              textInput("yaxisname", "Write Y Axis Name"),
                              textInput("title", "Write Title For the Graph")
                              
                            ), 
                            
                            mainPanel(tableOutput("tab1"),
                                      h3("Plots"),
                                      fluidRow(
                                        plotOutput("plot")
                                      ))
                          )
             ),
             navbarMenu("Model",
                        tabPanel("Mutiple Linear Regression",
                                 sidebarLayout(
                                   sidebarPanel(
                                     
                                     selectInput("linrvar", "Select Variable", choices = "", selected = ""),
                                   ),
                                   mainPanel(
                                     div(verbatimTextOutput("linearout"))
                                   )
                                 )
                                 ),
                        
                        tabPanel("Logistic Reg.",
                                 
                                 sidebarLayout(
                                   sidebarPanel(
                                     hr(),
                                     a(href="http://mlwiki.org/index.php/Logistic_Regression", "Logistic Regression")
                                   ),
                                   mainPanel(
                                     column(width = 7,
                                            plotOutput('plot_predictions')
                                     ),
                                     column(width = 4,
                                            uiOutput("sample_prediction_heading"),
                                            tableOutput("sample_predictions")
                                     )
                                   )
                                 )
                        ),
                        
                        tabPanel("kNN",
                                 sidebarLayout(
                                   sidebarPanel(
                                     textInput("knntrain", "Select Proportion", value = 0.8, placeholder = "Percentage of rows"),
                                     radioButtons("knnoption", "Select Option", choices = c("Show Prop.", "Show Train & Test Data", "Show No. of Classes", "Fit", "Accuracy")),
                                     hr(),
                                     helpText("First column of data set must be categorical/use '2. kNN_sepsis_numerical' from datasets for testing."),
                                     a(href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm", "kNN")
                                   ),
                                   mainPanel(
                                     div(verbatimTextOutput("knnoutput"))
                                   )
                                 )
                        ),
                        tabPanel("Decision Trees",
                                 sidebarLayout(
                                   sidebarPanel(
                                     selectInput("dtvar", "Select Variable", choices = "", selected = ""),
                                     selectInput("dtvar2", "Select Variable", choices = "", selected = "", multiple = TRUE),
                                     textInput("dtprop", "Select Proportion", value = 0.8, placeholder = "Percentage of rows"),
                                     textInput("dtyname", "Class Variable", value = "num", placeholder = "Class Variable"),
                                     radioButtons("dtoption", "Select Method", choices = c("No Option", "Table", "Show Prop.", "Train & Test Data", "Fit", "Predicted", "Pred. Accuracy")), 
                                     radioButtons("dtplot", "Select Plot", choices = c("No Plot", "QPlot", "DTree")),
                                     hr(),
                                     helpText("Variable selected must be categorical and numerical. Use '4. DT_breast_cancer.csv' from datasets for testing."),
                                     hr(),
                                     a(href="http://mlwiki.org/index.php/Decision_Tree_Exercises", "Decision Trees")
                                   ),
                                   mainPanel(
                                     div(verbatimTextOutput("dtoutput")),
                                     div(plotOutput("dtplot"))     
                                   )
                                 )
                        )
             ),
  
  
  )
  
)

# Define server logic required to draw a histogram
server <- function(input, output, session) {

  data_input <- reactive({
    infile <- input$file1
    req(infile)
    data.frame(read.csv(infile$datapath)) 
  })

# Function to clean data
  clean_data <- reactive({
    df <- data_input()
    
    # Remove rows containing NA values
    df <- na.omit(df)
    # Remove duplicate rows
    df <- unique(df)
    
    # Convert character columns to factors if needed
    char_columns <- sapply(df, is.character)
    df[char_columns] <- lapply(df[char_columns], as.factor)
    
    # Convert numerical columns to numeric if needed
    num_columns <- sapply(df, is.numeric)
    df[num_columns] <- lapply(df[num_columns], as.numeric)
    
    return(df)
  })
  
  output$tab1 <- renderTable({
    df <- clean_data() # Use cleaned data for display
  })
  
  observeEvent(input$file1,{
    updateSelectInput(session, inputId = "cols", choices = names(data_input()))
  }
  )
  
  
  
  
  output$tab1 <- renderTable(
    {
      df <- clean_data() # Use cleaned data for display
      if (input$indata == "Full"){
        print(df)
      } else if(input$trans1 == "Not-Required"){
        data <- df[, input$cols]
        print(data)
      } else if(input$trans1 == "log"){
        logdata()
        
      } else if(input$trans1 == "inverselog"){
        invlogdata()
      } else if(input$trans1 == "exponential"){
        expdata()
      } else if(input$trans1 == "lognormal"){
        logno()
      } else if(input$trans1 == "standardize"){
        standout()
      }
      if(input$disp == "head") {
        return(head(df))
      }
      else {
        return(df)
      }
      
    }
  )
  #plot
  observeEvent(input$file1, {
    updateSelectInput(session, inputId = "cols6", choices = names(data_input()))
  }
  )
  
  output$plot <- renderPlot({
    df <- data_input()
    if(input$plotoption == "Histogram"){
      hist(df[, input$cols6], freq = FALSE, xlab = input$xaxisname, ylab = input$yaxisname, main = input$title); lines(density(df[, input$cols6]), col = "red", lwd = 1.5)
    } else if(input$plotoption == "BarPlot"){
      barplot(df[, input$cols6], xlab = input$xaxisname, ylab = input$yaxisname, main = input$title)
    } else if(input$plotoption == "Scatter"){
      scatter.smooth(df[, input$cols6], xlab = input$xaxisname, ylab = input$yaxisname, main = input$title)
    } else {
      pie(table(df[, input$cols6]))
    }
  })
  
  #linear regression
  observeEvent(input$file1, {
    updateSelectInput(session, inputId = "linrvar", choices = names(data_input()))
  })
  
  linearout <- reactive({
    df <- data_input()  
    var <- input$linrvar  
    train_prop <- as.numeric(input$linrprop)  
    
    Train <- createDataPartition(df[, var], p=train_prop, list=FALSE)
    training <- df[Train, ]
    testing <- df[-Train, ]
    
    train_ratio <- nrow(training) / (nrow(testing) + nrow(training))
    
    mod_fit <- lm(as.formula(paste(var, "~ .")), data=training)
    
    coefficients <- coef(mod_fit)
    
    if (input$linroption == "Show Prop.") {
      return(train_ratio)
    } else if (input$linroption == "Fit") {
      return(mod_fit)
    } else if (input$linroption == "Coef.") {
      return(data.frame(coefficients))
    } else if (input$linroption == "Pred. Accuracy") {
      pred_out <- predict(mod_fit, newdata=testing)
      mse <- mean((pred_out - testing[, var])^2) 
      return(mse)
    }
  })
  output$linear <- renderPlot({
    df <- data_input()  
    
    if (is.null(df) || nrow(df) == 0) {  
      return(NULL)
    }
    
    if (!input$x_col %in% names(df) || !input$y_col %in% names(df)) {  
      return(NULL)
    }
    
    x_var <- df[, input$x_col]  
    y_var <- df[, input$y_col]  
    
    if (!is.numeric(y_var)) {  
      return(NULL)
    }
    
    lm_model <- lm(y_var ~ x_var)  
    
    plot(x_var, y_var, xlab = input$xaxisname, ylab = input$yaxisname, main = input$title)
    abline(lm_model, col = "red") 
  })
  
  

  
  # Logistic Regression
  
  
    
    #options(shiny.maxRequestSize = 800*1024^2)   # This is a number which specifies the maximum web request size, 
    # which serves as a size limit for file uploads. 
    # If unset, the maximum request size defaults to 5MB.
    # The value I have put here is 80MB
    
    
    output$sample_input_data_heading = renderUI({   # show only if data has been uploaded
      inFile <- input$file1
      
      if (is.null(inFile)){
        return(NULL)
      }else{
        tags$h4('Sample data')
      }
    })
    
    output$sample_input_data = renderTable({    # show sample of uploaded data
      inFile <- input$file1
      
      if (is.null(inFile)){
        return(NULL)
      }else{
        input_data =  readr::read_csv(input$file1$datapath, col_names = TRUE)
        
        colnames(input_data) = c("Test1", "Test2", "Label")
        
        input_data$Label = as.factor(input_data$Label )
        
        levels(input_data$Label) <- c("Failed", "Passed")
        head(input_data)
      }
    })
    
    
    
    predictions<-reactive({
      
      inFile <- input$file1
      
      if (is.null(inFile)){
        return(NULL)
      }else{
        withProgress(message = 'Predictions in progress. Please wait ...', {
          input_data =  readr::read_csv(input$file1$datapath, col_names = TRUE)
          
          colnames(input_data) = c("Test1", "Test2", "Label")
          
          input_data$Label = as.factor(input_data$Label )
          
          levels(input_data$Label) <- c("Failed", "Passed")
          
          mapped = feature_mapping(input_data)
          
          df_final = cbind(input_data, mapped)
          prediction = predict(my_model, df_final)
          
          input_data_with_prediction = cbind(input_data,prediction )
          input_data_with_prediction
          
        })
      }
    })
    
    
    output$sample_prediction_heading = renderUI({  # show only if data has been uploaded
      inFile <- input$file1
      
      if (is.null(inFile)){
        return(NULL)
      }else{
        tags$h4('Sample predictions')
      }
    })
    
    output$sample_predictions = renderTable({   # the last 6 rows to show
      pred = predictions()
      head(pred)
      
    })
    
    
    output$plot_predictions = renderPlot({   # the last 6 rows to show
      pred = predictions()
      cols <- c("Failed" = "red","Passed" = "blue")
      ggplot(pred, aes(x = Test1, y = Test2, color = factor(prediction))) + geom_point(size = 4, shape = 19, alpha = 0.6) +
        scale_colour_manual(values = cols,labels = c("Failed", "Passed"),name="Test Result")
      
    })
  # KNN
  
  knnout <- reactive({
    
    df <- data_input()
    
    rows <- round(as.numeric(input$knntrain)*dim(df)[1])
    
    if(input$knnoption == "Show Prop."){
      return(rows)  
    }
    
    lascol <- dim(df)[2]
    
    train_data <- df[1:rows, 2:lascol]
    test_data <- df[-(1:rows), 2:lascol]
    
    if(input$knnoption == "Show Train & Test Data"){
      return(list(head(train_data), head(test_data)))
    }
    
    train_labels <- df[1:rows, 1]
    test_labels <- df[-(1:rows), 1]
    
    k_ = round(sqrt(dim(df)[1]))
    
    if(input$knnoption == "Show No. of Classes"){
      return(k_)
    }
    
    fit <- knn(train = train_data, test = test_data, cl = train_labels, k = k_)
    
    if(input$knnoption == "Fit"){
      return(data.frame(fit))
    }
    
    out <- CrossTable(x = test_labels, y = fit, prop.chisq = FALSE)
    
    if(input$knnoption == "Accuracy"){
      return(out)
    }
    
  })
  
  output$knnoutput <- renderPrint({
    if(input$knnoption == "Show Prop."){
      knnout()
    } else if(input$knnoption == "Show Train & Test Data"){
      knnout()
    } else if(input$knnoption == "Show No. of Classes"){
      knnout()
    } else if(input$knnoption == "Fit"){
      knnout()
    } else if(input$knnoption == "Accuracy"){
      knnout()
    }
    
  })
  
  # DECISION TREE
  
  observeEvent(input$file1, {
    updateSelectInput(session, inputId = "dtvar", choices = names(data_input()))
    updateSelectInput(session, inputId = "dtvar2", choices = names(data_input()))
  }
  )
  
  dtout <- reactive({
    
    df <- data_input()
    tab = table(df[, input$dtvar])
    
    if (input$dtoption == "Table"){
      return(tab)
    }
    
    index = createDataPartition(y=df[, input$dtvar], p=0.7, list=FALSE)
    
    train.set = df[index,]
    test.set = df[-index,]
    
    if (input$dtoption == "Train & Test Data"){
      return(list(head(train.set), head(test.set)))
    }
    
    if (input$dtoption == "Show Prop."){
      return(dim(train.set))
    }
    
    var <- input$dtvar
    
    brest.tree = train(as.formula(paste(var, "~", ".")),
                       data=train.set,
                       method="rpart",
                       trControl = trainControl(method = "cv"))
    
    if (input$dtplot == "QPlot"){
      
      plot(brest.tree$finalModel, uniform=TRUE, main="Classification Tree"); text(brest.tree$finalModel, use.n.=TRUE, all=TRUE, cex=.8)
    }
    
    
    if (input$dtoption == "Fit"){
      return(brest.tree)
    }
    
    
    if (input$dtplot == "DTree"){
      fancyRpartPlot(brest.tree$finalModel)
    }
    
    pred <- predict(brest.tree, test.set)
    out <- confusionMatrix(pred, test.set[, "Class_num"])
    
    if (input$dtoption == "Predicted"){
      return(data.frame(pred))
    }
    
    if (input$dtoption == "Pred. Accuracy"){
      return(out)
    }
    
  })
  
  output$dtoutput <- renderPrint({
    dtout()
  })
  
  output$dtplot <- renderPlot({
    if (input$dtplot == "QPlot"){
      dtout()
    } else if (input$dtplot == "DTree"){
      dtout()
    } else if (input$dtoption == "Pred. Accuracy"){
      dtout()
    } else if (input$dtoption == "Predicted"){
      dtout()
    }
  })  
  
}

# Run the application 
shinyApp(ui = ui, server = server)
