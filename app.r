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
library(rpart)
library(rpart.plot)

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
                                     hr(),
                                     a(href="http://mlwiki.org/index.php/Decision_Tree_Exercises", "Decision Trees")
                                   ),
                                   mainPanel(
                                     div(plotOutput("dtplot"))     
                                   )
                                 )
                        ),
                        tabPanel("Random Forests", 
                                 sidebarLayout(
                                   sidebarPanel(
                                     selectInput("rfvar", "Select Variable", choices = "", selected = ""),
                                     
                                     textInput("rfprop", "Select Proportion", value = 0.8, placeholder = "Percentage of rows"),
                                     textInput("rfyname", "Class Variable", value = "old", placeholder = "Class Variable"),
                                     radioButtons("rfoption", "Select Method", choices = c("No Option", "Table", "Show Prop.", "Train & Test Data", "Fit", "Summary", "Predicted", "Pred. Accuracy")),
                                     hr(),
                                     helpText("Variable selected must be categorical and numerical. Use '5. RF_abalone_short.csv' from datasets for testing."),
                                     hr(),
                                     a(href="https://en.wikipedia.org/wiki/Random_forest", "Random Forest")
                                   ),
                                   mainPanel(
                                     div(verbatimTextOutput("rfoutput"))
                                     
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
    
    
    
    
    predictions <- reactive({
      inFile <- input$file1
      
      if (is.null(inFile)) {
        return(NULL)
      } else {
        withProgress(message = 'Predictions in progress. Please wait ...', {
          input_data <- readr::read_csv(input$file1$datapath, col_names = TRUE)
          
          colnames(input_data) <- c("Test1", "Test2", "Label")
          
          # Đảm bảo Label là factor với levels giống như đã huấn luyện mô hình
          input_data$Label <- factor(input_data$Label, levels = levels(my_model$Label))
          
          mapped <- feature_mapping(input_data)
          df_final <- cbind(input_data, mapped)
          prediction <- predict(my_model, df_final)
          
          input_data_with_prediction <- cbind(input_data, prediction)
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
    observeEvent(input$file1, {
      updateSelectInput(session, inputId = "dtvar1", choices = names(data_input()))
      updateSelectInput(session, inputId = "feature1", choices = names(data_input()))
      updateSelectInput(session, inputId = "feature2", choices = names(data_input()))
    }
    )
    
    knnout <- reactive({
      df <- clean_data()
      
      # Chia dữ liệu thành tập huấn luyện và kiểm tra
      set.seed(123)
      size <- floor(0.8 *  nrow(df))
      train_ind <- sample(seq_len(nrow(df)), size = size)
      train_labels <- df[train_ind, input$dtvar1]
      data_train <- df[train_ind, c(input$feature1, input$feature2 )]
      data_test <- df[-train_ind, c(input$feature1, input$feature2 )]
      data_test_labels <- df[-train_ind, input$dtvar1]
      
      # Fit KNN Model
      predictions <- knn(train = data_train,
                         test = data_test,
                         cl = train_labels,
                         k = 11)
      
      # Prepare data for plotting predictions
      plot_predictions <- data.frame(
        data_test,
        predicted = predictions
      )
      
      colnames(plot_predictions) <- c(input$feature1, input$feature2, "predicted")
      
      # Visualize the KNN algorithm results.
      p1 <- ggplot(plot_predictions, aes_string(x = input$feature1, y = input$feature2, color = "predicted", fill = "predicted")) + 
        geom_point(size = 5) + 
        geom_text(aes_string(label = data_test_labels), hjust = 1, vjust = 2) +
        ggtitle("Predicted relationship between Feature 1 and Feature 2") +
        theme(plot.title = element_text(hjust = 0.5)) +
        theme(legend.position = "none")
      
      # Sắp xếp các đồ thị cạnh nhau
      p1
    })
    
    output$knnoutput <- renderPlot({
      knnout()
    })
  
  # DECISION TREE
  
  
  gender_tree <- reactive({
    req(input$file1)  # Đảm bảo rằng đã có file được tải lên
    data <- data_input()
    
    # Sử dụng tất cả các biến trong file để tạo cây quyết định
    formula <- as.formula(paste(names(data)[ncol(data)], "~ ."))  # Chọn biến cuối cùng làm biến mục tiêu
    rpart(formula, data = data)
  })
  
  output$dtplot <- renderPlot({
    req(input$file1)  # Đảm bảo rằng đã có file được tải lên
    
    # Vẽ đồ thị cây quyết định
    rpart.plot(gender_tree(), type = 3, extra = 101)
  })
  
  # RANDOM FOREST
  
  observeEvent(input$file1, {
    updateSelectInput(session, inputId = "rfvar", choices = names(data_input()))
  }
  )
  
  rfout <- reactive({
    df <- data_input()
    
    if (input$rfoption == "Table"){
      return(table(df[, input$rfvar]))
    }
    
    # train_index <- sample(1:nrow(df), as.numeric(input$rfprop) * nrow(df))
    
    prop <- as.numeric(input$rfprop)
    
    train_set <- df[1:(nrow(df)*prop),]
    test_set <- df[-(1:(nrow(df)*prop)),]
    
    
    if (input$rfoption == "Show Prop."){
      return(dim(train_set)[1]/dim(df)[1])
    }
    
    
    if (input$rfoption == "Train & Test Data"){
      return(list(head(train_set), head(test_set), dim(train_set), dim(test_set)))
    }
    
    var <- input$rfvar
    
    rf_fit <- randomForest::randomForest(as.formula(paste(var, "~", ".")), data = train_set, importance = TRUE, proximity = TRUE)
    
    if (input$rfoption == "Fit"){
      return(rf_fit)
    }
    
    if (input$rfoption == "Summary"){
      return(summary(rf_fit))
    }
    
    rf_pred <- predict(rf_fit, test_set)
    
    out <- confusionMatrix(round(rf_pred), as.numeric(test_set[, input$rfvar]))
    
    if (input$rfoption == "Predicted"){
      return(data.frame(rf_pred))
    }
    
    if (input$rfoption == "Pred. Accuracy"){
      return(out)
    }
    
    # return(out)
  })
  
  output$rfoutput <- renderPrint({
    rfout()
  })
  
}


  


# Run the application 
shinyApp(ui = ui, server = server)
