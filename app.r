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
library(randomForest)
library(dplyr)

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
                                     selectInput("knn_x_vars", "Select Independent Variables", choices = "", selected = "", multiple = TRUE),
                                     selectInput("knn_y_var", "Select Dependent Variable", choices = "", selected = ""),
                                     textInput("knn_prop", "Select Proportion", value = 0.8, placeholder = "Percentage of rows"),
                                     actionButton("btn_knn_predict", "Predict"),
                                     hr(),
                                     helpText("Variable selected must be categorical and numerical. Use '5. RF_abalone_short.csv' from datasets for testing."),
                                     hr(),
                                     a(href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm", "kNN")
                                   ),
                                   mainPanel(
                                     textOutput("knn_prediction_result"),
                                     plotOutput("knn_plot")
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
                                     
                                     hr(),
                                     a(href="https://en.wikipedia.org/wiki/Random_forest", "Random Forest")
                                   ),
                                   mainPanel(
                                     div(plotOutput("rfoutput"))
                                     
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
      updateSelectInput(session, inputId = "knn_x_vars", choices = names(data_input()))
      updateSelectInput(session, inputId = "knn_y_var", choices = names(data_input()))
    })
    
    observeEvent(input$btn_knn_predict, {
      req(input$knn_x_vars, input$knn_y_var, input$knn_prop)
      
      df <- clean_data()
      
      x_cols <- input$knn_x_vars
      y_col <- input$knn_y_var
      
      X <- df[, x_cols, drop = FALSE]
      y <- df[, y_col]
      
      prop <- as.numeric(input$knn_prop)
      
      # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
      set.seed(42)
      train_index <- sample(1:nrow(df), size = prop * nrow(df))
      
      X_train <- X[train_index, ]
      X_test <- X[-train_index, ]
      y_train <- y[train_index]
      y_test <- y[-train_index]
      
      # Kiểm tra kích thước của dữ liệu
      print(paste("Training data size:", nrow(X_train)))
      print(paste("Testing data size:", nrow(X_test)))
      
      # Huấn luyện mô hình KNN
      knn_model <- knn(train = X_train, test = X_test, cl = y_train, k = 11)
      
      # Đánh giá độ chính xác của mô hình
      accuracy <- mean(knn_model == y_test)
      
      # Hiển thị độ chính xác
      output$knn_prediction_result <- renderText({
        paste("Accuracy: ", round(accuracy * 100, 2), "%")
      })
      
      # Hiển thị biểu đồ dự đoán
      output$knn_plot <- renderPlot({
        plot_data <- data.frame(X_test, Predicted = knn_model)
        ggplot(plot_data, aes_string(x = names(X_test)[1], y = names(X_test)[2], color = "Predicted")) +
          geom_point() +
          labs(title = "KNN Predictions")
      })
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
    req(input$rfvar)
    df <- data_input()
    
    df[[input$rfvar]] <- as.factor(df[[input$rfvar]])
    
    # Kiểm tra và loại bỏ các lớp rỗng
    df <- df[df[[input$rfvar]] != "",]
    
    set.seed(123)  # Để kết quả có thể tái lập lại
    
    # Ensure data_set_size is calculated correctly
    data_set_size <- floor(0.7 * nrow(df))
    index <- sample(seq_len(nrow(df)), size = data_set_size)
    training <- df[index,]
    testing <- df[-index,]
    
    training <- droplevels(training)
    
    # Đảm bảo không có lớp rỗng trong tập huấn luyện
    if(any(table(training[[input$rfvar]]) == 0)) {
      stop("Tập huấn luyện có lớp rỗng. Vui lòng chọn biến khác hoặc làm sạch dữ liệu.")
    }
    
    rf <- randomForest(as.formula(paste(input$rfvar, "~ .")),
                       data = training,
                       mtry = 4,
                       ntree = 200,
                       importance = TRUE)
    rf
  })
  
  output$rfoutput <- renderPlot({
    plot(rfout())
  })
  
}


  


# Run the application 
shinyApp(ui = ui, server = server)
