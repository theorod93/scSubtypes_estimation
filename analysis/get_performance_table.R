#' In this script we will create a Latex table by using R and the values in a given matrix
#' 
#' Reference: https://sdaza.com/blog/2020/descriptive-tables/
#' 
#' 
#' 

# Libraries
library(data.table)
set.seed(14332)

# Prepare data
# Compare with benchmark solutions
benchmarkResults <- vector("list")
benchmarkResults$acc <- read.csv("allAccScores_df.csv")
benchmarkResults$prec <- read.csv("allPrecScores_df.csv")
benchmarkResults$rec <- read.csv("allRecScores_df.csv")
benchmarkResults$f1 <- read.csv("allF1Scores_df.csv")
# Combine all results
# Add a new column 
allResults <- vector("list", length = length(benchmarkResults))
for (i in 1:length(benchmarkResults)){
  allResults[[i]] <- cbind(benchmarkResults[[i]], rep(0,nrow(benchmarkResults[[i]])))
  colnames(allResults[[i]]) <-  c(colnames(benchmarkResults[[i]]), "S_multiSNE")
}
names(allResults) <- names(benchmarkResults)

## Run plots_compareSamples_mca_local_temp.R -- same directory ## 

# Add results:
for (i in 1:length(evaluationMeasuresKnn)){
  temp_name <- names(evaluationMeasuresKnn)[i]
  split_name <- strsplit(temp_name, "_")
  selected_row <- which((benchmarkResults$acc$Train == split_name[[1]][1]) & (benchmarkResults$acc$Test == split_name[[1]][2]))
  temp_f1_perp <- c()
  for (k in 1:length(evaluationMeasuresKnn[[i]])){
    temp_f1_perp <- c(temp_f1_perp, max(evaluationMeasuresKnn[[i]][[k]]$f1_score))
  }
  chosen_k <- which.max(temp_f1_perp)
  allResults$acc$S_multiSNE[selected_row] <- evaluationMeasuresKnn[[i]][[chosen_k]]$evalMeasures[1]
  allResults$prec$S_multiSNE[selected_row] <- evaluationMeasuresKnn[[i]][[chosen_k]]$evalMeasures[2]
  allResults$rec$S_multiSNE[selected_row] <- evaluationMeasuresKnn[[i]][[chosen_k]]$evalMeasures[3]
  allResults$f1$S_multiSNE[selected_row] <- max(evaluationMeasuresKnn[[i]][[chosen_k]]$f1_score)
}
# Remove rows with Train and Test sharing the same data-view
# Same instances on all measure-matrices
k<-c()
for (j in 1:nrow(allResults[[1]])){
  if (allResults[[1]]$Train[j] == allResults[[1]]$Test[j])  {
    k <- c(k,j)
  }
}
for (i in 1:length(allResults)){
  allResults[[i]] <- allResults[[i]][-k,]
}

# Keep only numeric
onlyNumericResults <- lapply(allResults, function(x) return(x[,-c(12,13)]))
lapply(onlyNumericResults, colMeans)


library(data.table)

# Input 
datasets = list("Data" = t(onlyNumericResults$prec))
variables = list(rownames(onlyNumericResults$prec))
labels = list(rownames(onlyNumericResults$prec))
colnames =  colnames(onlyNumericResults$prec)[-1]


# Define descriptive function
myDescriptives = function(x) {
  x = as.numeric(x)
  m = mean(x, na.rm = TRUE)
  md = median(x, na.rm = TRUE)
  sd = sd(x, na.rm = TRUE)
  return(c(m, md, sd))
}

myDescriptives <- function(x){return(x[-1])}
# create table
createDescriptiveTable(datasets,
                       summary_function = myDescriptives,
                       column_names = colnames,
                       variable_names = variables,
                       variable_labels = labels,
                       arraystretch = 1.3,
                       title = "Descriptive statistics",
                       label = "tab:descriptive",
                       file = "example_01.tex")
