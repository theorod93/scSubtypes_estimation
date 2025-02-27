#' Pre-processing to keep a single sequencing technology 
#' -> In this case the example is smartseq2)


# Libraries
library(data.table)

celltypes <- as.vector(as.matrix(fread("celltype_panc8.txt")))
dataset <- as.vector(as.matrix(fread("dataset_panc8.txt")))
celltypes_num <- as.vector(as.matrix(fread("celltype_num_panc8.txt")))
dataset_num <- as.vector(as.matrix(fread("dataset_num_panc8.txt")))

nan_index <- which(dataset == "smartseq2")

label_matrix <- matrix(0, nrow = length(celltypes_num), ncol = length(unique(celltypes_num)))
for (i in 1:length(celltypes_num)){
  label_matrix[i,celltypes_num[i]] <- 1
}

# NaN label matrix
nan_label_matrix <- label_matrix
nan_label_matrix[nan_index,] <- NA
zero_label_matrix <- label_matrix
zero_label_matrix[nan_index,] <- 0

celltypes_missing <- celltypes
celltypes_missing[nan_index] <- NA
celltypes_missing_num <- celltypes_num
celltypes_missing_num[nan_index] <- 0



write.table(nan_label_matrix, "nan_label_matrix.txt", col.names = F, row.names = F, quote = F)
write.table(zero_label_matrix, "zero_label_matrix.txt", col.names = F, row.names = F, quote = F)
write.table(label_matrix, "label_matrix.txt", col.names = F, row.names = F, quote = F)
write.table(nan_index, "nan_index.txt", col.names = F, row.names = F, quote = F)
write.table(celltypes_missing, "celltypes_missing.txt", col.names = F, row.names = F, quote = F)
write.table(celltypes_missing_num, "celltypes_missing_num.txt", col.names = F, row.names = F, quote = F)


