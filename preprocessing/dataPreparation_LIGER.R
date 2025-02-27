#' Filename: dataPreparation_LIGER.R
#' Author: Theodoulos Rodosthenous
#' Date: 26-02-2025
#' Description: Pre-processing of scRNA-seq data via LIGER
#' This script was used in the batch correction section of the manuscript
#' 
#' In most of the analysis, the pre-processing and integration was performed 
#' by Seurat. 
#' Here, LIGER is used instead
#' 
#' This script describes the process between two samples (mouse1 and mouse2), 
#' but the exact same solution can be implemented for any case described in the
#' manuscript





#### Libraries ####
library(Seurat)
library(SeuratData)
library(SeuratDisk)
library(data.table)
library(tidyverse)
library(dplyr)

#### Data ####
mouse1 <- read.table("Lung1_rm.batch_dge.txt", header=T, sep = " ")#, fill=T)
mouse2 <- read.table("Lung2_rm.batch_dge.txt", header=T, sep = " ")#, fill=T)

# Assignments -- Annotations 
mca_anno <- read.csv("MCA_CellAssignments.csv")
mca_info <- read_excel("MCA_Batch Information.xlsx")
mca_merge <- read.csv("MCA_BatchRemoved_Merge_dge_cellinfo.csv")


# Prepare
row_names_mouse1 <- rownames(mouse1)
row_names_mouse2 <- rownames(mouse2)
# Keep the same genes
genes_common <- row_names_mouse1[which(row_names_mouse1 %in% row_names_mouse2)]

mouse1 <- mouse1[which(row_names_mouse1%in%genes_common),]
mouse2 <- mouse2[which(row_names_mouse2%in%genes_common),]

# New rownames
new_rownames_mouse1 <- row_names_mouse1[which(row_names_mouse1%in%genes_common)]
new_rownames_mouse2 <- row_names_mouse2[which(row_names_mouse2%in%genes_common)]


# Get annotations
mouse1_anno <- mca_anno[which(mca_anno$Batch == "Lung_1" |
                                mca_anno$ClusterID == "Lung_1"  ),]
mouse2_anno <- mca_anno[which(mca_anno$Batch == "Lung_2"|
                                mca_anno$ClusterID == "Lung_2" ),]

# Samples with unknown labels, are set as "Unknown" cell-type

### Mouse1
## A.
common_mouse1_names <- colnames(mouse1)[which(colnames(mouse1) %in% mouse1_anno$Cell.name)]
mouse1_A <- mouse1[,which(colnames(mouse1) %in% common_mouse1_names)]
mouse1_A_anno <- mouse1_anno[which(mouse1_anno$Cell.name %in% common_mouse1_names),]
# Order 
mouse1_A <- mouse1_A[,order(colnames(mouse1_A))]
mouse1_A_anno <- mouse1_A_anno[order(mouse1_A_anno$Cell.name),]
mouse1_A_labels <- mouse1_A_anno$Annotation
### mouse2
## A.
common_mouse2_names <- colnames(mouse2)[which(colnames(mouse2) %in% mouse2_anno$Cell.name)]
mouse2_A <- mouse2[,which(colnames(mouse2) %in% common_mouse2_names)]
mouse2_A_anno <- mouse2_anno[which(mouse2_anno$Cell.name %in% common_mouse2_names),]
# Order 
mouse2_A <- mouse2_A[,order(colnames(mouse2_A))]
mouse2_A_anno <- mouse2_A_anno[order(mouse2_A_anno$Cell.name),]
mouse2_A_labels <- mouse2_A_anno$Annotation

#### Pre-processing implementation ####
# Select input
mouse1_inp <- mouse1_A
mouse2_inp <- mouse2_A
mouse1_inp_anno <- mouse1_A_anno
mouse2_inp_anno <- mouse2_A_anno


# Find genes to use
# The distinction was already done in firstLook....R
# (Skip this part)
genes.use <- rownames(mouse1_inp)

## Stage I: Preprocessing and Normalization (3 - 5 seconds)
# Next we create a LIGER object with raw counts data from each batch.
ob.list <- list("mouse1" = mouse1_inp, "mouse2" = mouse2_inp)
data.liger <- createLiger(ob.list) 

# Normalize gene expression for each batch.
data.liger <- rliger::normalize(data.liger)

# Variable gene selection
data.liger <- selectGenes(data.liger, var.thresh = 0.1)

# Scale the gene expression across the datasets. 
# Why does LIGER not center the data? Hint, think about the use of 
# non-negative matrix factorization and the constraints that this imposes.
data.liger <- scaleNotCenter(data.liger)

# Use alternating least squares (ALS) to factorize the matrix.
#k.suggest.value <- 20  # with this line, we do not use the suggested k by suggestK()#
#k.suggest <- suggestK(data.liger)
k.suggest.value <- 20
data.liger <- optimizeALS(data.liger, k = k.suggest.value, rand.seed = 1) 

# Let's see what the integrated data looks like mapped onto a tSNE visualization.
data.liger <- runTSNE(data.liger, use.raw = T)
p <- plotByDatasetAndCluster(data.liger, return.plots = T)
print(p[[1]])  # plot by dataset

## Stage II: Joint Matrix Factorization (3 - 10 minutes)
# Next, do clustering of cells in shared nearest factor space, and then quantile alignment.
data.liger <- quantileAlignSNF(data.liger, resolution = 1.0) # SNF clustering and quantile alignment

## Stage III: Quantile Normalization and Joint Clustering (1 minute)
data.liger <- quantile_norm(data.liger)

## Stage IV: Visualization (2 - 3 minutes) and Downstream Analysis (25 - 40 seconds)
data.liger <- runUMAP(data.liger, distance = 'cosine', n_neighbors = 30, min_dist = 0.3)

# Visualize liger batch correction results.
data.liger <- runTSNE(data.liger)

#### Extract data
data.seurat <- ligerToSeurat(data.liger)

## Annotations ##
labels_names <- c(mouse1_inp_anno$Annotation, mouse2_inp_anno$Annotation)
labels_num <- as.numeric(as.factor(labels_names))
# Get NaN
nan_index <- seq(1,length(labels_names),1)
non_nans <- ncol(mouse1_inp)
nan_index <- nan_index[-(1:non_nans)] # Remove the first non_nans rows; the remaining should be NAN
# Label matrix:
label_matrix <- matrix(0, nrow = length(labels_num), ncol = length(unique(labels_num)))
for (i in 1:length(labels_num)){
  label_matrix[i,labels_num[i]] <- 1
}
nan_label_matrix <- label_matrix
nan_label_matrix[nan_index,] <- NA
zero_label_matrix <- label_matrix
zero_label_matrix[nan_index,] <- 0

lung_comp_data <- vector("list")
lung_comp_data$data <- as.matrix(GetAssayData(data.seurat), slot ="data")
lung_comp_data$labels_names <- labels_names
lung_comp_data$labels_num <- labels_num
lung_comp_data$nan_index <- nan_index
lung_comp_data$label_matrix <- label_matrix
lung_comp_data$nan_label_matrix <- nan_label_matrix
lung_comp_data$zero_label_matrix <- zero_label_matrix

save(lung_comp_data, file = "compareSamples/liger/lung_comp_mouse1_mouse2_liger.rda")
