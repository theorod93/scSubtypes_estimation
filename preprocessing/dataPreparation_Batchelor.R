#' Filename: dataPreparation_Batchelor.R
#' Author: Theodoulos Rodosthenous
#' Date: 26-02-2025
#' Description: Pre-processing of scRNA-seq data via Batchelor
#' This script was used in the batch correction section of the manuscript
#' 
#' In most of the analysis, the pre-processing and integration was performed 
#' by Seurat. 
#' Here, Batchelor is used instead
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


## Preparation ##
# Remove columns with same name to enforce unique cell names
common_cells <- colnames(mouse1_inp)[which(colnames(mouse1_inp) %in% colnames(mouse2_inp))]
mouse1_inp = mouse1_inp %>% select(-which(colnames(mouse1_inp) %in% common_cells))
mouse2_inp = mouse2_inp %>% select(-which(colnames(mouse2_inp) %in% common_cells))
# And annotations
if (any(mouse1_inp_anno$Cell.Barcode %in% common_cells)){
  mouse1_inp_anno <- mouse1_inp_anno[-which(mouse1_inp_anno$Cell.Barcode %in% common_cells),]
}
if (any(mouse2_inp_anno$Cell.Barcode %in% common_cells)){
  mouse2_inp_anno <- mouse2_inp_anno[-which(mouse2_inp_anno$Cell.Barcode %in% common_cells),]
}

## Integration ##
## Follow tutorial ##

m1 <- Matrix(as.matrix(mouse1_inp), sparse = TRUE)
m2 <- Matrix(as.matrix(mouse2_inp), sparse = TRUE)
# Create SingleCellExperiment objects
sce.m1 <- SingleCellExperiment(list(counts = m1))
sce.m2 <- SingleCellExperiment(list(counts = m2))


# Quality control
## mouse1
#sce.m1 <- addPerCellQC(sce.m1, subsets=list(Mito=grep("mt-", rownames(sce.m1))))
#qc1 <- quickPerCellQC(colData(sce.m1), sub.fields="subsets_Mito_percent")
#sce.m1 <- sce.m1[,!qc1$discard]
## mouse2
#sce.m2 <- addPerCellQC(sce.m2, subsets=list(Mito=grep("mt-", rownames(sce.m2))))
#qc1 <- quickPerCellQC(colData(sce.m2), sub.fields="subsets_Mito_percent")
#sce.m2 <- sce.m2[,!qc1$discard]

# Pre-processing to render sce.m1 and sce.m1 comparable
universe <- intersect(rownames(sce.m1), rownames(sce.m2))
sce.m1 <- sce.m1[universe,]
sce.m2 <- sce.m2[universe,]

# Log-normalised expression
out <- multiBatchNorm(sce.m1, sce.m2)
sce.m1 <- out[[1]]
sce.m2 <- out[[2]]

# Top 5000 genes with the largest biological components of their variance
dec1 <- modelGeneVar(sce.m1)
dec2 <- modelGeneVar(sce.m2)
combined.dec <- combineVar(dec1, dec2)
chosen.hvgs <- getTopHVGs(combined.dec, n=5000)

### COMBINE WITH NO BATCH CORRECTION ###
combined <- correctExperiments(A=sce.m1, B=sce.m2, PARAM=NoCorrectParam())
# t-SNE
combined <- runPCA(combined, subset_row=chosen.hvgs)
combined <- runTSNE(combined, dimred="PCA")
plotTSNE(combined, colour_by="batch")

### The new, fast method -- fastMNN() ###
f.out <- fastMNN(A=sce.m1, B=sce.m2, subset.row=chosen.hvgs)
str(reducedDim(f.out, "corrected"))
rle(f.out$batch)
f.out <- runTSNE(f.out, dimred="corrected")
plotTSNE(f.out, colour_by="batch")

### The old, classic method -- mnnCorrect() ###
# Using fewer genes as it is much, much slower. 
fewer.hvgs <- head(order(combined.dec$bio, decreasing=TRUE), 100)
classic.out <- mnnCorrect(sce.m1, sce.m2, subset.row=fewer.hvgs)

### The cluster-based method ###
# Removing the 'unclassified' cluster, which makes no sense:
not.unclass <- sce.m1$broad_type!="Unclassified"
clust.out <- clusterMNN(sce.m1, sce.m2,
                        subset.row=chosen.hvgs,
                        clusters=list(as.factor(mouse1_inp_anno$Annotation[which(mouse1_inp_anno$Cell.name %in% colnames(sce.m1))]), 
                                      as.factor(mouse2_inp_anno$Annotation[which(mouse2_inp_anno$Cell.name %in% colnames(sce.m2))]))) 
clust.info <- metadata(clust.out)$cluster
split(clust.info$cluster, clust.info$meta)


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

## Store Data ## 
lung_comp_data <- vector("list")
lung_comp_data$fastMNN <- as.matrix(assay(f.out))
lung_comp_data$classicMNN <- as.matrix(assay(classic.out))
lung_comp_data$clustMNN <- as.matrix(assay(clust.out))
lung_comp_data$noBatch <- as.matrix(assay(combined[which(rownames(combined) %in% chosen.hvgs),]))
lung_comp_data$labels_names <- labels_names
lung_comp_data$labels_num <- labels_num
lung_comp_data$nan_index <- nan_index
lung_comp_data$label_matrix <- label_matrix
lung_comp_data$nan_label_matrix <- nan_label_matrix
lung_comp_data$zero_label_matrix <- zero_label_matrix

save(lung_comp_data, file = "compareSamples/batchelor/lung_comp_mouse1_mouse2_batchelor.rda")
