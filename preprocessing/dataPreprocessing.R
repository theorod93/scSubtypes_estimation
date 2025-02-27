#' Filename: dataPreprocessing.R
#' Author: Theodoulos Rodosthenous
#' Date: 26-02-2025
#' Description: Pre-processing of scRNA-seq data
#' This script reads and prepares the data to be used as input for S-multi-SNE
#' Steps:
#' 1. Create Seurat objects for both training (reference) and testing
#' 2. Normalise
#' 3. Find variable features
#' 4. Integrate by using immune anchors
#' 5. Incorporate labels (reference cell-types)
#' 
#' This process can be repeated between 
#' - Different samples (e.g. mouse1, mouse2)
#' - Different species (e.g. mouse1, adult1)
#' - Different sequencing technologies (e.g. SmartSeq, CellSeq)
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
# Seurat Object
integrate_list <- vector("list")
# Adult
temp <- as.matrix(mouse1_inp)
integrate_list[[1]] <- CreateSeuratObject(counts = temp)
# Fetal
temp <- as.matrix(mouse2_inp)
integrate_list[[2]] <- CreateSeuratObject(counts = temp)
names(integrate_list) <- c("mouse1", "mouse2")

# Apply pre-processing
integrate_list <- lapply(X = integrate_list, FUN = function(x) {
  x <- NormalizeData(x)
  x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
})
# select features that are repeatedly variable across datasets for integration
features <- SelectIntegrationFeatures(object.list = integrate_list)
#### PERFORM INTEGRATION
immune.anchors <- FindIntegrationAnchors(object.list = integrate_list, anchor.features = features)
# this command creates an 'integrated' data assay
lung_comp_samples <- IntegrateData(anchorset = immune.anchors)

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
lung_comp_data$data <- as.matrix(GetAssayData(lung_comp_samples), slot ="data")
lung_comp_data$labels_names <- labels_names
lung_comp_data$labels_num <- labels_num
lung_comp_data$nan_index <- nan_index
lung_comp_data$label_matrix <- label_matrix
lung_comp_data$nan_label_matrix <- nan_label_matrix
lung_comp_data$zero_label_matrix <- zero_label_matrix

#save(lung_comp_data, file = "lung_comp_mouse1_mouse2.rda")



#### Alternate annotations to reflect corresponding ones between species:
#### Connect human and mouse ####
# Adjust the label names to be aligned (different names given but same label)
change_labels_species_mca <- function(labels_names_orig){
  labels_names <- labels_names_orig
  # Manually change of labels_names for consistency
  for (i in 1:length(labels_names_orig)){
    if (labels_names_orig[i] == "AT2 Cell(Lung)"){
      labels_names[i] <- "AT2"
    } else if(labels_names_orig[i] == "Dendritic cell_Naaa high(Lung)"){
      labels_names[i] <- "Dendritic"
    } else if(labels_names_orig[i] == "Igâˆ’producing B cell(Lung)"){
      labels_names[i] <- "B"
    } else if(labels_names_orig[i] == "Conventional dendritic cell_H2-M2 high(Lung)"){
      labels_names[i] <- "Conventional dendritic"
    } else if(labels_names_orig[i] == "Endothelial cells_Vwf high(Lung)"){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] == "Stromal cell_Acta2 high(Lung)" ){
      labels_names[i] <- "Stromal"
    } else if(labels_names_orig[i] == "Ciliated cell(Lung)"){
      labels_names[i] <- "Ciliated"
    } else if(labels_names_orig[i] == "T Cell_Cd8b1 high(Lung)"){
      labels_names[i] <- "T"
    } else if(labels_names_orig[i] == "B Cell(Lung)"){
      labels_names[i] <- "B"
    } else if(labels_names_orig[i] == "Nuocyte(Lung)"){
      labels_names[i] <- "Nuocyte"
    } else if(labels_names_orig[i] == "Dividing dendritic cells(Lung)"  ){
      labels_names[i] <- "Dividing Dendritic"
    } else if(labels_names_orig[i] == "Alveolar macrophage_Ear2 high(Lung)"){
      labels_names[i] <- "Alveolar macrophage"
    } else if(labels_names_orig[i] == "Stromal cell_Dcn high(Lung)"){
      labels_names[i] <- "Stromal"
    } else if(labels_names_orig[i] =="Interstitial macrophage(Lung)"){
      labels_names[i] <- "Interstitial macrophage"
    } else if(labels_names_orig[i] == "Plasmacytoid dendritic cell(Lung)"){
      labels_names[i] <- "Plasmacytoid Dendritic"
    } else if(labels_names_orig[i] == "Monocyte progenitor cell(Lung)" ){
      labels_names[i] <- "Monocyte"
    } else if(labels_names_orig[i] == "Endothelial cell_Tmem100 high(Lung)"){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] == "AT1 Cell(Lung)"){
      labels_names[i] <- "AT1"
    } else if(labels_names_orig[i] == "Neutrophil granulocyte(Lung)"){
      labels_names[i] <- "Neutrophil granulocyte"
    } else if(labels_names_orig[i] == "NK Cell(Lung)"){
      labels_names[i] <- "NK"
    } else if(labels_names_orig[i] ==  "Eosinophil granulocyte(Lung)"){
      labels_names[i] <- "Eosinophil granulocyte"
    } else if(labels_names_orig[i] == "Stromal cell_Inmt high(Lung)"){
      labels_names[i] <- "Stromal"
    } else if(labels_names_orig[i] == "Conventional dendritic cell_Gngt2 high(Lung)"){
      labels_names[i] <- "Conventional dendritic"
    } else if(labels_names_orig[i] == "Clara Cell(Lung)" ){
      labels_names[i] <- "Clara"
    } else if(labels_names_orig[i] == "Conventional dendritic cell_Tubb5 high(Lung)"){
      labels_names[i] <- "Conventional dendritic"
    } else if(labels_names_orig[i] == "Basophil(Lung)"){
      labels_names[i] <- "Basophil"
    } else if(labels_names_orig[i] == "Endothelial cell_Kdr high(Lung)"){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] == "Conventional dendritic cell_Mgl2 high(Lung)"){
      labels_names[i] <- "Conventional dendritic"
    } else if(labels_names_orig[i] ==  "Dividing T cells(Lung)" ){
      labels_names[i] <- "Dividing T"
    } else if(labels_names_orig[i] == "Dividing cells(Lung)"){
      labels_names[i] <- "Dividing"
    } else if(labels_names_orig[i] == "Alveolar macrophage_Pclaf high(Lung)"){
      labels_names[i] <- "Alveolar macrophage"
    } else if(labels_names_orig[i] == "Fibroblast_A2M high"){
      labels_names[i] <- "Fibroblast"
    } else if(labels_names_orig[i] == "AT2 cell"){
      labels_names[i] <- "AT2"
    } else if(labels_names_orig[i] == "Endothelial cell_VWF high" ){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] == "Endothelial cell_TMEM100 high"){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] == "AT1 cell "){
      labels_names[i] <- "AT1"
    } else if(labels_names_orig[i] == "B cell (Plasmocyte)_IGHA/HM high"){
      labels_names[i] <- "B"
    } else if(labels_names_orig[i] == "Neutrophil"){
      labels_names[i] <- "Neutrophil"
    } else if(labels_names_orig[i] ==  "Endothelial cell_SELE high" ){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] == "Macrophage_M2"){
      labels_names[i] <- "Macrophage"
    } else if(labels_names_orig[i] == "Macrophage"){
      labels_names[i] <- "Macrophage "
    } else if(labels_names_orig[i] == "Conventional dendritic cell"){
      labels_names[i] <- "Conventional dendritic"
    } else if(labels_names_orig[i] == "Smooth muscle cell"){
      labels_names[i] <- "Smooth muscle"
    } else if(labels_names_orig[i] == "Club cell_BPIFB1 high" ){
      labels_names[i] <- "Club"
    } else if(labels_names_orig[i] == "Club cell"){
      labels_names[i] <- "Club"
    } else if(labels_names_orig[i] == "Megakaryocyte"){
      labels_names[i] <- "Megakaryocyte"
    } else if(labels_names_orig[i] == "Basal/Epithelial cell"){
      labels_names[i] <- "Basal/Epithelial"
    } else if(labels_names_orig[i] == "Fibroblast"){
      labels_names[i] <- "Fibroblast"
    } else if(labels_names_orig[i] == "T cell"){
      labels_names[i] <- "T"
    } else if(labels_names_orig[i] ==  "Ciliated cell" ){
      labels_names[i] <- "Ciliated"
    } else if(labels_names_orig[i] == "Artry endothelial cell" ){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] == "Alveolar bipotent/intermediate cell"){
      labels_names[i] <- "Alveolar intermediate"
    } else if(labels_names_orig[i] == "Fibroblast_SFRP high"){
      labels_names[i] <- "Fibroblast"
    } else if(labels_names_orig[i] == "Chondrocyte"){
      labels_names[i] <- "Chondrocyte"
    } else if(labels_names_orig[i] ==  "Mast  cell" ){
      labels_names[i] <- "Mast "
    } else if(labels_names_orig[i] == "B cell (Plasmocyte)_IGHG high" ){
      labels_names[i] <- "B"
    } else if(labels_names_orig[i] == "Dendritic cell" ){
      labels_names[i] <- "Dendritic"
    } else if(labels_names_orig[i] == "AT1 cell" ){
      labels_names[i] <- "AT1"
    } else if(labels_names_orig[i] == "Lymphatic endothelial cell" ){
      labels_names[i] <- "Lymphatic endothelial"
    } else if(labels_names_orig[i] == "Club cell_KLK11 high" ){
      labels_names[i] <- "Club"
    } else if(labels_names_orig[i] ==  "B cell (Plasmocyte)" ){
      labels_names[i] <- "B"
    } else if(labels_names_orig[i] ==  "Epithelial cell_S100A2 high" ){
      labels_names[i] <- "Epithelial"
    } else if(labels_names_orig[i] ==  "Epithelial cell_PLA2G2A high" ){
      labels_names[i] <- "Epithelial"
    } else if(labels_names_orig[i] ==  "Proliferating alveolar bipotent progenitor cell" ){
      labels_names[i] <- "Alveolar intermediate"
    } else if(labels_names_orig[i] ==  "Proliferating cell" ){
      labels_names[i] <- "Proliferating"
    } else if(labels_names_orig[i] ==  "Arterial endothelial cell" ){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] ==  "Mast cell" ){
      labels_names[i] <- "Mast"
    }
  }
  return(labels_names)
}


## A more compact list than the previous one
change_labels_species_mca_compact <- function(labels_names_orig){
  labels_names <- labels_names_orig
  # Manually change of labels_names for consistency
  for (i in 1:length(labels_names_orig)){
    if (labels_names_orig[i] == "AT2 Cell(Lung)"){
      labels_names[i] <- "AT2"
    } else if(labels_names_orig[i] == "Dendritic cell_Naaa high(Lung)"){
      labels_names[i] <- "Dendritic"
    } else if(labels_names_orig[i] == "Ig?^'producing B cell(Lung)"){
      labels_names[i] <- "B"
    } else if(labels_names_orig[i] == "Conventional dendritic cell_H2-M2 high(Lung)"){
      labels_names[i] <- "Dendritic"
    } else if(labels_names_orig[i] == "Endothelial cells_Vwf high(Lung)"){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] == "Stromal cell_Acta2 high(Lung)" ){
      labels_names[i] <- "Stromal"
    } else if(labels_names_orig[i] == "Ciliated cell(Lung)"){
      labels_names[i] <- "Ciliated"
    } else if(labels_names_orig[i] == "T Cell_Cd8b1 high(Lung)"){
      labels_names[i] <- "T"
    } else if(labels_names_orig[i] == "B Cell(Lung)"){
      labels_names[i] <- "B"
    } else if(labels_names_orig[i] == "Nuocyte(Lung)"){
      labels_names[i] <- "Nuocyte"
    } else if(labels_names_orig[i] == "Dividing dendritic cells(Lung)"  ){
      labels_names[i] <- "Dendritic"
    } else if(labels_names_orig[i] == "Alveolar macrophage_Ear2 high(Lung)"){
      labels_names[i] <- "Macrophage"
    } else if(labels_names_orig[i] == "Stromal cell_Dcn high(Lung)"){
      labels_names[i] <- "Stromal"
    } else if(labels_names_orig[i] =="Interstitial macrophage(Lung)"){
      labels_names[i] <- "Macrophage"
    } else if(labels_names_orig[i] == "Plasmacytoid dendritic cell(Lung)"){
      labels_names[i] <- "Dendritic"
    } else if(labels_names_orig[i] == "Monocyte progenitor cell(Lung)" ){
      labels_names[i] <- "Monocyte"
    } else if(labels_names_orig[i] == "Endothelial cell_Tmem100 high(Lung)"){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] == "AT1 Cell(Lung)"){
      labels_names[i] <- "AT1"
    } else if(labels_names_orig[i] == "Neutrophil granulocyte(Lung)"){
      labels_names[i] <- "Granulocyte"
    } else if(labels_names_orig[i] == "NK Cell(Lung)"){
      labels_names[i] <- "NK"
    } else if(labels_names_orig[i] ==  "Eosinophil granulocyte(Lung)"){
      labels_names[i] <- "Granulocyte"
    } else if(labels_names_orig[i] == "Stromal cell_Inmt high(Lung)"){
      labels_names[i] <- "Stromal"
    } else if(labels_names_orig[i] == "Conventional dendritic cell_Gngt2 high(Lung)"){
      labels_names[i] <- "Dendritic"
    } else if(labels_names_orig[i] == "Clara Cell(Lung)" ){
      labels_names[i] <- "Club"
    } else if(labels_names_orig[i] == "Conventional dendritic cell_Tubb5 high(Lung)"){
      labels_names[i] <- "Dendritic"
    } else if(labels_names_orig[i] == "Basophil(Lung)"){
      labels_names[i] <- "Granulocyte"
    } else if(labels_names_orig[i] == "Endothelial cell_Kdr high(Lung)"){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] == "Conventional dendritic cell_Mgl2 high(Lung)"){
      labels_names[i] <- "Dendritic"
    } else if(labels_names_orig[i] ==  "Dividing T cells(Lung)" ){
      labels_names[i] <- "Dividing"
    } else if(labels_names_orig[i] == "Dividing cells(Lung)"){
      labels_names[i] <- "Dividing"
    } else if(labels_names_orig[i] == "Alveolar macrophage_Pclaf high(Lung)"){
      labels_names[i] <- "Macrophage"
    } else if(labels_names_orig[i] == "Fibroblast_A2M high"){
      labels_names[i] <- "Fibroblast"
    } else if(labels_names_orig[i] == "AT2 cell"){
      labels_names[i] <- "AT2"
    } else if(labels_names_orig[i] == "Endothelial cell_VWF high" ){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] == "Endothelial cell_TMEM100 high"){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] == "AT1 cell "){
      labels_names[i] <- "AT1"
    } else if(labels_names_orig[i] == "B cell (Plasmocyte)_IGHA/HM high"){
      labels_names[i] <- "B"
    } else if(labels_names_orig[i] == "Neutrophil"){
      labels_names[i] <- "Neutrophil"
    } else if(labels_names_orig[i] ==  "Endothelial cell_SELE high" ){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] == "Macrophage_M2"){
      labels_names[i] <- "Macrophage"
    } else if(labels_names_orig[i] == "Conventional dendritic cell"){
      labels_names[i] <- "Dendritic"
    } else if(labels_names_orig[i] == "Smooth muscle cell"){
      labels_names[i] <- "Smooth muscle"
    } else if(labels_names_orig[i] == "Club cell_BPIFB1 high" ){
      labels_names[i] <- "Club"
    } else if(labels_names_orig[i] == "Club cell"){
      labels_names[i] <- "Club"
    } else if(labels_names_orig[i] == "Megakaryocyte"){
      labels_names[i] <- "Macrophage"
    } else if(labels_names_orig[i] == "Basal/Epithelial cell"){
      labels_names[i] <- "Epithelial"
    } else if(labels_names_orig[i] == "Fibroblast"){
      labels_names[i] <- "Fibroblast"
    } else if(labels_names_orig[i] == "T cell"){
      labels_names[i] <- "T"
    } else if(labels_names_orig[i] ==  "Ciliated cell" ){
      labels_names[i] <- "Ciliated"
    } else if(labels_names_orig[i] == "Artry endothelial cell" ){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] == "Alveolar bipotent/intermediate cell"){
      labels_names[i] <- "Alveolar intermediate"
    } else if(labels_names_orig[i] == "Fibroblast_SFRP high"){
      labels_names[i] <- "Fibroblast"
    } else if(labels_names_orig[i] == "Chondrocyte"){
      labels_names[i] <- "Chondrocyte"
    } else if(labels_names_orig[i] ==  "Mast  cell" ){
      labels_names[i] <- "Granulocyte "
    } else if(labels_names_orig[i] == "B cell (Plasmocyte)_IGHG high" ){
      labels_names[i] <- "B"
    } else if(labels_names_orig[i] == "Dendritic cell" ){
      labels_names[i] <- "Dendritic"
    } else if(labels_names_orig[i] == "AT1 cell" ){
      labels_names[i] <- "AT1"
    } else if(labels_names_orig[i] == "Lymphatic endothelial cell" ){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] == "Club cell_KLK11 high" ){
      labels_names[i] <- "Club"
    } else if(labels_names_orig[i] ==  "B cell (Plasmocyte)" ){
      labels_names[i] <- "B"
    } else if(labels_names_orig[i] ==  "Epithelial cell_S100A2 high" ){
      labels_names[i] <- "Epithelial"
    } else if(labels_names_orig[i] ==  "Epithelial cell_PLA2G2A high" ){
      labels_names[i] <- "Epithelial"
    } else if(labels_names_orig[i] ==  "Proliferating alveolar bipotent progenitor cell" ){
      labels_names[i] <- "Alveolar intermediate"
    } else if(labels_names_orig[i] ==  "Proliferating cell" ){
      labels_names[i] <- "Proliferating"
    } else if(labels_names_orig[i] ==  "Arterial endothelial cell" ){
      labels_names[i] <- "Endothelial"
    } else if(labels_names_orig[i] ==  "Mast cell" ){
      labels_names[i] <- "Granulocyte"
    }
  }
  return(labels_names)
}

