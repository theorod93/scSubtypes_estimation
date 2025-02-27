#' Filename: run_SmultiSNE_example.R
#' Author: Theodoulos Rodosthenous
#' Date: 26-02-2025
#' 
#' Description: Run S-multi-SNE
#' This script implements S-multi-SNE
#' This script describes the process of running S-multi-SNE on a the PBMC dataset
#' collected from Seurat Data. 
#' In this particular scenario, both pre-training with PCA and sCCA are explored
#' In this scenario 80% of the data are missing.
#' 
#' However the same methodology can be used for all S-multi-SNE implementations
#' described in the manuscript.
#' 
#'


library(Seurat)
library(SeuratData)
library(patchwork)
library(multiSNE)
library(Signac)
library(PMA)

# Data
load("pbmcMultiOme_data_full.rda")
load("labels_data.rda")

#### PCA ####
sne_data_pca <- vector("list")
sne_data_pca$rna <- pbmcMultiOme_data$rna@reductions$pca@cell.embeddings
sne_data_pca$atac <- pbmcMultiOme_data$atac@reductions$lsi@cell.embeddings
sne_data_pca$labels <- labels_data$nan_label_matrix_80

# Run S-multi-SNE
perp = 2
Y_SmultiSNE_pca <- SmultiSNE(sne_data_pca, 
                             whitening=F, 
                             perplexity = perp,
                             max_iter = 1000)
# Save
write.table(Y_SmultiSNE_pca$Y, "results/Y_SmultiSNE_pbmcMultiome_tr20_pca_full_p2.txt", row.names=F, col.names=F, quote=F)
write.table(Y_SmultiSNE_pca$Weights, "results/Weights_multiSNE_pbmcMultiome_tr20_pca_full_p2.txt", row.names=F, col.names=F, quote=F)
write.table(Y_SmultiSNE_pca$Errors, "results/Errors_multiSNE_pbmcMultiome_tr20_pca_full_p2.txt", row.names=F, col.names=F, quote=F)

#### CCA ####
rna_data <- GetAssayData(pbmcMultiOme_data$rna, "data")
atac_data <- GetAssayData(pbmcMultiOme_data$atac, "data")

# Transpose
cca_rna <- t(as.matrix(rna_data))
cca_atac <- t(as.matrix(atac_data))
# CCA
perm_cca <- PMA::CCA.permute(x = cca_rna, z = cca_atac, typex =  "standard", typez = "standard", nperms = 8)
out_cca <- PMA::CCA(x = cca_rna, z = cca_atac, typex =  "standard", typez = "standard", penaltyx=perm_cca$bestpenaltyx, v=perm_cca$v.init, penaltyz=perm_cca$bestpenaltyz)

# Get input from CCA - canononical vectors
cca_input_rna <- cca_rna %*% out_cca$u
cca_input_atac <- cca_atac %*% out_cca$v

## Prepare input matrices
sne_data_cca <- vector("list")
sne_data_cca$rna <- cca_input_rna
sne_data_cca$atac <- cca_input_rna
sne_data_cca$labels <- labels_data$nan_label_matrix_20

# Run S-multi-SNE

perp = 2
Y_SmultiSNE_cca <- SmultiSNE(sne_data_cca,
                             whitening=F,
                             perplexity = perp,
                             max_iter = 1000)
# Save
write.table(Y_SmultiSNE_cca$Y, "results/Y_SmultiSNE_pbmcMultiome_tr20_cca_full_p2.txt", row.names=F, col.names=F, quote=F)
write.table(Y_SmultiSNE_cca$Weights, "results/Weights_multiSNE_pbmcMultiome_tr20_cca_full_p2.txt", row.names=F, col.names=F, quote=F)
write.table(Y_SmultiSNE_cca$Errors, "results/Errors_multiSNE_pbmcMultiome_tr20_cca_full_p2.txt", row.names=F, col.names=F, quote=F)
