#'  Get the two dataset from Seurat. More datasets are available
#' 
#' 


# Libraries
library(Seurat)
library(SeuratData)

load("bmcite.SeuratData/data/bmcite.rda")
load("pbmc3k.SeuratData/data/pbmc3k.rda")

## Understanding anchors and integration technique by Stuart in 'Comprehensive integration of single-cell data'
# Reference: https://satijalab.org/seurat/archive/v3.2/integration.html
data(panc8)
pancreas.list <- SplitObject(panc8, split.by = "tech")
pancreas.list <- pancreas.list[c("celseq", "celseq2", "fluidigmc1", "smartseq2")]
for (i in 1:length(pancreas.list)) {
  pancreas.list[[i]] <- NormalizeData(pancreas.list[[i]], verbose = FALSE)
  pancreas.list[[i]] <- FindVariableFeatures(pancreas.list[[i]], selection.method = "vst", 
                                             nfeatures = 2000, verbose = FALSE)
}
reference.list <- pancreas.list[c("celseq", "celseq2", "smartseq2")]
pancreas.anchors <- FindIntegrationAnchors(object.list = reference.list, dims = 1:30)


## pbmc Multiome
load("pbmcMultiome.SeuratData/data/pbmc.atac.rda")
load("pbmcMultiome.SeuratData/data/pbmc.rna.rda")
pbmc.rna <- CreateSeuratObject(counts = pbmc.rna.counts)
pbmc.rna <- NormalizeData(object = pbmc.rna)
pbmc.rna <- FindVariableFeatures(object = pbmc.rna)
pbmc.rna <- ScaleData(object = pbmc.rna)
pbmc.rna <- RunPCA(object = pbmc.rna)
pbmc.rna <- FindNeighbors(object = pbmc.rna)
data_pbmc.rna <- as.matrix(GetAssayData(pbmc.rna), slot ="data")

pbmc.atac <- NormalizeData(object = pbmc.atac)
pbmc.atac <- FindVariableFeatures(object = pbmc.atac)
pbmc.atac <- ScaleData(object = pbmc.atac)
pbmc.atac <- RunPCA(object = pbmc.atac)
pbmc.atac <- FindNeighbors(object = pbmc.atac)
data_pbmc.atac <- as.matrix(GetAssayData(pbmc.atac), slot ="data")
write.table(data_pbmc.atac, "data_pbmc.rna.txt", row.names = F, col.names = F, quote = F)
write.table(data_pbmc.rna, "data_pbmc.rna.txt", row.names = F, col.names = F, quote = F)