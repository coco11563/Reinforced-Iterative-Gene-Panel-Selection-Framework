

FindRegionalGenes_<-function (obj, dims = 1:10, nfeatures = 2000, overlap_stop = 0.75, 
          max_iteration = 10, snn = NULL, do_test, p_threshold, verbose = TRUE, 
          neigh_num = 20, is.save = FALSE, dir = "") 
{
  if (is.save) {
    gene_all = list()
  }
  all.genes = rownames(obj)
  block.size = 1000
  max.block <- ceiling(x = length(x = all.genes)/block.size)
  cell_num = dim(obj)[2]
  obj = ScaleData(obj, features = all.genes, verbose = FALSE)
  obj_data <- GetAssayData(object = obj, slot = "scale.data")
  if (is.null(snn)) {
    obj <- FindNeighbors(obj, dims = dims, k.param = neigh_num, 
                         verbose = FALSE)
    snn <- obj$RNA_snn
    diag(snn) <- 0
    if (verbose) {
      message("calculating gene score")
      pb <- txtProgressBar(min = 0, max = max.block, style = 3, 
                           file = stderr())
    }
    HRG_score = c()
    for (i in 1:max.block) {
      my.inds <- ((block.size * (i - 1)):(block.size * 
                                            i - 1)) + 1
      my.inds <- my.inds[my.inds <= length(x = all.genes)]
      for (index in my.inds) {
        data_temp = as.matrix(obj_data[index, ])
        HRG_score[index] = as.numeric(t(data_temp) %*% 
                                        snn %*% (data_temp))
      }
      if (verbose) {
        setTxtProgressBar(pb = pb, value = i)
      }
    }
    if (verbose) {
      close(con = pb)
    }
    names(HRG_score) <- all.genes
    feature_gene = names(sort(HRG_score, decreasing = TRUE))[1:nfeatures]
    if (is.save) {
      gene_all[[1]] = feature_gene
    }
    overlap = 0
    count <- 0
    while ((overlap < overlap_stop) & (count < max_iteration)) {
      count = count + 1
      obj = RunPCA(obj, features = feature_gene, verbose = FALSE)
      obj <- FindNeighbors(obj, dims = dims, k.param = neigh_num, 
                           verbose = FALSE)
      snn <- obj$RNA_snn
      diag(snn) <- 0
      if (verbose) {
        message("calculating gene score")
        pb <- txtProgressBar(min = 0, max = max.block, 
                             style = 3, file = stderr())
      }
      for (i in 1:max.block) {
        my.inds <- ((block.size * (i - 1)):(block.size * 
                                              i - 1)) + 1
        my.inds <- my.inds[my.inds <= length(x = all.genes)]
        for (index in my.inds) {
          data_temp = as.matrix(obj_data[index, ])
          HRG_score[index] = as.numeric(t(data_temp) %*% 
                                          snn %*% (data_temp))
        }
        if (verbose) {
          setTxtProgressBar(pb = pb, value = i)
        }
      }
      if (verbose) {
        close(con = pb)
      }
      feature_gene_new = names(sort(HRG_score, decreasing = TRUE))[1:nfeatures]
      overlap = length(intersect(feature_gene_new, feature_gene))/nfeatures
      if (verbose) {
        message(paste0("overlap is ", overlap))
      }
      feature_gene = feature_gene_new
      if (is.save) {
        gene_all[[count]] = feature_gene
      }
    }
  }
  else {
    size = dim(snn)
    if (size[1] != cell_num | size[2] != cell_num) {
      stop("the snn should be cellnumberxcellnumber matrix")
    }
    if (verbose) {
      message("calculating gene score")
      pb <- txtProgressBar(min = 0, max = max.block, style = 3, 
                           file = stderr())
    }
    HRG_score = c()
    for (i in 1:max.block) {
      my.inds <- ((block.size * (i - 1)):(block.size * 
                                            i - 1)) + 1
      my.inds <- my.inds[my.inds <= length(x = all.genes)]
      for (index in my.inds) {
        data_temp = as.matrix(obj_data[index, ])
        HRG_score[index] = as.numeric(t(data_temp) %*% 
                                        snn %*% (data_temp))
      }
      if (verbose) {
        setTxtProgressBar(pb = pb, value = i)
      }
    }
    if (verbose) {
      close(con = pb)
    }
    names(HRG_score) <- all.genes
  }
    HRG_rank = rank(-HRG_score)
    HRG_score = as.matrix(HRG_score)[, 1]
    HRG_score = sort(HRG_score, decreasing = TRUE)
    elbow_point = KneeArrower::findCutoff(1:length(HRG_score), HRG_score, 'curvature')
    gene_num = floor(elbow_point$x)
    HRG_rank = as.matrix(HRG_rank)[, 1]

  return (names(sort(HRG_rank))[1:gene_num])
}


tasks<-c('Leng', 'Koh', 'Puram', 'Han', 'Kumar', 'Yang', 'Robert', 
'Maria1', 'MacParland', 'Chu2', 'Maria2', 'Goolam', 'Engel', 
'Darmanis','Ting','Chung', 'Li', 'Cao', 'Chu1',
'CITE CBMC','Human Pancreas1','Human Pancreas2','Human Pancreas3','Mouse Pancreas1','Mouse Pancreas2')



library(Seurat)
library(rhdf5)
base_path<-"data/"
save_dir<-'HRG_result/'

suffix<-"_total.hdf"
for(task in tasks){
  path <- paste(base_path,task,'/',task,suffix,sep="")

  hd5 <- h5read(path,'/') 

  start_time <- Sys.time()


  expression_matrix <- as.matrix(hd5$df$block0_values)
  expression_matrix<-expression_matrix[1:nrow(expression_matrix)-1,1:ncol(expression_matrix)]
  sobj <- CreateSeuratObject(counts = expression_matrix)


  all.genes=rownames(sobj)
  sobj <- NormalizeData(sobj, normalization.method = "LogNormalize", scale.factor = 10000)
  sobj=ScaleData(sobj,features=all.genes,verbose = FALSE)
  sobj=RunPCA(sobj,features=all.genes,verbose = FALSE)


  feature_selected=FindRegionalGenes_(sobj,dims = 1:10,nfeatures = 2000,overlap_stop = 0.95)

  
  end_time <- Sys.time()
  running_time <- (end_time - start_time)


  save_path<- paste(save_dir,task,'.txt',sep="")
  con <- file(save_path, open = "wt")

  cat(feature_selected,"\n",running_time,file=con)

  
  close(con)
  print(task)
}