library(scater)
library(geneBasisR)
library(rhdf5)

tasks<-c('Leng', 'Koh', 'Puram', 'Han', 'Kumar', 'Yang', 'Robert', 
'Maria1', 'MacParland', 'Chu2', 'Maria2', 'Goolam', 'Engel', 
'Darmanis','Ting','Chung', 'Li', 'Cao', 'Chu1',
'CITE CBMC','Human Pancreas1','Human Pancreas2','Human Pancreas3','Mouse Pancreas1','Mouse Pancreas2')



base_path<-"data/"
save_dir<-"/GeneBasis_result/"
suffix<-"_total.hdf"

for(task in tasks){
    path <- paste(base_path,task,'/',task,suffix,sep="")

    hd5 <- h5read(path,'/') 
  
    start_time <- Sys.time()


    expression_matrix <- as.matrix(hd5$df$block0_values)
    colnames(expression_matrix)<-1:ncol(expression_matrix)
    rownames(expression_matrix)<-1:nrow(expression_matrix)
    expression_matrix<-expression_matrix[1:nrow(expression_matrix)-1,1:ncol(expression_matrix)]

    sce <- SingleCellExperiment(
    assays = list(counts = expression_matrix),
    )
    sce<-logNormCounts(sce,transform='log')
    sce <- retain_informative_genes(sce)

    feature_selected <- gene_search(sce, n_genes_total = 50)

    
    end_time <- Sys.time()
    running_time <- (end_time - start_time)


    save_path<- paste(save_dir,task,'.txt',sep="")
    con <- file(save_path, open = "wt")
    
    cat(paste(feature_selected, collapse = ","),"\n",running_time,file=con)


    close(con)
    print(task)
    print(running_time)

}