if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(feather)

JobId=as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
print(JobId)

data <- as.data.frame(as.matrix(read.csv(paste0("/well/nichols/users/qcv214/KGPNN/pile/re_jan31_bimodal_gpnn_bbig_init_minweightsdmn__jobid_",JobId,".csv"))))
write_feather(data,paste0( '/well/nichols/users/qcv214/KGPNN/pile/re_jan31_bimodal_gpnn_bbig_init_minweightsdmn__jobid_',JobId,'.feather'))

