source("~/Desktop/MacroRandomForest/R/MRF_non_random.R")
library(pracma)
options("max.print" = 1000000)

data.input = read.csv("/Users/ryanlucas/Desktop/MacroRandomForest/Datasets/UNRATE.csv")
print(head(data.input))
data.input = data.input[, 2:101]

print(head(data.input))
mrf.output=MRF(data=data.input,y.pos=1,x.pos=2:4,S.pos=2:ncol(data.input),oos.pos=164: 213, VI = FALSE, fast.rw = TRUE, resampling.opt = 2, mtry.frac = 0.25, trend.push=4,quantile.rate=0.3, B = 200)
end_time <- Sys.time()
print(end_time - start_time)
write.csv(mrf.output$betas, "/Users/ryanlucas/Desktop/MRF/mrf_betas_inf_h1.csv")
write.csv(mrf.output$pred.ensemble, "/Users/ryanlucas/Desktop/MRF/predictions_inf_h1.csv")

