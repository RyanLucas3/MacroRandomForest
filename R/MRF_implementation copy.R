source("~/Desktop/MRF/MRF_non_random.R")
library(pracma)
options("max.print" = 1000000)

getwd()
data.in = read.csv("/Users/ryanlucas/Desktop/MRF/mrf_data.csv")

mrf.output=MRF(data=data.in,y.pos=1,x.pos=2:4,S.pos=2:ncol(data.in),oos.pos=150:200,mtry.frac = 0.25, trend.push=4,quantile.rate=0.3, B=1000)

write.csv(mrf.output$betas, "/Users/ryanlucas/Desktop/MRF/mrf_betas_1000.csv")
write.csv(mrf.output$pred.ensemble, "/Users/ryanlucas/Desktop/MRF/predictions_1000.csv")

